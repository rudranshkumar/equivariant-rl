import os
import random
import time
from dataclasses import dataclass

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from so2_equiv_rl.storage.transitions import Transition, TransitionBatch
from so2_equiv_rl.storage.buffers import AugReplayBuffer
from so2_equiv_rl.utils.bulletarm_utils import (
    so2_augment_close_loop_env,
    get_action_from_plan,
    pack_state_obs,
    normalize_transition,
    denormalize_observation,
)
from so2_equiv_rl.utils.env_wrapper import EnvWrapper
from so2_equiv_rl.utils.torch_utils import weights_init

ENV_NAME = "close_loop_block_pulling"
OBS_TYPE = "pixel"
ACTION_SEQUENCE = "pxyzr"


# -------------------------
# Debug helpers (prints only)
# -------------------------
PRINT_ENV_RESET = True
PRINT_ACT_EVERY = 1000
PRINT_RB_ADD_EVERY = 2000
PRINT_SAMPLE_EVERY = 2000
PRINT_TARGET_EVERY = 1000
PRINT_GRAD_EVERY = 2000
PRINT_ALPHA_EVERY = 1000


def _tstats(name, x: torch.Tensor, max_items=5):
    if not isinstance(x, torch.Tensor):
        print(f"[!!] {name}: not a torch.Tensor ({type(x)})")
        return
    x = x.detach()
    finite = torch.isfinite(x)
    if not finite.all():
        print(f"[!!] {name}: non-finite count = {(~finite).sum().item()} / {x.numel()}")
        bad = x[~finite].flatten()[:max_items]
        print(f"     bad values sample: {bad.cpu().numpy()}")
    xf = x[finite]
    if xf.numel() == 0:
        print(f"[!!] {name}: all values non-finite")
        return
    print(
        f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={xf.min().item():.6g} max={xf.max().item():.6g} "
        f"mean={xf.mean().item():.6g} std={xf.std(unbiased=False).item():.6g}"
    )


def _npstats(name, x: np.ndarray, max_items=5):
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.all():
        print(f"[!!] {name}: non-finite count = {np.sum(~finite)} / {x.size}")
        bad = x[~finite].ravel()[:max_items]
        print(f"     bad values sample: {bad}")
    xf = x[finite]
    if xf.size == 0:
        print(f"[!!] {name}: all values non-finite")
        return
    print(
        f"[{name}] shape={x.shape} dtype={x.dtype} "
        f"min={xf.min():.6g} max={xf.max():.6g} "
        f"mean={xf.mean():.6g} std={xf.std():.6g}"
    )


def _grad_l2_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isfinite(g).all():
            total += g.norm(2).item() ** 2
        else:
            # still include finite part if possible
            gf = g[torch.isfinite(g)]
            if gf.numel() > 0:
                total += gf.norm(2).item() ** 2
    return total**0.5


@dataclass
class Args:

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "so2_equiv_rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # RL / SAC
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor args.gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    alpha_lr: float = 1e-3
    """the learning rate of the temperature coefficient optimizer"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Training
    buffer_aug_n: int = 4
    """number of augmentations per transition"""
    planner_episode: int = 20
    """number of demonstrations to gather before training"""

    # Environment Configuration
    workspace_size: float = 0.3
    """size of the workspace in meters"""
    max_episode_steps: int = 50
    """maximum number of steps per episode"""
    heightmap_size: int = 128
    """size of the heightmap in pixels"""
    action_sequence: str = "pxyzr"
    """the action space"""
    render: bool = False
    """whether to render the PyBullet environment"""
    num_objects: int = 1
    """the number of objects in the environment"""
    random_orientation: bool = True
    """whether to allow the environment to initialize with random orientations"""
    robot: str = "kuka"
    """the robot to use in the environment"""
    view_type: str = "camera_center_xyz"
    """the view type for the observation"""
    dpos: float = 0.05
    """the maximal positional delta"""
    drot_n: int = 4
    """the maximal rotational delta is pi / drot_n"""


class SACEncoder(nn.Module):
    """Encoder for SAC TwinSoftQ and Actor networks."""

    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(256, out_dim, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.conv(x)


class TwinSoftQNetworks(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.encoder = SACEncoder(obs_shape, 128)

        self.qf1 = torch.nn.Sequential(
            torch.nn.Linear(128 + action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
        )

        self.qf2 = torch.nn.Sequential(
            torch.nn.Linear(128 + action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
        )

        self.apply(weights_init)

    def forward(self, obs, action):
        x = self.encoder(obs)
        qf1_out = self.qf1(torch.cat([x, action], 1))
        qf2_out = self.qf2(torch.cat([x, action], 1))
        return qf1_out, qf2_out


LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Wang et al. (ICLR 2022) use -20 instead of CleanRL, which uses -5
EPSILON = 1e-6


class Actor(nn.Module):

    def __init__(
        self,
        obs_shape=(2, 128, 128),
        action_dim=5,
        dx=0.05,
        dy=0.05,
        dz=0.05,
        dr=np.pi / 4,
    ):
        super().__init__()

        self.encoder = SACEncoder(obs_shape, 128)
        self.fc_mean = nn.Linear(128, action_dim)
        self.fc_logstd = nn.Linear(128, action_dim)

        action_space_low = np.array([0, -dx, -dy, -dz, -dr])
        action_space_high = np.array([1, dx, dy, dz, dr])

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space_high - action_space_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space_high + action_space_low) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.apply(weights_init)

    def forward(self, obs):
        x = self.encoder(obs)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    # parse the args
    args = tyro.cli(Args)

    # create run name
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # wandb setup
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # tensorboard setup
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # torch setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(
        f"[setup] device={device} cuda_available={torch.cuda.is_available()} num_envs={args.num_envs}"
    )

    # env config setup
    workspace = np.asarray(
        [
            [0.45 - args.workspace_size / 2, 0.45 + args.workspace_size / 2],
            [0 - args.workspace_size / 2, 0 + args.workspace_size / 2],
            [0.01, 0.25],
        ]
    )
    env_config = {
        "workspace": workspace,
        "max_steps": args.max_episode_steps,
        "obs_size": args.heightmap_size,
        "action_sequence": ACTION_SEQUENCE,
        "render": args.render,
        "num_objects": args.num_objects,
        "random_orientation": args.random_orientation,
        "robot": args.robot,
        "view_type": args.view_type,
        "obs_type": OBS_TYPE,
        "close_loop_tray": False,
        "seed": args.seed,
    }

    # planner config setup
    dpos = args.dpos
    drot = np.pi / args.drot_n
    planner_config = {
        "random_orientation": args.random_orientation,
        "dpos": dpos,
        "drot": drot,
    }

    # env construction
    envs = EnvWrapper(
        num_processes=args.num_envs,
        env=ENV_NAME,
        env_config=env_config,
        planner_config=planner_config,
    )

    # twin soft Q networks construction
    qfs = TwinSoftQNetworks(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
    ).to(device)

    # target twin soft Q networks construction
    qf_targets = TwinSoftQNetworks(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
    ).to(device)
    qf_targets.load_state_dict(qfs.state_dict())

    # actor network construction
    actor = Actor(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
        dx=dpos,
        dy=dpos,
        dz=dpos,
        dr=drot,
    ).to(device)

    # optimizers setup
    q_optimizer = optim.Adam(list(qfs.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -len(ACTION_SEQUENCE)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
        print(
            f"[setup] autotune=True target_entropy={target_entropy} init_alpha={alpha} alpha_lr={args.alpha_lr}"
        )
    else:
        alpha = args.alpha
        print(f"[setup] autotune=False fixed_alpha={alpha}")

    # reply buffer setup
    rb = AugReplayBuffer(
        size=args.buffer_size,
        aug_n=args.buffer_aug_n,
        augment_fn=so2_augment_close_loop_env,
    )
    print(
        f"[setup] replay_buffer size={args.buffer_size} aug_n={args.buffer_aug_n} augment_fn={so2_augment_close_loop_env.__name__}"
    )

    start_time = time.time()
    states_t, obs_t = envs.reset()

    if PRINT_ENV_RESET:
        print("=== ENV RESET (initial) ===")
        _npstats("states_t (np)", np.asarray(states_t))
        _npstats("obs_t (np)", np.asarray(obs_t))
        print("===========================")

    # collect initial demonstrations with planner
    if args.planner_episode > 0:
        planner_envs = envs
        i = 0
        while i < args.planner_episode:
            planner_actions = planner_envs.getNextAction()
            _, planner_actions_scaled = get_action_from_plan(
                planner_actions,
                action_sequence=ACTION_SEQUENCE,
                dx=dpos,
                dy=dpos,
                dz=dpos,
                dr=drot,
            )
            next_states_t, next_obs_t, rewards_t, dones_t = planner_envs.step(
                planner_actions_scaled, auto_reset=True
            )
            for j in range(args.num_envs):
                transition = Transition(
                    state=states_t[j].numpy(),
                    obs=obs_t[j].numpy(),
                    action=planner_actions_scaled[j].numpy(),
                    reward=rewards_t[j].numpy(),
                    next_state=next_states_t[j].numpy(),
                    next_obs=next_obs_t[j].numpy(),
                    done=dones_t[j].numpy(),
                )
                if PRINT_RB_ADD_EVERY > 0 and (i == 0 and j == 0):
                    print("=== DEMO ADD (first transition) ===")
                    _npstats("demo raw obs", transition.obs)
                    _npstats("demo raw action", transition.action)
                transition = normalize_transition(transition)
                if PRINT_RB_ADD_EVERY > 0 and (i == 0 and j == 0):
                    _npstats("demo normalized obs", transition.obs)
                    print("===================================")
                rb.add(transition)
            states_t = copy.copy(next_states_t)
            obs_t = copy.copy(next_obs_t)

            i += dones_t.sum().item()

    # main training loop
    global_step = 0
    states_t, obs_t = envs.reset()

    if PRINT_ENV_RESET:
        print("=== ENV RESET (training) ===")
        _npstats("states_t (np)", np.asarray(states_t))
        _npstats("obs_t (np)", np.asarray(obs_t))
        print("============================")

    ep_len = np.zeros(args.num_envs, dtype=np.int32)
    ep_ret = np.zeros(args.num_envs, dtype=np.float32)  # running discounted return
    ep_disc = np.ones(args.num_envs, dtype=np.float32)  # running args.gamma^t

    while global_step < args.total_timesteps:

        global_step += args.num_envs

        # get action from policy
        with torch.no_grad():
            s_t = torch.as_tensor(states_t, device=device)
            o_t = torch.as_tensor(obs_t, device=device)
            packed_t = pack_state_obs(s_t, o_t)
            actions_t, logp_act_t, mean_act_t = actor.get_action(packed_t)

        if PRINT_ACT_EVERY > 0 and global_step % PRINT_ACT_EVERY == 0:
            print(f"\n=== ACT STEP {global_step} ===")
            _tstats("s_t", s_t)
            _tstats("o_t", o_t)
            _tstats("packed_t", packed_t)
            _tstats("actions_t", actions_t)
            _tstats("logp_act_t", logp_act_t)
            _tstats("mean_act_t", mean_act_t)
            with torch.no_grad():
                _, ls_dbg = actor(packed_t)
            _tstats("log_std_used", ls_dbg)
            _tstats("std_used", ls_dbg.exp())
            print("============================\n")

        # step the env with the action selected by the policy
        next_states_t, next_obs_t, rewards_t, dones_t = envs.step(
            actions_t, auto_reset=True
        )
        rewards = rewards_t.numpy()
        dones = dones_t.numpy()

        # logging episodic return and length
        ep_len += 1
        ep_ret += ep_disc * rewards
        ep_disc *= args.gamma

        done_mask = dones.astype(bool)
        if done_mask.any():
            mean_ep_ret = np.mean(ep_ret[done_mask])
            mean_ep_len = np.mean(ep_len[done_mask])
            print(
                f"global_step={global_step}, episodic_return={mean_ep_ret}, episodic_len={mean_ep_len}"
            )
            writer.add_scalar("charts/episodic_return", mean_ep_ret, global_step)
            writer.add_scalar("charts/episodic_length", mean_ep_len, global_step)
            ep_len[done_mask] = 0
            ep_ret[done_mask] = 0.0
            ep_disc[done_mask] = 1.0

        # add transitions to replay buffer
        for i in range(args.num_envs):
            transition = Transition(
                state=states_t[i].numpy(),
                obs=obs_t[i].numpy(),
                action=actions_t[i].cpu().numpy(),
                reward=rewards[i],
                next_state=next_states_t[i].numpy(),
                next_obs=next_obs_t[i].numpy(),
                done=dones[i],
            )

            if (
                PRINT_RB_ADD_EVERY > 0
                and global_step % PRINT_RB_ADD_EVERY == 0
                and i == 0
            ):
                print(f"\n=== REPLAY ADD step {global_step} (env {i}) ===")
                _npstats("raw obs", transition.obs)
                _npstats("raw next_obs", transition.next_obs)
                _npstats("raw action", transition.action)
                _npstats("raw reward", np.array([transition.reward]))
                _npstats("raw done", np.array([transition.done]))

            transition = normalize_transition(transition)

            if (
                PRINT_RB_ADD_EVERY > 0
                and global_step % PRINT_RB_ADD_EVERY == 0
                and i == 0
            ):
                _npstats("normalized obs", transition.obs)
                _npstats("normalized next_obs", transition.next_obs)
                print("====================================\n")

            rb.add(transition)

        states_t, obs_t = next_states_t, next_obs_t

        # sample batch
        data_cpu = rb.sample(args.batch_size, as_tensor=True)
        data = TransitionBatch(
            state=data_cpu.state.to(device),
            obs=denormalize_observation(data_cpu.obs).to(device),
            action=data_cpu.action.to(device),
            reward=data_cpu.reward.to(device),
            next_state=data_cpu.next_state.to(device),
            next_obs=denormalize_observation(data_cpu.next_obs).to(device),
            done=data_cpu.done.to(device),
        )

        if PRINT_SAMPLE_EVERY > 0 and global_step % PRINT_SAMPLE_EVERY == 0:
            print(f"\n=== REPLAY SAMPLE step {global_step} ===")
            _tstats("data_cpu.obs (stored)", data_cpu.obs)
            _tstats("data.obs (denorm)", data.obs)
            _tstats("data.next_obs (denorm)", data.next_obs)
            _tstats("data.state", data.state)
            _tstats("data.action", data.action)
            _tstats("data.reward", data.reward)
            _tstats("data.done", data.done)
            print("=====================================\n")

        with torch.no_grad():

            # get policy actions for next t+1
            next_state_actions_t, next_state_log_pi_t, _ = actor.get_action(
                pack_state_obs(data.next_state, data.next_obs)
            )

            # compute the target Q value
            qf1_next_target_t, qf2_next_target_t = qf_targets(
                pack_state_obs(data.next_state, data.next_obs),
                next_state_actions_t,
            )
            min_qf_next_target = (
                torch.min(qf1_next_target_t, qf2_next_target_t)
                - alpha * next_state_log_pi_t
            )
            next_q_value_t = data.reward.flatten() + (
                1 - data.done.flatten()
            ) * args.gamma * (min_qf_next_target).view(-1)

            if PRINT_TARGET_EVERY > 0 and global_step % PRINT_TARGET_EVERY == 0:
                print(f"\n=== CRITIC TARGET step {global_step} ===")
                _tstats("next_state_actions_t", next_state_actions_t)
                _tstats("next_state_log_pi_t", next_state_log_pi_t)
                _tstats("qf1_next_target_t", qf1_next_target_t)
                _tstats("qf2_next_target_t", qf2_next_target_t)
                _tstats("min_qf_next_target", min_qf_next_target)
                _tstats("next_q_value_t", next_q_value_t)
                print("=======================================\n")

        qf1_a_values_t, qf2_a_values_t = qfs(
            pack_state_obs(data.state, data.obs), data.action
        )
        qf1_a_values_t, qf2_a_values_t = qf1_a_values_t.view(-1), qf2_a_values_t.view(
            -1
        )
        qf1_loss = F.mse_loss(qf1_a_values_t, next_q_value_t)
        qf2_loss = F.mse_loss(qf2_a_values_t, next_q_value_t)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()

        if PRINT_GRAD_EVERY > 0 and global_step % PRINT_GRAD_EVERY == 0:
            print(
                f"[grad] step {global_step} qfs grad L2 norm: {_grad_l2_norm(qfs):.6g}"
            )
            # also print Q/value stats
            _tstats("qf1_a_values_t", qf1_a_values_t)
            _tstats("qf2_a_values_t", qf2_a_values_t)
            _tstats("qf_loss", qf_loss)

        q_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD 3 delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi_t, log_pi_t, _ = actor.get_action(
                    pack_state_obs(data.state, data.obs)
                )
                qf1_pi_t, qf2_pi_t = qfs(pack_state_obs(data.state, data.obs), pi_t)
                min_qf_pi_t = torch.min(qf1_pi_t, qf2_pi_t)
                actor_loss = ((alpha * log_pi_t) - min_qf_pi_t).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()

                if PRINT_GRAD_EVERY > 0 and global_step % PRINT_GRAD_EVERY == 0:
                    print(
                        f"[grad] step {global_step} actor grad L2 norm: {_grad_l2_norm(actor):.6g}"
                    )
                    _tstats("actor_loss", actor_loss)
                    _tstats("log_pi_t", log_pi_t)
                    _tstats("min_qf_pi_t", min_qf_pi_t)
                    # inspect std used for this batch
                    with torch.no_grad():
                        _, ls_dbg2 = actor(pack_state_obs(data.state, data.obs))
                    _tstats("log_std_used (train batch)", ls_dbg2)

                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(
                            pack_state_obs(data.state, data.obs)
                        )

                    # NOTE: prints only; leaving your current alpha_loss form unchanged
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                    if PRINT_ALPHA_EVERY > 0 and global_step % PRINT_ALPHA_EVERY == 0:
                        with torch.no_grad():
                            entropy = (-log_pi).mean().item()
                            entropy_err = (log_pi + target_entropy).mean().item()
                        print(
                            f"[alpha] step={global_step} alpha={alpha:.6g} "
                            f"target_entropy={target_entropy:.6g} "
                            f"entropy={entropy:.6g} entropy_error_mean={entropy_err:.6g} "
                            f"alpha_loss={alpha_loss.item():.6g} log_alpha={log_alpha.item():.6g}"
                        )

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qfs.parameters(), qf_targets.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        # log training stats
        if global_step % 100 == 0:
            writer.add_scalar(
                "losses/qf1_values", qf1_a_values_t.mean().item(), global_step
            )
            writer.add_scalar(
                "losses/qf2_values", qf2_a_values_t.mean().item(), global_step
            )
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
