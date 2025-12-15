import os
import random
import time
from dataclasses import dataclass

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from escnn import gspaces
from escnn import nn

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

ENV_NAME = "close_loop_block_pulling"
OBS_TYPE = "pixel"
ACTION_SEQUENCE = "pxyzr"


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
    total_timesteps: int = 20_000
    """total timesteps of the experiments"""
    num_envs: int = 5
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor args.gamma"""
    tau: float = 1e-2
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
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

    # Equivariant Network
    equi_n: int = 8
    """the N in C_N equivariant networks"""

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
    drot_n: int = 8
    """the maximal rotational delta is pi / drot_n"""


class EquivariantSACEncoder(torch.nn.Module):

    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(
                nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.PointwiseMaxPool(
                nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), 2
            ),
            # 64x64
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.PointwiseMaxPool(
                nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]), 2
            ),
            # 32x32
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.PointwiseMaxPool(
                nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]), 2
            ),
            # 16x16
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.PointwiseMaxPool(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2
            ),
            # 8x8
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.PointwiseMaxPool(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2
            ),
            # 3x3
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)


class TwinSoftQNetworks(torch.nn.Module):

    def __init__(
        self,
        obs_shape=(2, 128, 128),
        action_dim=5,
        n_hidden=128,
        initialize=True,
        N=4,
    ):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.rot2dOnR2(N)
        self.img_conv = EquivariantSACEncoder(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N == 2 else 1

        self.critic_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(
                nn.FieldType(
                    self.c4_act,
                    n_hidden * [self.c4_act.regular_repr]
                    + (action_dim - 2) * [self.c4_act.trivial_repr]
                    + self.n_rho1 * [self.c4_act.irrep(1)],
                ),
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                kernel_size=1,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.GroupPooling(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])
            ),
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                kernel_size=1,
                padding=0,
                initialize=initialize,
            ),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(
                nn.FieldType(
                    self.c4_act,
                    n_hidden * [self.c4_act.regular_repr]
                    + (action_dim - 2) * [self.c4_act.trivial_repr]
                    + self.n_rho1 * [self.c4_act.irrep(1)],
                ),
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                kernel_size=1,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                inplace=True,
            ),
            nn.GroupPooling(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])
            ),
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                kernel_size=1,
                padding=0,
                initialize=initialize,
            ),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(
            obs,
            nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]),
        )
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat(
            (
                conv_out.tensor,
                inv_act.reshape(batch_size, n_inv, 1, 1),
                dxy.reshape(batch_size, 2, 1, 1),
            ),
            dim=1,
        )
        cat_geo = nn.GeometricTensor(
            cat,
            nn.FieldType(
                self.c4_act,
                self.n_hidden * [self.c4_act.regular_repr]
                + n_inv * [self.c4_act.trivial_repr]
                + self.n_rho1 * [self.c4_act.irrep(1)],
            ),
        )
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2


LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Wang et al. (ICLR 2022) use -20 instead of CleanRL, which uses -5
EPSILON = 1e-6


class Actor(torch.nn.Module):

    def __init__(
        self,
        obs_shape=(2, 128, 128),
        action_dim=5,
        initialize=True,
        N=4,
        dx=0.05,
        dy=0.05,
        dz=0.05,
        dr=np.pi / 4,
        n_hidden=128,
    ):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.rot2dOnR2(N)
        self.n_rho1 = 2 if N == 2 else 1
        self.conv = torch.nn.Sequential(
            EquivariantSACEncoder(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                # mixed representation including action_dim trivial representations (for the std of all actions),
                # (action_dim-2) trivial representations (for the mu of invariant actions),
                # and 1 standard representation (for the mu of equivariant actions)
                nn.FieldType(
                    self.c4_act,
                    self.n_rho1 * [self.c4_act.irrep(1)]
                    + (action_dim * 2 - 2) * [self.c4_act.trivial_repr],
                ),
                kernel_size=1,
                padding=0,
                initialize=initialize,
            ),
        )

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

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(
            obs,
            nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]),
        )
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2 : self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim :]
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
        N=args.equi_n,
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
    else:
        alpha = args.alpha

    # reply buffer setup
    rb = AugReplayBuffer(
        size=args.buffer_size,
        aug_n=args.buffer_aug_n,
        augment_fn=so2_augment_close_loop_env,
    )

    start_time = time.time()
    states_t, obs_t = envs.reset()

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
                transition = normalize_transition(transition)
                rb.add(transition)
            states_t = copy.copy(next_states_t)
            obs_t = copy.copy(next_obs_t)

            i += dones_t.sum().item()

    # main training loop
    global_step = 0
    states_t, obs_t = envs.reset()
    ep_len = np.zeros(args.num_envs, dtype=np.int32)
    ep_ret = np.zeros(args.num_envs, dtype=np.float32)  # running discounted return
    ep_disc = np.ones(args.num_envs, dtype=np.float32)  # running args.gamma^t

    while global_step < args.total_timesteps:

        global_step += 1

        # get action from policy
        with torch.no_grad():
            s_t = torch.as_tensor(states_t, device=device)
            o_t = torch.as_tensor(obs_t, device=device)
            actions_t, _, _ = actor.get_action(pack_state_obs(s_t, o_t))

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
            print(f"global_step={global_step}, episodic_return={mean_ep_ret}")
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
            transition = normalize_transition(transition)
            rb.add(transition)

        states_t, obs_t = next_states_t, next_obs_t

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
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(
                            pack_state_obs(data.state, data.obs)
                        )
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

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
