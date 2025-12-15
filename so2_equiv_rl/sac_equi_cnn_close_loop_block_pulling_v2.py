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
    decode_actions,  # <-- NEW: decode [-1,1] actions to env-scale actions
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
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "so2_equiv_rl"
    wandb_entity: str = None

    # RL / SAC
    total_timesteps: int = 20_000
    num_envs: int = 5
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 1e-2
    batch_size: int = 64
    training_offset: int = 100
    policy_lr: float = 1e-3
    q_lr: float = 1e-3
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    alpha_lr: float = 1e-3
    autotune: bool = True

    # Training
    buffer_aug_n: int = 4
    planner_episode: int = 20

    # Equivariant Network
    equi_n: int = 8
    n_hidden: int = 64

    # Environment Configuration
    workspace_size: float = 0.3
    max_episode_steps: int = 50
    heightmap_size: int = 128
    action_sequence: str = "pxyzr"
    render: bool = False
    num_objects: int = 1
    random_orientation: bool = True
    robot: str = "kuka"
    view_type: str = "camera_center_xyz"
    dpos: float = 0.05
    drot_n: int = 8


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
        # NOTE: `act` is now expected to be UN-SCALED in [-1, 1]
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(
            obs,
            nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]),
        )
        conv_out = self.img_conv(obs_geo)

        dxy = act[:, 1:3]  # unscaled equivariant vector
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)  # unscaled invariants
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
LOG_STD_MIN = -20
EPSILON = 1e-6


class Actor(torch.nn.Module):
    def __init__(
        self,
        obs_shape=(2, 128, 128),
        action_dim=5,
        initialize=True,
        N=4,
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        """
        Returns UN-SCALED action in [-1, 1] for every dimension.
        Env scaling is handled externally via `decode_actions(...)`.
        """
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)  # unscaled action in [-1, 1]
        action_unscaled = y_t

        log_prob = normal.log_prob(x_t)
        # tanh correction (NO action_scale term now)
        log_prob -= torch.log((1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_unscaled = torch.tanh(mean)
        return action_unscaled, log_prob, mean_unscaled


def _decode_for_env(
    action_unscaled: torch.Tensor, *, action_sequence: str, dx, dy, dz, dr
):
    """
    action_unscaled: (B, A) in [-1,1]
    returns env_action: (B, A) in true env scale
    """
    if "r" in action_sequence:
        _, env_action = decode_actions(
            action_unscaled[:, 0],
            action_unscaled[:, 1],
            action_unscaled[:, 2],
            action_unscaled[:, 3],
            action_unscaled[:, 4],
            action_sequence=action_sequence,
            dx=dx,
            dy=dy,
            dz=dz,
            dr=dr,
        )
    else:
        _, env_action = decode_actions(
            action_unscaled[:, 0],
            action_unscaled[:, 1],
            action_unscaled[:, 2],
            action_unscaled[:, 3],
            action_sequence=action_sequence,
            dx=dx,
            dy=dy,
            dz=dz,
            dr=dr,
        )
    return env_action


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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

    dpos = args.dpos
    drot = np.pi / args.drot_n
    planner_config = {
        "random_orientation": args.random_orientation,
        "dpos": dpos,
        "drot": drot,
    }

    envs = EnvWrapper(
        num_processes=args.num_envs,
        env=ENV_NAME,
        env_config=env_config,
        planner_config=planner_config,
    )

    qfs = TwinSoftQNetworks(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
        N=args.equi_n,
        n_hidden=args.n_hidden,
    ).to(device)

    qf_targets = TwinSoftQNetworks(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
        N=args.equi_n,
        n_hidden=args.n_hidden,
    ).to(device)
    qf_targets.load_state_dict(qfs.state_dict())

    actor = Actor(
        obs_shape=(2, args.heightmap_size, args.heightmap_size),
        action_dim=len(ACTION_SEQUENCE),
        N=args.equi_n,
        n_hidden=args.n_hidden,
    ).to(device)

    q_optimizer = optim.Adam(list(qfs.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -len(ACTION_SEQUENCE)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    rb = AugReplayBuffer(
        size=args.buffer_size,
        aug_n=args.buffer_aug_n,
        augment_fn=so2_augment_close_loop_env,
    )

    start_time = time.time()
    states_t, obs_t = envs.reset()

    # -------------------------
    # Planner demos: step env with SCALED actions, store UN-SCALED actions
    # -------------------------
    if args.planner_episode > 0:
        planner_envs = envs
        i = 0
        while i < args.planner_episode:
            planner_actions = planner_envs.getNextAction()

            planner_actions_unscaled, planner_actions_scaled = get_action_from_plan(
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
                    action=planner_actions_unscaled[j].numpy(),  # <-- UN-SCALED stored
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

    # -------------------------
    # Main training loop
    # -------------------------
    global_step = 0
    states_t, obs_t = envs.reset()
    ep_len = np.zeros(args.num_envs, dtype=np.int32)
    ep_ret = np.zeros(args.num_envs, dtype=np.float32)
    ep_disc = np.ones(args.num_envs, dtype=np.float32)

    while global_step < args.total_timesteps:
        global_step += 1

        # get UN-SCALED action from policy in [-1,1]
        with torch.no_grad():
            s_t = torch.as_tensor(states_t, device=device)
            o_t = torch.as_tensor(obs_t, device=device)
            actions_unscaled_t, _, _ = actor.get_action(pack_state_obs(s_t, o_t))

        # decode to env-scale actions for stepping the env
        actions_env_t = _decode_for_env(
            actions_unscaled_t,
            action_sequence=ACTION_SEQUENCE,
            dx=dpos,
            dy=dpos,
            dz=dpos,
            dr=drot,
        )

        # async step
        envs.stepAsync(actions_env_t, auto_reset=True)

        # train
        if len(rb) > args.training_offset:
            data_cpu = rb.sample(args.batch_size, as_tensor=True)
            data = TransitionBatch(
                state=data_cpu.state.to(device),
                obs=denormalize_observation(data_cpu.obs).to(device),
                action=data_cpu.action.to(device),  # <-- UN-SCALED actions from replay
                reward=data_cpu.reward.to(device),
                next_state=data_cpu.next_state.to(device),
                next_obs=denormalize_observation(data_cpu.next_obs).to(device),
                done=data_cpu.done.to(device),
            )

            with torch.no_grad():
                next_state_actions_unscaled_t, next_state_log_pi_t, _ = (
                    actor.get_action(pack_state_obs(data.next_state, data.next_obs))
                )

                qf1_next_target_t, qf2_next_target_t = qf_targets(
                    pack_state_obs(data.next_state, data.next_obs),
                    next_state_actions_unscaled_t,  # <-- UN-SCALED
                )

                min_qf_next_target = (
                    torch.min(qf1_next_target_t, qf2_next_target_t)
                    - alpha * next_state_log_pi_t
                )
                next_q_value_t = data.reward.flatten() + (
                    1 - data.done.flatten()
                ) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values_t, qf2_a_values_t = qfs(
                pack_state_obs(data.state, data.obs), data.action  # <-- UN-SCALED
            )
            qf1_a_values_t = qf1_a_values_t.view(-1)
            qf2_a_values_t = qf2_a_values_t.view(-1)

            qf1_loss = F.mse_loss(qf1_a_values_t, next_q_value_t)
            qf2_loss = F.mse_loss(qf2_a_values_t, next_q_value_t)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi_unscaled_t, log_pi_t, _ = actor.get_action(
                        pack_state_obs(data.state, data.obs)
                    )
                    qf1_pi_t, qf2_pi_t = qfs(
                        pack_state_obs(data.state, data.obs), pi_unscaled_t
                    )
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
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qfs.parameters(), qf_targets.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

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
                    writer.add_scalar("charts/alpha_value", alpha, global_step)
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

        # wait for env results
        next_states_t, next_obs_t, rewards_t, dones_t = envs.stepWait()
        rewards = rewards_t.numpy()
        dones = dones_t.numpy()

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

        # store transition with UN-SCALED action in replay
        for i in range(args.num_envs):
            transition = Transition(
                state=states_t[i].numpy(),
                obs=obs_t[i].numpy(),
                action=actions_unscaled_t[i].cpu().numpy(),  # <-- UN-SCALED stored
                reward=rewards[i],
                next_state=next_states_t[i].numpy(),
                next_obs=next_obs_t[i].numpy(),
                done=dones[i],
            )
            transition = normalize_transition(transition)
            rb.add(transition)

        states_t, obs_t = next_states_t, next_obs_t

    envs.close()
    writer.close()
