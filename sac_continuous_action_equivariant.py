# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate (harmless elsewhere)

import optuna

import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # key fix

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from buffers import ReplayBuffer

from escnn import gspaces
from escnn import nn as enn


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

@dataclass
class Args:
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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "FetchReachDense-v4"
    """the environment id of the task"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
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
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    dihedral_N: int = 4
    """The resolution of the dihedral group used for the symmetries"""
    reg_rep_N: int = 16
    """The number of regular rep features in the channel space for the hidden MLP layers"""
    eval_n_episodes: int = 25
    """Number of evaluation episodes in a test"""

class FetchObsWrapper(gym.ObservationWrapper):
    """
    Converts absolute coordinates to relative ones depending on the Fetch task.
    
    env_name:
        "FetchReach"
        "FetchPush"
        "FetchPickAndPlace"
        "FetchSlide"
    """

    def __init__(self, env, env_name: str):
        super().__init__(env)
        self.env_name = env_name

        # All Fetch envs expose dict obs
        base_obs_space = env.observation_space["observation"]
        goal_space = env.observation_space["desired_goal"]

        # --- Compute final observation dimension dynamically ---
        self.obs_dim = self._compute_obs_dim()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # Step 1: Compute the OUTPUT dimension based on env_name
    # ---------------------------------------------------------------------
    def _compute_obs_dim(self):
        if "FetchReach" in self.env_name:
            # goal_rel (3) + end_effector_vel (3)
            return 6

        if "FetchPush" in self.env_name or "FetchPickAndPlace" in self.env_name or "FetchSlide" in self.env_name:
            # goal_rel (3) + object_rel (3) + ee_vel (3) + object_rotation(3) + object_velocity(3)
            return 19
        
        raise ValueError(f"Unknown Fetch task: {self.env_name}")

    # ---------------------------------------------------------------------
    # Step 2: Compute RELATIVE observations at runtime
    # ---------------------------------------------------------------------
    def observation(self, obs):
        o = obs["observation"]
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        if "FetchReach" in self.env_name:
            ee_abs = o[0:3]
            goal_rel = ee_abs - desired
            ee_vel = o[5:8]
            return np.concatenate([goal_rel, ee_vel]).astype(np.float32)

        if "FetchPush" in self.env_name or "FetchPickAndPlace" in self.env_name or "FetchSlide" in self.env_name:
            ee_abs = o[0:3]
            goal_rel = ee_abs - desired
            object_rel = o[6:9]
            ee_vel = o[20:23]
            grip_pos = o[9:11]
            grip_vel = o[23:]
            object_rot = o[11:14]
            object_vel = o[14:17]
            return np.concatenate([goal_rel, 
                                   object_rel, 
                                   ee_vel, 
                                   object_rot,
                                   object_vel,
                                   grip_pos, 
                                   grip_vel]).astype(np.float32)

        raise ValueError(f"Unknown Fetch task: {self.env_name}")



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FetchObsWrapper(env, env_id)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, N=4, N_rr=16):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(N)
        act_repr_list = [r2_act.irrep(1, 1)] + 2*[r2_act.trivial_repr]
        if env.observation_space.shape[1] == 19:
            obs_repr_list = 5*[r2_act.irrep(1, 1), r2_act.trivial_repr] + 4*[r2_act.trivial_repr]
            self.input_type = enn.FieldType(r2_act, obs_repr_list + act_repr_list)
        else:
            obs_repr_list = 2*[r2_act.irrep(1, 1), r2_act.trivial_repr]
            self.input_type = enn.FieldType(r2_act, obs_repr_list + act_repr_list)
        layer1_type = enn.FieldType(r2_act, N_rr*[r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, N_rr*[r2_act.regular_repr])
        output_type = enn.FieldType(r2_act, 1*[r2_act.trivial_repr])


        in_dim =  np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
        self.net = nn.Sequential(
                enn.R2Conv(self.input_type, layer1_type, kernel_size=1, stride=1, padding=0, initialize=True),
                enn.ReLU(layer1_type),
                enn.R2Conv(layer1_type, layer2_type, kernel_size=1, stride=1, padding=0, initialize=True),
                enn.ReLU(layer2_type),
                enn.R2Conv(layer2_type, output_type, kernel_size=1, stride=1, padding=0, initialize=True)
                )
       #self.net = nn.Sequential(
       #        nn.Linear(in_dim, 256),
       #        nn.ReLU(),
       #        nn.Linear(256, 256),
       #        nn.ReLU(),
       #        nn.Linear(256, 1)
       #        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        b = x.shape[0]
        x = x.view(b, -1, 1, 1)
        x = self.input_type(x)
        x = self.net(x).tensor.view(b, -1)
       #x = F.relu(self.fc1(x))
       #x = F.relu(self.fc2(x))
       #x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, N=4, N_rr=16):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(N)
        act_repr_list = [r2_act.irrep(1, 1)] + 2*[r2_act.trivial_repr]
        if env.observation_space.shape[1] == 19:
            obs_repr_list = 5*[r2_act.irrep(1, 1), r2_act.trivial_repr] + 4*[r2_act.trivial_repr]
            self.input_type = enn.FieldType(r2_act, obs_repr_list)
        else:
            obs_repr_list = 2*[r2_act.irrep(1, 1), r2_act.trivial_repr]
            self.input_type = enn.FieldType(r2_act, obs_repr_list)
        layer1_type = enn.FieldType(r2_act, N_rr*[r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, N_rr*[r2_act.regular_repr])
        output_type = enn.FieldType(r2_act, act_repr_list)

        self.fc1 = enn.R2Conv(self.input_type,layer1_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc1_relu = enn.ReLU(layer1_type)
        self.fc2 = enn.R2Conv(layer1_type,layer2_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc2_relu = enn.ReLU(layer2_type)
        self.fc_mean = enn.R2Conv(layer2_type,output_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc_logstd = enn.R2Conv(layer2_type,output_type, kernel_size=1, stride=1, padding=0, initialize=True)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        b = x.shape[0]
        x = x.reshape(b, -1, 1, 1)
        x = self.input_type(x)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))
        mean = self.fc_mean(x).tensor.view(b,-1)
        log_std = self.fc_logstd(x).tensor.view(b, -1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def evaluate_policy(actor, eval_env, device, n_episodes: int = 10) -> float:
    """
    Run the current policy for n_episodes with a deterministic policy (mean action)
    and return mean episodic return.
    """

    returns = []

    with torch.no_grad():
        actor.eval()
        for _ in range(n_episodes):
            done = False
            ep_ret = 0.0
            obs, _ = eval_env.reset()
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # deterministic: use mean, not sampled action
                _, _, mean_action = actor.get_action(obs_t)
                action = mean_action.detach().cpu().numpy()[0]
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_ret += float(reward)
            returns.append(ep_ret)

    return float(np.mean(returns))


def train_and_eval(args: Args, trial: optuna.Trial | None = None) -> float:
    best_eval_return = -float("inf")
    eval_interval = 50_000

    #args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #Eval Env
    eval_env = gym.make(args.env_id)
    eval_env = FetchObsWrapper(eval_env, args.env_id)

    # env setup
    ctx = mp.get_context("spawn")
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, args.dihedral_N, args.reg_rep_N).to(device)
    qf1 = SoftQNetwork(envs, args.dihedral_N, args.reg_rep_N).to(device)
    qf2 = SoftQNetwork(envs, args.dihedral_N, args.reg_rep_N).to(device)
    qf1_target = SoftQNetwork(envs, args.dihedral_N, args.reg_rep_N).to(device)
    qf2_target = SoftQNetwork(envs, args.dihedral_N, args.reg_rep_N).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    av_r = 0
    r_num = 0
    eval_step = eval_interval

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            mask = infos["_episode"]
            ended = np.nonzero(mask)[0]
            for i in ended:
                r = infos["episode"]["r"][i]
                av_r += (r - av_r)/(r_num+1)
                r_num += 1


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                #print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar("charts/episodic_return", av_r, global_step)
                #print("Average Reward",flush=True)
                #print(av_r, flush=True)
                av_r = 0
                r_num = 0
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                writer.flush()

        # ---- EVALUATION BLOCK ----
        if global_step > 0 and global_step  >= eval_step:
            eval_return = evaluate_policy(actor, eval_env, device, n_episodes=args.eval_n_episodes)
            best_eval_return = max(best_eval_return, eval_return)

            writer.add_scalar("charts/eval_return", eval_return, global_step)
            print(f"[step {global_step}] eval_return = {eval_return:.3f}", flush=True)

            eval_step += eval_interval
            writer.flush()


    eval_env.close()
    envs.close()
    writer.close()
    return best_eval_return

if __name__ == "__main__":
    args = tyro.cli(Args)
    train_and_eval(args)
