
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
import random
import time
from dataclasses import dataclass

import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # key fix

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from escnn import gspaces
from escnn import nn as enn

from cleanrl_utils.buffers import ReplayBuffer

from ple.games.snake import Snake
from ple import PLE
from preprocessing.snake_preprocessing import frame_history

from gymnasium.wrappers import TimeLimit

class SnakeEnv(gym.Env):
    def __init__(self, height=32, width=32, fps=15, frame_history_size=4):
    # create the game environment and initialize the attribute values
        super().__init__()
        self.game = Snake(height=height,width=width,init_length=4)
        reward_dict = {"positive": 1.0, "negative": -1.0, "tick": 0.0, "loss": -1.0, "win": 1.0}
        self.environment = PLE(self.game, fps=fps, reward_values=reward_dict, num_steps=2, display_screen=False)
        self.init_env() # initialize the game
        self.allowed_actions = self.environment.getActionSet() # the list of allowed actions to be taken by an agent
        self.num_actions = len(self.allowed_actions) - 1  # number of actions that are allowed in this env
        self.frame_hist = frame_history(height=height,width=width,frame_history_size=frame_history_size,num_channels=3);
        self.input_shape = self.frame_hist.get_history().shape  # shape of the game input screen

        # Variables for accumulation
        self.episode_length = 0
        self.episode_reward = 0

        # Metadata
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(frame_history_size*3, height, width), dtype=np.uint8)

    def init_env(self):
        # initialize the variables and screen of the game
        self.environment.init()


    def get_current_state(self):
    # get the current state in the game. Returns the current screen of the game with snake and food positions with
    # a sequence of past .
        cur_frame = np.transpose(self.environment.getScreenRGB(),(2,0,1))
        #cur_frame = np.transpose(np.expand_dims(self.environment.getScreenGrayscale(),axis=0), (2, 0, 1))
        self.frame_hist.push(cur_frame)
        return self.frame_hist.get_history()


    def check_game_over(self):
        # check if the game has terminated
        return self.environment.game_over()

    def reset(self, seed=None, options=None):
        # resets the game to initial values and refreshes the screen with a new small snake and random food position.
        self.environment.reset_game()
        _ = self.environment.act(None)
        self.frame_hist.reset(np.transpose(self.environment.getScreenRGB(),(2,0,1)))
        self.episode_reward = 0
        self.episode_length = 0
        return self.frame_hist.get_history(), {}

    def step(self, action):
        # lets the snake take the chosen action of moving in some direction
        reward = self.environment.act(self.allowed_actions[action])
        self.episode_reward += reward
        self.episode_length += 1
        next_state = self.get_current_state()
        done = self.check_game_over()
        info = {}
        if done:
            info['episode'] = {"r" : self.episode_reward, "l" : self.episode_length}
        return next_state, reward, done, False, info

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Snake"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    eval_interval: int = 250_000
    """How often (in env steps) to run evaluation"""
    eval_episodes: int = 50
    """How many evaluation episodes to average over"""
    log_interval: int = 1000
    """log training progress every N environment steps"""
    eval_max_steps: int = 300_000
    """Maximum length of an evaluation training run"""

def make_env():
    def thunk():
        env = SnakeEnv()
        return env
    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(N=4)
        self.input_type = enn.FieldType(r2_act, 12*[r2_act.trivial_repr])
        layer1_type = enn.FieldType(r2_act, 8*[r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, 12*[r2_act.regular_repr])
        layer3_type = enn.FieldType(r2_act, 12*[r2_act.regular_repr])
        layer4_type = enn.FieldType(r2_act, 32*[r2_act.regular_repr])
        
        self.network = enn.SequentialModule(
            enn.R2Conv(self.input_type, layer1_type, kernel_size=7, stride=2, padding=2, bias=False),
            enn.ReLU(layer1_type, inplace=True),
            enn.R2Conv(layer1_type, layer2_type, kernel_size=5, stride=2, padding=1, bias=False),
            enn.ReLU(layer2_type, inplace=True),
            enn.R2Conv(layer2_type, layer3_type, kernel_size=5, stride=1, padding=1, bias=False),
            enn.ReLU(layer3_type, inplace=True),
            enn.R2Conv(layer3_type, layer4_type, kernel_size=7, stride=1, padding=1, bias=False),
            enn.ReLU(layer4_type, inplace=True),
            enn.GroupPooling(layer4_type),
           #nn.Conv2d(12, 32, 7, stride=2, padding=2),
           #nn.ReLU(),
           #nn.Conv2d(32, 64, 5, stride=2, padding=1),
           #nn.ReLU(),
           #nn.Conv2d(64, 64, 3, stride=1, padding=1),
           #nn.ReLU(),
           #nn.Flatten(),
           #nn.Linear(3136, 256),
           #nn.ReLU(),
           #nn.Linear(256, env.single_action_space.n),
        )
        self.head = torch.nn.Linear(self.network.out_type.size, env.single_action_space.n)

    def forward(self, x):
        x = x.view(-1, 12, 32, 32)
        x = self.network(self.input_type(x / 255.0)).tensor
        x = x.view(x.size(0), -1)
        return self.head(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



def evaluate_policy(q_network, eval_env, device, n_episodes: int = 10) -> float:
    """Evaluate with a greedy (argmax-Q) policy and return mean episodic return."""
    returns = []

    q_network.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_ret = 0.0

            while not done:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
                q_values = q_network(obs_t)
                action = int(torch.argmax(q_values, dim=1).item())

                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_ret += float(reward)

            returns.append(ep_ret)

    q_network.train()
    return float(np.mean(returns))



if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env() for i in range(args.num_envs)]
    )
    eval_env = make_env()()
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    eval_env = TimeLimit(eval_env, max_episode_steps=args.eval_max_steps)

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Evaluation Parameters
    next_eval = args.eval_interval
    best_eval_return = -float("inf")
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step % args.log_interval == 0:
            sps = int(global_step / (time.time() - start_time)) if global_step > 0 else 0
            print(f"step={global_step} sps={sps}", flush=True)
            writer.add_scalar("charts/SPS", sps, global_step)
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            # envs.num_envs == 1
            ep_r = infos["episode"]["r"]
            ep_l = infos["episode"]["l"]
            writer.add_scalar("charts/episodic_return", ep_r, global_step)
            writer.add_scalar("charts/episodic_length", ep_l, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                print(trunc, True)
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 20_000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # ---- EVALUATION BLOCK ----
        if global_step > args.learning_starts and global_step >= next_eval:
            eval_return = evaluate_policy(q_network, eval_env, device, n_episodes=args.eval_episodes)
            best_eval_return = max(best_eval_return, eval_return)
            writer.add_scalar("charts/eval_return", eval_return, global_step)
            print(f"[step {global_step}] eval_return = {eval_return:.3f} (best={best_eval_return:.3f})", flush=True)
            next_eval += args.eval_interval


    envs.close()
    if hasattr(eval_env, "close"):
        eval_env.close()
    writer.close()
