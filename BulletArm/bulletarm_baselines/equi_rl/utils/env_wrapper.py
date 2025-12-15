import torch
from bulletarm import env_factory
from bulletarm_baselines.equi_rl.utils.parameters import (
    symbreak_pad,
    symbreak_xoffset,
)

import torch
import torch.nn.functional as F


def pad_center_with_xoffset(
    x: torch.Tensor, pad: int, xoffset: int = 0
) -> torch.Tensor:
    """
    Zero-pad a tensor to (H' = H + pad, W' = W + pad), placing the original image
    centered when xoffset=0, and shifted by xoffset along width when non-zero.

    Accepts x shaped (H,W), (1,H,W), or (B,1,H,W).
    Returns the same number of dims as input, but with H', W'.

    xoffset > 0 shifts the image to the right (increasing width index).
    """
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")

    orig_dim = x.dim()
    if orig_dim not in (2, 3, 4):
        raise ValueError(f"Expected x with 2, 3, or 4 dims; got shape {tuple(x.shape)}")

    # Normalize to (B, C, H, W)
    if orig_dim == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif orig_dim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W) for 3D input; got {tuple(x.shape)}")
        x4 = x.unsqueeze(0)  # (1,1,H,W)
    else:  # orig_dim == 4
        if x.shape[1] != 1:
            raise ValueError(f"Expected (B,1,H,W) for 4D input; got {tuple(x.shape)}")
        x4 = x

    # Split total extra pixels (pad) across both sides
    top = pad // 2
    bottom = pad - top

    base_left = pad // 2
    left = base_left + xoffset
    if left < 0 or left > pad:
        raise ValueError(
            f"xoffset={xoffset} makes placement invalid: left padding would be {left}, "
            f"but must be in [0, {pad}] to keep final W' = W + pad."
        )
    right = pad - left

    y4 = F.pad(x4, (left, right, top, bottom), mode="constant", value=0)

    # Restore original dimensionality
    if orig_dim == 2:
        return y4[0, 0]
    if orig_dim == 3:
        return y4[0]
    return y4


class EnvWrapper:
    def __init__(self, num_processes, env, env_config, planner_config):
        self.envs = env_factory.createEnvs(
            num_processes, env, env_config, planner_config
        )

    def reset(self):
        (states, in_hands, obs) = self.envs.reset()
        states = torch.tensor(states).float()
        if symbreak_pad > 0:
            obs = pad_center_with_xoffset(
                torch.tensor(obs).float(), symbreak_pad, symbreak_xoffset
            )
        else:
            obs = torch.tensor(obs).float()
        return states, obs

    def getNextAction(self):
        return torch.tensor(self.envs.getNextAction()).float()

    def step(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        (states_, in_hands_, obs_), rewards, dones = self.envs.step(actions, auto_reset)
        states_ = torch.tensor(states_).float()
        if symbreak_pad > 0:
            obs_ = pad_center_with_xoffset(
                torch.tensor(obs_).float(), symbreak_pad, symbreak_xoffset
            )
        else:
            obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, obs_, rewards, dones

    def stepAsync(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        self.envs.stepAsync(actions, auto_reset)

    def stepWait(self):
        (states_, in_hands_, obs_), rewards, dones = self.envs.stepWait()
        states_ = torch.tensor(states_).float()
        if symbreak_pad > 0:
            obs_ = pad_center_with_xoffset(
                torch.tensor(obs_).float(), symbreak_pad, symbreak_xoffset
            )
        else:
            obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, obs_, rewards, dones

    def getStepLeft(self):
        return torch.tensor(self.envs.getStepsLeft()).float()

    def reset_envs(self, env_nums):
        states, in_hands, obs = self.envs.reset_envs(env_nums)
        states = torch.tensor(states).float()
        if symbreak_pad > 0:
            obs = pad_center_with_xoffset(
                torch.tensor(obs).float(), symbreak_pad, symbreak_xoffset
            )
        else:
            obs = torch.tensor(obs).float()
        return states, obs

    def close(self):
        self.envs.close()

    def saveToFile(self, envs_save_path):
        return self.envs.saveToFile(envs_save_path)

    def getEnvGitHash(self):
        return self.envs.getEnvGitHash()

    def getEmptyInHand(self):
        return torch.tensor(self.envs.getEmptyInHand()).float()
