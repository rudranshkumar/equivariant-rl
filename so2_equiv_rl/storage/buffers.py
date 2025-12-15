from typing import Callable
import numpy as np
import numpy.random as npr
import torch
from so2_equiv_rl.storage.transitions import Transition, TransitionBatch


class ReplayBuffer:

    def __init__(self, size: int):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, key) -> Transition:
        return self._storage[key]

    def __setitem__(self, key, value: Transition) -> None:
        self._storage[key] = value

    def add(self, data: Transition) -> None:

        if len(self._storage) < self._max_size:
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(
        self, batch_size: int, as_tensor: bool = False
    ) -> list[Transition] | TransitionBatch:

        n = len(self)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_indexes = npr.choice(n, batch_size).tolist()
        batch = [self._storage[idx] for idx in batch_indexes]

        if not as_tensor:
            return batch

        def _stack(xs):
            arr0 = np.asarray(xs[0])
            if arr0.shape == ():
                return np.asarray(xs)
            return np.stack(xs, axis=0)

        states = [t.state for t in batch]
        obs = [t.obs for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        next_obs = [t.next_obs for t in batch]
        dones = [t.done for t in batch]

        states_np = _stack(states)
        obs_np = _stack(obs)
        actions_np = _stack(actions)
        rewards_np = _stack(rewards)
        next_states_np = _stack(next_states)
        next_obs_np = _stack(next_obs)
        dones_np = _stack(dones)

        # ensure obs has a channel dim: (B,H,W) -> (B,1,H,W)
        if obs_np.ndim == 3:
            obs_np = obs_np[:, None, :, :]
        if next_obs_np.ndim == 3:
            next_obs_np = next_obs_np[:, None, :, :]

        # ensure rewards/dones are (B,)
        rewards_np = np.asarray(rewards_np).reshape(-1)
        dones_np = np.asarray(dones_np).reshape(-1)

        states_t = torch.as_tensor(states_np, dtype=torch.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        actions_t = torch.as_tensor(actions_np, dtype=torch.float32)
        rewards_t = torch.as_tensor(rewards_np, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32)
        dones_t = torch.as_tensor(dones_np, dtype=torch.float32)

        return TransitionBatch(
            state=states_t,
            obs=obs_t,
            action=actions_t,
            reward=rewards_t,
            next_state=next_states_t,
            next_obs=next_obs_t,
            done=dones_t,
        )


class AugReplayBuffer(ReplayBuffer):

    def __init__(
        self, size: int, aug_n: int, augment_fn: Callable[[Transition], Transition]
    ) -> None:
        super().__init__(size)
        self.aug_n = aug_n
        self.augment_fn = augment_fn

    def add(self, data: Transition, augment: bool = True) -> None:
        super().add(data)
        if augment:
            for _ in range(self.aug_n):
                aug_transition = self.augment_fn(data)
                super().add(aug_transition)
