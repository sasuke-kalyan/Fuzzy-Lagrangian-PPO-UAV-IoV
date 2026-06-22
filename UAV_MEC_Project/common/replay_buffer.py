from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: torch.Tensor
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_states: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        obs_shape: tuple[int, int],
        action_shape: tuple[int, int],
        cost_dim: int,
        device: str,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, cost_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, obs, action, reward, costs, next_state, next_obs, done) -> None:
        i = self.ptr
        self.states[i] = state
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.costs[i] = costs
        self.next_states[i] = next_state
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            states=self._t(self.states[idx]),
            obs=self._t(self.obs[idx]),
            actions=self._t(self.actions[idx]),
            rewards=self._t(self.rewards[idx]),
            costs=self._t(self.costs[idx]),
            next_states=self._t(self.next_states[idx]),
            next_obs=self._t(self.next_obs[idx]),
            dones=self._t(self.dones[idx]),
        )

    def _t(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)
