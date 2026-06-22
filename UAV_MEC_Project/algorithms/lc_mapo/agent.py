from __future__ import annotations

import copy
from typing import List

import torch

from .networks import Actor


class LCMAPOAgentSet:
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, max_action: float, hidden_dims: list[int], device: str):
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.actors: List[Actor] = [Actor(obs_dim, action_dim, max_action, hidden_dims).to(self.device) for _ in range(num_agents)]
        self.target_actors: List[Actor] = [copy.deepcopy(actor).to(self.device) for actor in self.actors]
        self.optimizers = [torch.optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]

    def set_lr(self, lr: float) -> None:
        self.optimizers = [torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]

    def act(self, obs, noise=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                actions.append(actor(obs_t[i]).cpu())
        action = torch.stack(actions).numpy()
        if noise is not None:
            action = action + noise.sample(action.shape)
        return action

    def joint_action_tensor(self, obs_batch: torch.Tensor, target: bool = False) -> torch.Tensor:
        actors = self.target_actors if target else self.actors
        actions = [actors[i](obs_batch[:, i, :]) for i in range(self.num_agents)]
        return torch.stack(actions, dim=1)
