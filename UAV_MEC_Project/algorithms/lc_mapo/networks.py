from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float, hidden_dims: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dims, action_dim)
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q = mlp(state_dim + joint_action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        if joint_action.ndim > 2:
            joint_action = joint_action.flatten(start_dim=1)
        return self.q(torch.cat([state, joint_action], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q1 = Critic(state_dim, joint_action_dim, hidden_dims)
        self.q2 = Critic(state_dim, joint_action_dim, hidden_dims)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, joint_action), self.q2(state, joint_action)

    def q1_value(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, joint_action)
