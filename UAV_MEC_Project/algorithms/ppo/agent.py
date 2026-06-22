from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from .networks import Actor, Critic


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim,
        lr,
        device,
    ):
        self.device = torch.device(device)

        self.max_action = max_action

        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim,
            hidden_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

        self.log_std = torch.zeros(
            action_dim,
            device=self.device,
        )

    def select_action(self, state):
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mean = self.actor(state) * self.max_action

        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.detach().cpu().numpy()[0]

        return (
            np.clip(
                action_np,
                -self.max_action,
                self.max_action,
            ),
            log_prob.item(),
        )

    def evaluate(self, states, actions):
        mean = self.actor(states) * self.max_action

        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(
            dim=-1,
            keepdim=True,
        )

        entropy = dist.entropy().sum(
            dim=-1,
            keepdim=True,
        )

        values = self.critic(states)

        return log_probs, entropy, values