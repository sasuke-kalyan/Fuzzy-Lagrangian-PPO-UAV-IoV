from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from common.checkpointing import save_checkpoint
from common.noise import GaussianNoise
from common.replay_buffer import ReplayBuffer
from .agent import LCMAPOAgentSet
from .lagrangian import LagrangianMultiplierController
from .networks import TwinCritic


class LCMAPOTrainer:
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        alg = config["algorithm"]
        self.device = torch.device(alg.get("device", "cpu"))
        self.num_agents = env.num_drones
        self.action_dim = env.action_dim
        self.max_action = config["env"]["max_drone_speed_mps"]
        hidden = alg["hidden_dims"]
        self.agents = LCMAPOAgentSet(self.num_agents, env.obs_dim, env.action_dim, self.max_action, hidden, str(self.device))
        self.agents.set_lr(alg["actor_lr"])
        joint_action_dim = self.num_agents * self.action_dim
        self.reward_critic = TwinCritic(env.state_dim, joint_action_dim, hidden).to(self.device)
        self.reward_target = copy.deepcopy(self.reward_critic).to(self.device)
        self.cost_critics = [TwinCritic(env.state_dim, joint_action_dim, hidden).to(self.device) for _ in range(2)]
        self.cost_targets = [copy.deepcopy(c).to(self.device) for c in self.cost_critics]
        self.reward_opt = torch.optim.Adam(self.reward_critic.parameters(), lr=alg["critic_lr"])
        self.cost_opts = [torch.optim.Adam(c.parameters(), lr=alg["critic_lr"]) for c in self.cost_critics]
        self.lagrangian = LagrangianMultiplierController(alg["cost_limits"], alg["lambda_lr"], str(self.device))
        self.replay = ReplayBuffer(
            alg["buffer_size"],
            env.state_dim,
            (env.num_drones, env.obs_dim),
            (env.num_drones, env.action_dim),
            2,
            str(self.device),
        )
        self.exploration_noise = GaussianNoise(alg["exploration_noise"], self.max_action)
        self.total_updates = 0

    def train(self) -> list[Dict]:
        cfg = self.config["algorithm"]
        logs = []
        for ep in range(cfg["episodes"]):
            obs, _ = self.env.reset(seed=self.config.get("seed", 1) + ep)
            ep_reward = 0.0
            for _ in range(cfg["max_steps_per_episode"]):
                action = self.agents.act(obs["local_obs"], noise=self.exploration_noise)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.replay.add(
                    obs["global_state"], obs["local_obs"], action, reward, info["costs"],
                    next_obs["global_state"], next_obs["local_obs"], terminated or truncated,
                )
                obs = next_obs
                ep_reward += reward
                if self.replay.ready(cfg["batch_size"]):
                    self.update(cfg["batch_size"])
                if terminated or truncated:
                    break
            row = {"episode": ep, "reward": ep_reward, **info["metrics"]}
            logs.append(row)
            print(f"LC-MAPO episode {ep+1}/{cfg['episodes']} reward={ep_reward:.2f} completed={row['completed_tasks']}")
        return logs

    def update(self, batch_size: int) -> Dict[str, float]:
        alg = self.config["algorithm"]
        batch = self.replay.sample(batch_size)
        with torch.no_grad():
            next_actions = self.agents.joint_action_tensor(batch.next_obs, target=True)
            noise = torch.randn_like(next_actions) * alg["target_noise"]
            noise = noise.clamp(-alg["target_noise_clip"], alg["target_noise_clip"])
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            q1_t, q2_t = self.reward_target(batch.next_states, next_actions)
            y_r = batch.rewards + alg["gamma"] * (1.0 - batch.dones) * torch.minimum(q1_t, q2_t)
        q1, q2 = self.reward_critic(batch.states, batch.actions)
        reward_loss = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)
        self.reward_opt.zero_grad()
        reward_loss.backward()
        self.reward_opt.step()

        cost_losses = []
        for j, critic in enumerate(self.cost_critics):
            with torch.no_grad():
                cq1_t, cq2_t = self.cost_targets[j](batch.next_states, next_actions)
                y_c = batch.costs[:, j : j + 1] + alg["gamma"] * (1.0 - batch.dones) * torch.minimum(cq1_t, cq2_t)
            cq1, cq2 = critic(batch.states, batch.actions)
            loss = F.mse_loss(cq1, y_c) + F.mse_loss(cq2, y_c)
            self.cost_opts[j].zero_grad()
            loss.backward()
            self.cost_opts[j].step()
            cost_losses.append(loss.item())

        actor_loss_value = 0.0
        lambda_loss_value = 0.0
        if self.total_updates % alg["policy_delay"] == 0:
            joint_actions = self.agents.joint_action_tensor(batch.obs, target=False)
            qr = self.reward_critic.q1_value(batch.states, joint_actions)
            cost_values = torch.cat([c.q1_value(batch.states, joint_actions) for c in self.cost_critics], dim=1)
            lambdas = self.lagrangian.lambdas.detach()
            actor_loss = -(qr - (lambdas * cost_values).sum(dim=1, keepdim=True)).mean()
            for opt in self.agents.optimizers:
                opt.zero_grad()
            actor_loss.backward()
            for opt in self.agents.optimizers:
                opt.step()
            lambda_loss_value = self.lagrangian.update(cost_values)
            self.soft_update_all(alg["tau"])
            actor_loss_value = actor_loss.item()
        self.total_updates += 1
        return {"reward_loss": reward_loss.item(), "actor_loss": actor_loss_value, "lambda_loss": lambda_loss_value}

    def soft_update_all(self, tau: float) -> None:
        def soft(src, dst):
            for p, tp in zip(src.parameters(), dst.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        soft(self.reward_critic, self.reward_target)
        for c, tc in zip(self.cost_critics, self.cost_targets):
            soft(c, tc)
        for actor, target in zip(self.agents.actors, self.agents.target_actors):
            soft(actor, target)

    def save(self, path: str | Path) -> None:
        save_checkpoint({"actors": [a.state_dict() for a in self.agents.actors], "config": self.config}, path)
