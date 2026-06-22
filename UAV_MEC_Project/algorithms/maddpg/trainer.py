from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn.functional as F

from common.noise import GaussianNoise
from common.replay_buffer import ReplayBuffer
from algorithms.lc_mapo.networks import Actor, Critic


class MADDPGTrainer:
    """Unconstrained CTDE baseline. Costs are logged but not optimized."""

    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        alg = config["algorithm"]
        self.device = torch.device(alg.get("device", "cpu"))
        self.num_agents = env.num_drones
        self.max_action = config["env"]["max_drone_speed_mps"]
        hidden = alg["hidden_dims"]
        self.actors = [Actor(env.obs_dim, env.action_dim, self.max_action, hidden).to(self.device) for _ in range(env.num_drones)]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=alg["actor_lr"]) for a in self.actors]
        joint_action_dim = env.num_drones * env.action_dim
        self.critics = [Critic(env.state_dim, joint_action_dim, hidden).to(self.device) for _ in range(env.num_drones)]
        self.target_critics = [copy.deepcopy(c).to(self.device) for c in self.critics]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=alg["critic_lr"]) for c in self.critics]
        self.replay = ReplayBuffer(
            alg["buffer_size"], env.state_dim, (env.num_drones, env.obs_dim), (env.num_drones, env.action_dim), 2, str(self.device)
        )
        self.noise = GaussianNoise(alg["exploration_noise"], self.max_action)

    def act(self, obs, noise=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                acts.append(actor(obs_t[i]).cpu())
        action = torch.stack(acts).numpy()
        if noise:
            action = action + self.noise.sample(action.shape)
        return action

    def _joint_actions(self, obs_batch, target=False):
        actors = self.target_actors if target else self.actors
        return torch.stack([actors[i](obs_batch[:, i, :]) for i in range(self.num_agents)], dim=1)

    def train(self):
        alg = self.config["algorithm"]
        logs = []
        for ep in range(alg["episodes"]):
            obs, _ = self.env.reset(seed=self.config.get("seed", 1) + ep)
            ep_reward = 0.0
            for _ in range(alg["max_steps_per_episode"]):
                action = self.act(obs["local_obs"], noise=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.replay.add(obs["global_state"], obs["local_obs"], action, reward, info["costs"], next_obs["global_state"], next_obs["local_obs"], terminated or truncated)
                obs = next_obs
                ep_reward += reward
                if self.replay.ready(alg["batch_size"]):
                    self.update(alg["batch_size"])
                if terminated or truncated:
                    break
            row = {"episode": ep, "reward": ep_reward, **info["metrics"]}
            logs.append(row)
            print(f"MADDPG episode {ep+1}/{alg['episodes']} reward={ep_reward:.2f} completed={row['completed_tasks']}")
        return logs

    def update(self, batch_size: int):
        alg = self.config["algorithm"]
        batch = self.replay.sample(batch_size)
        with torch.no_grad():
            next_actions = self._joint_actions(batch.next_obs, target=True)
        for i in range(self.num_agents):
            with torch.no_grad():
                y = batch.rewards + alg["gamma"] * (1.0 - batch.dones) * self.target_critics[i](batch.next_states, next_actions)
            q = self.critics[i](batch.states, batch.actions)
            loss_c = F.mse_loss(q, y)
            self.critic_opts[i].zero_grad()
            loss_c.backward()
            self.critic_opts[i].step()
            actions = self._joint_actions(batch.obs, target=False)
            loss_a = -self.critics[i](batch.states, actions).mean()
            self.actor_opts[i].zero_grad()
            loss_a.backward()
            self.actor_opts[i].step()
        self.soft_update(alg["tau"])

    def soft_update(self, tau: float):
        for actor, target in zip(self.actors, self.target_actors):
            for p, tp in zip(actor.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
        for critic, target in zip(self.critics, self.target_critics):
            for p, tp in zip(critic.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
