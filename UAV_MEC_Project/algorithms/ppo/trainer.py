from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .agent import PPOAgent


class PPOTrainer:
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        alg = config["algorithm"]

        self.device = alg.get("device", "cpu")
        self.num_agents = env.num_drones
        self.single_action_dim = env.action_dim
        self.action_dim = self.num_agents * self.single_action_dim
        self.state_dim = env.state_dim
        self.max_action = config["env"]["max_drone_speed_mps"]
        self.greedy_ratio = alg.get("greedy_ratio", 0.7)

        hidden_dim = alg.get("hidden_dim", 64)
        lr = alg.get("lr", 0.0003)

        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            hidden_dim=hidden_dim,
            lr=lr,
            device=self.device,
        )

    def greedy_action(self) -> np.ndarray:
        action = np.zeros(
            (self.num_agents, self.single_action_dim),
            dtype=np.float32,
        )

        pending_tasks = [
            task for task in self.env.pending_tasks
            if not task.completed and not task.failed
        ]

        if not pending_tasks:
            return action

        for i, drone in enumerate(self.env.drones):
            nearest_task = min(
                pending_tasks,
                key=lambda task: np.linalg.norm(
                    task.source_position - drone.position[:2]
                ),
            )

            direction = nearest_task.source_position - drone.position[:2]
            norm = np.linalg.norm(direction)

            if norm > 1e-6:
                direction = direction / norm
                action[i] = direction * self.max_action

        return action

    def train(self) -> List[Dict]:
        alg = self.config["algorithm"]
        episodes = alg["episodes"]
        max_steps = alg["max_steps_per_episode"]

        logs = []

        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.config.get("seed", 1) + ep)

            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []

            ep_reward = 0.0
            info = None

            for _ in range(max_steps):
                state = obs["global_state"]

                ppo_action_flat, log_prob = self.agent.select_action(state)
                ppo_action = ppo_action_flat.reshape(
                    self.num_agents,
                    self.single_action_dim,
                )

                greedy_action = self.greedy_action()

                final_action = (
                    self.greedy_ratio * greedy_action
                    + (1.0 - self.greedy_ratio) * ppo_action
                )

                final_action = np.clip(
                    final_action,
                    -self.max_action,
                    self.max_action,
                )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    final_action
                )

                states.append(state)
                actions.append(final_action.reshape(-1))
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(terminated or truncated))

                obs = next_obs
                ep_reward += reward

                if terminated or truncated:
                    break

            self.update(states, actions, log_probs, rewards, dones)

            row = {
                "episode": ep,
                "reward": ep_reward,
                **info["metrics"],
            }

            logs.append(row)

            print(
                f"Greedy-PPO episode {ep + 1}/{episodes} "
                f"reward={ep_reward:.2f} completed={row['completed_tasks']}"
            )

        return logs

    def compute_returns(self, rewards, dones, gamma):
        returns = []
        discounted = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted = 0.0

            discounted = reward + gamma * discounted
            returns.insert(0, discounted)

        return returns

    def update(self, states, actions, old_log_probs, rewards, dones):
        alg = self.config["algorithm"]

        gamma = alg.get("gamma", 0.99)
        clip_eps = alg.get("clip_eps", 0.2)
        update_epochs = alg.get("update_epochs", 5)
        entropy_coef = alg.get("entropy_coef", 0.01)
        value_coef = alg.get("value_coef", 0.5)

        returns = self.compute_returns(rewards, dones, gamma)

        states_t = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.agent.device,
        )

        actions_t = torch.tensor(
            np.array(actions),
            dtype=torch.float32,
            device=self.agent.device,
        )

        old_log_probs_t = torch.tensor(
            old_log_probs,
            dtype=torch.float32,
            device=self.agent.device,
        ).view(-1, 1)

        returns_t = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.agent.device,
        ).view(-1, 1)

        with torch.no_grad():
            values = self.agent.critic(states_t)
            advantages = returns_t - values
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        for _ in range(update_epochs):
            new_log_probs, entropy, values = self.agent.evaluate(
                states_t,
                actions_t,
            )

            ratio = torch.exp(new_log_probs - old_log_probs_t)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns_t)
            entropy_loss = entropy.mean()

            loss = (
                actor_loss
                + value_coef * critic_loss
                - entropy_coef * entropy_loss
            )

            self.agent.actor_optimizer.zero_grad()
            self.agent.critic_optimizer.zero_grad()

            loss.backward()

            self.agent.actor_optimizer.step()
            self.agent.critic_optimizer.step()