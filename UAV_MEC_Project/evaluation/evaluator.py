from __future__ import annotations

from typing import Callable, Dict, List


class Evaluator:
    def __init__(self, env):
        self.env = env

    def evaluate_policy(self, policy_fn: Callable, episodes: int = 5) -> List[Dict]:
        rows = []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.env.config.get("seed", 1) + 1000 + ep)
            total_reward = 0.0
            while True:
                action = policy_fn(obs, self.env)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rows.append({"episode": ep, "reward": total_reward, **info["metrics"]})
        return rows
