from __future__ import annotations

from .policy import GreedyPolicy


def run_greedy(env, episodes: int = 3):
    policy = GreedyPolicy(env.config["env"]["max_drone_speed_mps"])
    rows = []
    for ep in range(episodes):
        _, _ = env.reset(seed=env.config.get("seed", 1) + ep)
        total = 0.0
        while True:
            action = policy.act(env)
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rows.append({"episode": ep, "reward": total, **info["metrics"]})
    return rows
