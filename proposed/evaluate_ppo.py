"""
Evaluate trained PPO vs random and greedy (best signal) policies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from communication_model import compute_link_metrics, shaped_reward
from uav_iov_env import UAVIoVEnv

MODEL_PATH = Path("models/ppo_uav_iov")


def _greedy_action(env: UAVIoVEnv) -> int:
    """Pick UAV with highest instantaneous signal (oracle-style baseline)."""
    best_a = 0
    best_signal = -1.0
    vx, vy = env._vehicle_xy
    for i in range(env.NUM_UAVS):
        ux, uy, uz = env._uav_positions[i]
        m = compute_link_metrics(vx, vy, ux, uy, uz, rng=env._rng)
        if m["Signal_Strength"] > best_signal:
            best_signal = m["Signal_Strength"]
            best_a = i
    return best_a


def run_episodes(policy_fn, n_episodes: int = 20, seed: int = 42) -> dict[str, float]:
    env = UAVIoVEnv()
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy_fn(env, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            steps += 1

        returns.append(ep_return)
        lengths.append(steps)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def _random_policy(env: UAVIoVEnv, obs) -> int:
    return int(env._rng.integers(0, env.NUM_UAVS))


def _greedy_policy(env: UAVIoVEnv, obs) -> int:
    return _greedy_action(env)


def _ppo_policy(model, env: UAVIoVEnv, obs) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def main():
    print("\n======================================")
    print(" PPO Policy Evaluation ")
    print("======================================\n")

    n_episodes = 30

    random_stats = run_episodes(
        lambda env, obs: _random_policy(env, obs),
        n_episodes=n_episodes,
    )
    greedy_stats = run_episodes(
        lambda env, obs: _greedy_policy(env, obs),
        n_episodes=n_episodes,
    )

    print(f"Random policy  — mean return: {random_stats['mean_return']:.2f} "
          f"(±{random_stats['std_return']:.2f})")
    print(f"Greedy (signal)— mean return: {greedy_stats['mean_return']:.2f} "
          f"(±{greedy_stats['std_return']:.2f})")

    zip_path = MODEL_PATH.with_suffix(".zip")
    if not zip_path.exists() and not MODEL_PATH.exists():
        print(f"\nNo trained model at {MODEL_PATH}.zip")
        print("Run: python train_ppo.py")
        return

    from stable_baselines3 import PPO

    env_check = UAVIoVEnv()
    model = PPO.load(str(MODEL_PATH))
    if model.observation_space.shape != env_check.observation_space.shape:
        print(
            f"\nStale PPO model: expected obs shape {env_check.observation_space.shape}, "
            f"got {model.observation_space.shape}."
        )
        print("Re-train after fleet size change: python train_ppo.py")
        env_check.close()
        return
    env_check.close()

    ppo_stats = run_episodes(
        lambda env, obs: _ppo_policy(model, env, obs),
        n_episodes=n_episodes,
    )
    print(f"PPO (neural)   — mean return: {ppo_stats['mean_return']:.2f} "
          f"(±{ppo_stats['std_return']:.2f})")

    improvement = ppo_stats["mean_return"] - random_stats["mean_return"]
    print(f"\nPPO vs random: {improvement:+.2f} mean return")
    print("\nEvaluation completed.\n")


if __name__ == "__main__":
    main()
