"""
Episode-based PPO training per simulation scenario (default: 100 episodes).

Each episode runs STEPS_PER_EPISODE environment steps (= dataset timestamps).
Rewards accumulate per step; episode total and average are logged.

Usage:
    python train_scenario_rl.py
    python train_scenario_rl.py --episodes 100
    python train_scenario_rl.py --episodes 100 --scenarios urban_canyon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from episode_training_callback import EpisodeRewardCallback, EpisodeTrainingLog
from evaluation_pipeline import run_post_training_evaluation
from results_paths import ensure_results_dirs, model_path_for_scenario, training_log_path
from reward_plotting import plot_all_scenario_rewards
from scenarios import SCENARIOS, list_scenario_ids
from training_config import (
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_N_EPOCHS,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_TRAINING_EPISODES,
    STEPS_PER_EPISODE,
)
from training_tables import export_all_training_tables
from uav_iov_env import UAVIoVEnv


def warm_start_policy(
    model: PPO,
    scenario_id: str,
    seed: int,
    samples: int = 800,
    batch_size: int = 64,
    epochs: int = 6,
) -> None:
    """Behavior-clone the fuzzy-ranked top action before PPO learning."""
    env = UAVIoVEnv(scenario_id=scenario_id)
    observations: list[np.ndarray] = []
    actions: list[int] = []

    obs, _ = env.reset(seed=seed)
    for _ in range(samples):
        observations.append(obs.copy())
        actions.append(0)
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            obs, _ = env.reset(seed=int(env._rng.integers(0, 1_000_000)))
    env.close()

    obs_tensor = torch.as_tensor(
        np.asarray(observations, dtype=np.float32),
        device=model.device,
    )
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=model.device)

    model.policy.set_training_mode(True)
    for _ in range(epochs):
        indices = torch.randperm(len(actions), device=model.device)
        for start in range(0, len(actions), batch_size):
            batch_idx = indices[start : start + batch_size]
            distribution = model.policy.get_distribution(obs_tensor[batch_idx])
            log_prob = distribution.distribution.log_prob(action_tensor[batch_idx])
            loss = -log_prob.mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()


def train_one_scenario(
    scenario_id: str,
    n_episodes: int,
    seed: int = 42,
    verbose: int = 1,
) -> EpisodeTrainingLog:
    cfg = SCENARIOS[scenario_id]
    env = Monitor(
        UAVIoVEnv(scenario_id=scenario_id),
        filename=None,
    )

    callback = EpisodeRewardCallback(
        scenario_id=scenario_id,
        scenario_name=cfg.name,
        target_episodes=n_episodes,
        steps_per_episode=STEPS_PER_EPISODE,
        verbose=verbose,
    )

    total_timesteps = n_episodes * STEPS_PER_EPISODE

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=DEFAULT_PPO_LEARNING_RATE,
        n_steps=min(DEFAULT_PPO_N_STEPS, STEPS_PER_EPISODE * 4),
        batch_size=64,
        n_epochs=DEFAULT_PPO_N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1 if verbose else 0,
        seed=seed,
    )

    print(
        f"\n--- Training PPO | {cfg.name} ({scenario_id}) ---\n"
        f"Episodes: {n_episodes} | Steps/episode: {STEPS_PER_EPISODE} | "
        f"Total steps: {total_timesteps:,}\n"
    )

    print("Warm-starting PPO with fuzzy-priority top-candidate selections...")
    warm_start_policy(model, scenario_id=scenario_id, seed=seed)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=bool(verbose),
    )

    save_path = model_path_for_scenario(scenario_id)
    model.save(str(save_path))
    env.close()

    log_path = training_log_path(scenario_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario_id,
                "scenario_name": cfg.name,
                "n_episodes": len(callback.log.records),
                "steps_per_episode": STEPS_PER_EPISODE,
                "records": callback.log.to_dataframe_rows(),
            },
            f,
            indent=2,
        )

    print(f"Model saved: {save_path}.zip")
    print(f"Episode log saved: {log_path}")
    return callback.log


def train_all_scenarios(
    n_episodes: int,
    scenario_ids: list[str] | None = None,
    seed: int = 42,
    run_evaluation: bool = True,
    verbose: int = 1,
) -> dict[str, EpisodeTrainingLog]:
    ensure_results_dirs()
    ids = scenario_ids or list_scenario_ids()
    logs: dict[str, EpisodeTrainingLog] = {}

    for i, sid in enumerate(ids):
        logs[sid] = train_one_scenario(
            sid,
            n_episodes=n_episodes,
            seed=seed + i * 1000,
            verbose=verbose,
        )

    plot_all_scenario_rewards(logs)
    export_all_training_tables(logs)

    if run_evaluation:
        run_post_training_evaluation(scenario_ids=ids)

    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Episode-based PPO training per UAV-IoV scenario"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
        help=f"Training episodes per scenario (default: {DEFAULT_TRAINING_EPISODES})",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario IDs (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-training evaluation graphs",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    train_all_scenarios(
        n_episodes=args.episodes,
        scenario_ids=args.scenarios,
        seed=args.seed,
        run_evaluation=not args.no_eval,
        verbose=0 if args.quiet else 1,
    )
    print("\nEpisode-based training pipeline finished.\n")


if __name__ == "__main__":
    main()
