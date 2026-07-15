"""
Train PPO with episode-based learning (delegates to train_scenario_rl.py).

Default: 100 episodes per scenario, Episode vs Reward graphs, Excel training tables.

Usage:
    python train_ppo.py
    python train_ppo.py --episodes 100
    python train_ppo.py --episodes 100 --scenarios urban_canyon suburban_crossroads
"""

from __future__ import annotations

import argparse

from train_scenario_rl import train_all_scenarios
from training_config import DEFAULT_TRAINING_EPISODES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Episode-based PPO training (per scenario)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
        help=f"Episodes per scenario (default: {DEFAULT_TRAINING_EPISODES})",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Deprecated: use --episodes (steps = episodes * steps_per_episode)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario IDs (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.timesteps is not None:
        print(
            "Note: --timesteps is deprecated. "
            "Training is episode-based; use --episodes instead."
        )

    train_all_scenarios(
        n_episodes=args.episodes,
        scenario_ids=args.scenarios,
        seed=args.seed,
        run_evaluation=not args.no_eval,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
