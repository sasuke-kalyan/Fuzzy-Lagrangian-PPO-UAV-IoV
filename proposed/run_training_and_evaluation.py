#!/usr/bin/env python3
"""
Full end-to-end workflow: datasets → RL training → all evaluation outputs.

    python run_training_and_evaluation.py
    python run_training_and_evaluation.py --episodes 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_scenario_rl import train_all_scenarios
from training_config import DEFAULT_TRAINING_EPISODES

PROJECT_DIR = Path(__file__).resolve().parent


def run_dataset_generation() -> None:
    print("\n=== Step 1/3: Dataset generation ===\n")
    subprocess.run(
        [sys.executable, str(PROJECT_DIR / "dataset_generation.py")],
        cwd=str(PROJECT_DIR),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: datasets + PPO training + evaluation"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
    )
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset_generation.py (use existing CSVs)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only run evaluation_pipeline",
    )
    args = parser.parse_args()

    if not args.skip_dataset and not args.eval_only:
        run_dataset_generation()

    if args.eval_only:
        from evaluation_pipeline import run_post_training_evaluation

        print("\n=== Evaluation only ===\n")
        run_post_training_evaluation(scenario_ids=args.scenarios)
    else:
        print("\n=== Step 2/3: Episode-based RL training ===\n")
        train_all_scenarios(
            n_episodes=args.episodes,
            scenario_ids=args.scenarios,
            seed=args.seed,
            run_evaluation=True,
        )
    print("\n=== Full pipeline finished ===\n")


if __name__ == "__main__":
    main()
