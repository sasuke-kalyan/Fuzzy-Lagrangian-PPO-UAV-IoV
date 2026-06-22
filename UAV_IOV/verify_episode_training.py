"""
Verify episode-based training configuration and outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from results_paths import (
    REWARD_GRAPHS_DIR,
    TRAINING_TABLES_DIR,
    all_scenario_ids,
    reward_graph_path,
    training_xlsx_path,
)
from scenarios import NUM_SAMPLES
from training_config import DEFAULT_TRAINING_EPISODES, STEPS_PER_EPISODE


def verify() -> int:
    errors: list[str] = []
    ids = all_scenario_ids()

    if STEPS_PER_EPISODE != NUM_SAMPLES:
        errors.append(f"STEPS_PER_EPISODE ({STEPS_PER_EPISODE}) != NUM_SAMPLES ({NUM_SAMPLES})")

    if DEFAULT_TRAINING_EPISODES < 1000:
        errors.append("DEFAULT_TRAINING_EPISODES should be >= 1000")

    for sid in ids:
        rg = reward_graph_path(sid)
        if not rg.is_file():
            errors.append(f"Missing reward graph: {rg}")

        xlsx = training_xlsx_path(sid)
        if not xlsx.is_file():
            errors.append(f"Missing training xlsx: {xlsx}")

        log = Path(__file__).parent / "results" / "training_logs" / f"{sid}_episode_rewards.json"
        if log.is_file():
            data = json.loads(log.read_text())
            n = data.get("n_episodes", 0)
            if n < 1:
                errors.append(f"No episodes in log for {sid}")
        else:
            errors.append(f"Missing training log: {log}")

    print("Scenarios:", ids)
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    print(f"Default training episodes: {DEFAULT_TRAINING_EPISODES}")
    print(f"Reward graphs dir: {REWARD_GRAPHS_DIR}")
    print(f"Training tables dir: {TRAINING_TABLES_DIR}")

    if errors:
        print("\nVerification FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print("\nVerification PASSED (outputs present for all scenarios).")
    return 0


if __name__ == "__main__":
    sys.exit(verify())
