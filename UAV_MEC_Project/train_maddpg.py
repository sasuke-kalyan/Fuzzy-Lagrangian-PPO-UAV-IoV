from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithms.maddpg import MADDPGTrainer
from common.config import load_config
from common.seeding import set_seed
from envs import UAVIoVEnv
from visualization.plots import plot_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_maddpg.json")
    parser.add_argument("--output-dir", default="results/maddpg")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 1))
    env = UAVIoVEnv(config)
    trainer = MADDPGTrainer(env, config)
    rows = trainer.train()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "training_log.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    plot_learning_curve(rows, out / "learning_curve.png", "MADDPG")
    print(f"Saved MADDPG results to {out}")


if __name__ == "__main__":
    main()
