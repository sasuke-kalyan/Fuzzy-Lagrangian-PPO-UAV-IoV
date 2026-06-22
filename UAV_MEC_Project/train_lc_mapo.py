from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithms.lc_mapo import LCMAPOTrainer
from common.config import load_config
from common.seeding import set_seed
from envs import UAVIoVEnv
from visualization.plots import plot_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_lc_mapo.json")
    parser.add_argument("--output-dir", default="results/lc_mapo")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 1))
    env = UAVIoVEnv(config)
    trainer = LCMAPOTrainer(env, config)
    rows = trainer.train()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "training_log.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    trainer.save("checkpoints/lc_mapo_latest.pt")
    plot_learning_curve(rows, out / "learning_curve.png", "LC-MAPO")
    print(f"Saved LC-MAPO results to {out}")


if __name__ == "__main__":
    main()
