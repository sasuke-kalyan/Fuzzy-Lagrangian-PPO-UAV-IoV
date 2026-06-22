from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms.greedy import GreedyPolicy
from common.config import load_config
from envs import UAVIoVEnv
from evaluation.evaluator import Evaluator
from evaluation.kpi import aggregate_kpis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_lc_mapo.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--algorithm", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    env = UAVIoVEnv(config)
    evaluator = Evaluator(env)
    if args.algorithm == "greedy":
        policy = GreedyPolicy(config["env"]["max_drone_speed_mps"])
        policy_fn = lambda obs, e: policy.act(e)
    else:
        policy_fn = lambda obs, e: np.zeros((e.num_drones, e.action_dim), dtype=np.float32)
    rows = evaluator.evaluate_policy(policy_fn, episodes=args.episodes)
    kpis = aggregate_kpis(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.algorithm}_rows.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(out / f"{args.algorithm}_kpis.json", "w", encoding="utf-8") as handle:
        json.dump(kpis, handle, indent=2)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
