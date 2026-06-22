#!/usr/bin/env python3
"""Compare training results from multiple method folders and generate comparison graphs.

Usage examples:
  python scripts/compare_results.py MADDPG=/path/to/results/maddpg LC-MAPO=/path/to/results/lc_mapo -o results/comparison

The script looks for a `training_log.json` file inside each provided path (or accepts a direct path
to a JSON/CSV file). It plots reward, completed tasks, and total energy for each method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def moving_average(data: List[float], window: int) -> List[float]:
    if window <= 1:
        return data
    arr = np.array(data, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad the beginning so result length matches
    pad = [np.mean(arr[:i + 1]) for i in range(window - 1)]
    return pad + ma.tolist()


def load_rows(path: Path) -> List[Dict]:
    path = Path(path)
    if path.is_dir():
        candidate = path / "training_log.json"
        if candidate.exists():
            path = candidate
        else:
            # try to find any json or csv with episode-like structure
            for ext in ("training_log.json", "log.json", "results.json"):
                p = path / ext
                if p.exists():
                    path = p
                    break

    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    if path.suffix.lower() in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect a list of dicts with keys 'episode', 'reward', 'completed_tasks', 'total_energy_j'
        return data

    if path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    raise RuntimeError(f"Unsupported file type: {path}")


def plot_metric(all_rows: Dict[str, List[Dict]], metric: str, ylabel: str, outpath: Path, window: int):
    plt.figure(figsize=(10, 6))

    for label, rows in all_rows.items():
        xs = [r.get("episode", i) for i, r in enumerate(rows)]
        ys = [r.get(metric, np.nan) for r in rows]
        ys = [float(x) if x is not None else np.nan for x in ys]
        ys_smooth = moving_average(ys, window)
        plt.plot(xs, ys_smooth, label=label)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="List of NAME=PATH pairs pointing to result folders or log files")
    parser.add_argument("-o", "--output", default="results/comparison", help="Output directory for comparison graphs")
    parser.add_argument("--window", type=int, default=100, help="Smoothing window for moving average")
    args = parser.parse_args()

    pairs = {}
    for item in args.inputs:
        if "=" not in item:
            parser.error("Each input must be NAME=PATH")
        name, path = item.split("=", 1)
        pairs[name] = Path(path)

    all_rows: Dict[str, List[Dict]] = {}
    for name, p in pairs.items():
        try:
            rows = load_rows(p)
            all_rows[name] = rows
            print(f"Loaded {len(rows)} rows for {name} from {p}")
        except Exception as e:
            print(f"Failed to load {p} for {name}: {e}")

    if not all_rows:
        print("No data loaded; exiting.")
        return

    outdir = Path(args.output)

    # Reward
    plot_metric(all_rows, "reward", "Reward", outdir / "reward_comparison.png", args.window)

    # Completed tasks
    plot_metric(all_rows, "completed_tasks", "Completed Tasks", outdir / "task_comparison.png", args.window)

    # Energy
    # try different keys for energy
    for k in ("total_energy_j", "energy_j", "energy"):
        # check if any row has this key
        if any(k in r for rows in all_rows.values() for r in rows):
            plot_metric(all_rows, k, "Energy (J)", outdir / "energy_comparison.png", args.window)
            break

    print(f"Comparison graphs generated in {outdir}")


if __name__ == "__main__":
    main()
