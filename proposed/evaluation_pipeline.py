"""
Post-training and standalone evaluation: dataset metrics + existing graph_outputs.

Generates per-scenario graphs under results/ without removing graph_outputs/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import GRAPH_OUTPUTS_DIR, iter_scenario_frames, load_dataset
from episode_horizon_config import EPISODE_HORIZON_STEPS
from plot_smoothing import smooth_curve
from results_paths import (
    DELAY_GRAPHS_DIR,
    ENERGY_GRAPHS_DIR,
    LIFETIME_GRAPHS_DIR,
    PDR_GRAPHS_DIR,
    THROUGHPUT_GRAPHS_DIR,
    UTILIZATION_GRAPHS_DIR,
    ensure_results_dirs,
    scenario_alias,
)
from scenarios import list_scenario_ids

PROJECT_DIR = Path(__file__).resolve().parent
PUBLICATION_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _save_line_plot(
    x,
    y,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    x_max: int | None = None,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    y_plot = smooth_curve(
        y,
        window_fraction=0.10,
        min_window=21,
        max_window=101,
        passes=2,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, y_plot, color="#1B4965", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xmax = x_max if x_max is not None else EPISODE_HORIZON_STEPS
    ax.set_xlim(0, xmax)
    if xmax >= 1000:
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_dataset_metric_graphs(
    scenario_ids: list[str] | None = None,
) -> None:
    """Per-scenario delay, throughput, PDR, energy, utilization, lifetime graphs."""
    ensure_results_dirs()
    df = load_dataset()
    ids = scenario_ids or list_scenario_ids()

    for sid, sname, sdf in iter_scenario_frames(df):
        if sid not in ids:
            continue
        alias = scenario_alias(sid)
        by_t = sdf.groupby("Timestamp").agg({
            "Delay": "mean",
            "PDR": "mean",
            "Energy": "mean",
            "Signal_Strength": "mean",
        }).reset_index().sort_values("Timestamp")

        timestamps = by_t["Timestamp"].values

        _save_line_plot(
            timestamps,
            by_t["Delay"],
            f"{sname} — Communication Delay",
            f"Step within episode horizon (0–{EPISODE_HORIZON_STEPS})",
            "Delay (ms)",
            DELAY_GRAPHS_DIR / f"{alias}_delay.png",
        )

        throughput = by_t["PDR"] * by_t["Signal_Strength"] * 100.0
        _save_line_plot(
            timestamps,
            throughput,
            f"{sname} — Throughput Proxy",
            "Timestamp",
            "Throughput (arb. units)",
            THROUGHPUT_GRAPHS_DIR / f"{alias}_throughput.png",
        )

        _save_line_plot(
            timestamps,
            by_t["PDR"],
            f"{sname} — Packet Delivery Ratio",
            "Timestamp",
            "PDR (%)",
            PDR_GRAPHS_DIR / f"{alias}_pdr.png",
        )

        _save_line_plot(
            timestamps,
            by_t["Energy"],
            f"{sname} — UAV Energy",
            "Timestamp",
            "Energy",
            ENERGY_GRAPHS_DIR / f"{alias}_energy.png",
        )

        util = sdf.groupby("UAV_ID")["Energy"].mean().sort_index()
        plt.rcParams.update(PUBLICATION_RC)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(range(len(util)), util.values, color="#2E86AB")
        ax.set_xticks(range(len(util)))
        ax.set_xticklabels(util.index, rotation=45)
        ax.set_title(f"{sname} — UAV Utilization (mean energy)")
        ax.set_ylabel("Mean energy level")
        out = UTILIZATION_GRAPHS_DIR / f"{alias}_utilization.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

        lifetime = sdf.groupby("Timestamp")["Energy"].min().sort_index()
        _save_line_plot(
            lifetime.index.values,
            lifetime.values,
            f"{sname} — Network Lifetime Proxy (min fleet energy)",
            "Timestamp",
            "Min UAV energy",
            LIFETIME_GRAPHS_DIR / f"{alias}_lifetime.png",
        )


def _run_legacy_analysis_scripts() -> None:
    """Invoke existing evaluation scripts (graph_outputs preserved)."""
    scripts = [
        "pdr_analysis.py",
        "delay_analysis.py",
        "reliability_score.py",
        "final_performance_comparison.py",
        "scenario_analysis.py",
        "energy_optimization.py",
    ]
    for script in scripts:
        path = PROJECT_DIR / script
        if not path.is_file():
            continue
        print(f"\nRunning legacy analysis: {script}")
        subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT_DIR),
            check=False,
        )


def _mirror_graph_outputs_to_results() -> None:
    """Copy per-scenario graph_outputs into results folders for unified layout."""
    src_root = GRAPH_OUTPUTS_DIR / "scenarios"
    if not src_root.is_dir():
        return

    mapping = {
        "average_delay_per_vehicle.png": DELAY_GRAPHS_DIR,
        "average_pdr_per_vehicle.png": PDR_GRAPHS_DIR,
        "average_reliability_score.png": PDR_GRAPHS_DIR,
        "traditional_vs_proposed_comparison.png": PDR_GRAPHS_DIR,
    }

    for sid_dir in src_root.iterdir():
        if not sid_dir.is_dir():
            continue
        alias = scenario_alias(sid_dir.name)
        for fname, dest_dir in mapping.items():
            src = sid_dir / fname
            if src.is_file():
                dest = dest_dir / f"{alias}_{fname.replace('.png', '')}_legacy.png"
                shutil.copy2(src, dest)
                print(f"Mirrored: {dest}")


def run_post_training_evaluation(
    scenario_ids: list[str] | None = None,
) -> None:
    """Full evaluation after RL training (retains all existing outputs)."""
    print("\n=== Post-training evaluation pipeline ===\n")
    from episode_horizon_graphs import generate_episode_horizon_graphs

    generate_episode_horizon_graphs(scenario_ids=scenario_ids)
    generate_dataset_metric_graphs(scenario_ids=scenario_ids)
    _run_legacy_analysis_scripts()
    _mirror_graph_outputs_to_results()
    print("\n=== Evaluation pipeline complete ===\n")


def main() -> None:
    run_post_training_evaluation()


if __name__ == "__main__":
    main()
