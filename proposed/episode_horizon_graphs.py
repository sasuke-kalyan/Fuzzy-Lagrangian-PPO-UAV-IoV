"""
Episode horizon graphs: metrics over all 1000 steps within one episode (per scenario).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import iter_scenario_frames, load_dataset
from episode_horizon_config import EPISODE_HORIZON_STEPS
from plot_smoothing import smooth_curve
from results_paths import EPISODE_HORIZON_GRAPHS_DIR, ensure_results_dirs, scenario_alias
from scenarios import list_scenario_ids

PUBLICATION_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
}

HORIZON_XLABEL = f"Step within episode horizon (0–{EPISODE_HORIZON_STEPS})"


def _horizon_ticks(ax, n_steps: int) -> None:
    ax.set_xlim(0, max(EPISODE_HORIZON_STEPS, n_steps - 1))
    if EPISODE_HORIZON_STEPS >= 1000:
        ax.set_xticks([0, 200, 400, 600, 800, 1000])


def _plot_horizon(
    steps,
    values,
    title: str,
    ylabel: str,
    out_path,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    y = smooth_curve(
        values,
        window_fraction=0.10,
        min_window=21,
        max_window=101,
        passes=2,
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(steps, y, color="#C44E52", linewidth=2.0, label="Proposed UAV-IoV")
    ax.set_xlabel(HORIZON_XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _horizon_ticks(ax, len(steps))
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_episode_horizon_graphs(scenario_ids: list[str] | None = None) -> None:
    """One graph per metric × scenario over the full 1000-step episode horizon."""
    ensure_results_dirs()
    EPISODE_HORIZON_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    ids = scenario_ids or list_scenario_ids()

    for sid, sname, sdf in iter_scenario_frames(df):
        if sid not in ids:
            continue
        alias = scenario_alias(sid)
        by_t = (
            sdf.groupby("Timestamp")
            .agg({"Delay": "mean", "PDR": "mean", "Energy": "mean", "Signal_Strength": "mean"})
            .reset_index()
            .sort_values("Timestamp")
        )
        steps = by_t["Timestamp"].values
        n_steps = int(by_t["Timestamp"].nunique())
        print(f"\n{sname}: {n_steps} horizon steps (target {EPISODE_HORIZON_STEPS})")

        _plot_horizon(
            steps,
            by_t["Delay"],
            f"{sname} — Delay over Episode Horizon",
            "Delay (ms)",
            EPISODE_HORIZON_GRAPHS_DIR / f"{alias}_delay_horizon.png",
        )
        _plot_horizon(
            steps,
            by_t["PDR"],
            f"{sname} — PDR over Episode Horizon",
            "PDR (%)",
            EPISODE_HORIZON_GRAPHS_DIR / f"{alias}_pdr_horizon.png",
        )
        _plot_horizon(
            steps,
            by_t["PDR"] * by_t["Signal_Strength"] * 100.0,
            f"{sname} — Throughput over Episode Horizon",
            "Throughput (arb. units)",
            EPISODE_HORIZON_GRAPHS_DIR / f"{alias}_throughput_horizon.png",
        )
        _plot_horizon(
            steps,
            by_t["Energy"],
            f"{sname} — Energy over Episode Horizon",
            "UAV Energy",
            EPISODE_HORIZON_GRAPHS_DIR / f"{alias}_energy_horizon.png",
        )


if __name__ == "__main__":
    generate_episode_horizon_graphs()
