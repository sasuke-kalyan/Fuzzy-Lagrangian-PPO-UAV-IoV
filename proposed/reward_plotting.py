"""
Paper-style Episode vs Episode Reward — single smooth line (proposed method only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from episode_training_callback import EpisodeRecord, EpisodeTrainingLog
from episode_horizon_config import EPISODE_HORIZON_STEPS, TRAINING_EPISODES
from plot_smoothing import smooth_curve
from results_paths import REWARD_GRAPHS_DIR, reward_graph_path, training_log_path
from scenarios import SCENARIOS, get_scenario, list_scenario_ids

# IEEE-style figure (matches reference paper subplot look)
PUBLICATION_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
}

PROPOSED_LABEL = "Proposed UAV-IoV"
PROPOSED_COLOR = "#C44E52"  # red line like LC-MAPO in reference figure

SCENARIO_PANEL_TITLE: dict[str, str] = {
    "urban_canyon": "(a) Urban canyon scenario",
    "suburban_crossroads": "(b) Suburban crossroads scenario",
    "emergency_response": "(c) Emergency response scenario",
}


def plot_episode_reward(
    training_log: EpisodeTrainingLog,
) -> str:
    """
    One line only: smoothed episode reward for the proposed PPO method.
    """
    plt.rcParams.update(PUBLICATION_RC)
    REWARD_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    returns = np.array(
        [r.total_reward for r in training_log.records],
        dtype=np.float64,
    )
    n = len(returns)
    if n == 0:
        raise ValueError(f"No episode records for {training_log.scenario_id}")

    episodes = np.arange(0, n)
    smoothed = smooth_curve(
        returns,
        window_fraction=0.10,
        min_window=51,
        max_window=101,
        passes=2,
    )

    panel = SCENARIO_PANEL_TITLE.get(
        training_log.scenario_id,
        get_scenario(training_log.scenario_id).name,
    )

    # X-axis fixed to full training run (1000 episodes) like reference paper
    x_max = max(TRAINING_EPISODES, n - 1, 1)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        episodes,
        smoothed,
        color=PROPOSED_COLOR,
        linewidth=2.0,
        label=PROPOSED_LABEL,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_xlim(0, x_max)
    if x_max >= 1000:
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ylo = float(smoothed.min()) * 0.96
    yhi = float(smoothed.max()) * 1.04
    ax.set_ylim(ylo, yhi)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False)

    # Scenario caption under plot (paper style)
    fig.text(0.5, 0.02, panel, ha="center", fontsize=12, fontweight="bold")

    out = reward_graph_path(training_log.scenario_id)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def load_log_from_json(scenario_id: str) -> EpisodeTrainingLog | None:
    path = training_log_path(scenario_id)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    log = EpisodeTrainingLog(
        scenario_id=data["scenario_id"],
        scenario_name=data["scenario_name"],
    )
    for row in data["records"]:
        log.records.append(
            EpisodeRecord(
                episode=int(row["Episode"]),
                total_reward=float(row["Total Reward"]),
                average_reward=float(row["Average Reward"]),
                scenario_id=data["scenario_id"],
                scenario_name=data["scenario_name"],
                training_time_sec=float(row["Training Time"]),
                n_steps=int(data.get("steps_per_episode", EPISODE_HORIZON_STEPS)),
            )
        )
    return log


def plot_all_scenario_rewards(logs: dict[str, EpisodeTrainingLog]) -> list[str]:
    paths = []
    for sid, log in logs.items():
        if not log.records:
            print(f"Warning: no episodes for {sid}; skip reward graph.")
            continue
        path = plot_episode_reward(log)
        print(f"Saved reward graph: {path}")
        paths.append(path)
    return paths


def replot_from_saved_logs() -> list[str]:
    """Regenerate paper-style graphs from results/training_logs/*.json."""
    logs: dict[str, EpisodeTrainingLog] = {}
    for sid in list_scenario_ids():
        log = load_log_from_json(sid)
        if log and log.records:
            logs[sid] = log
    if not logs:
        raise FileNotFoundError(
            "No training logs found. Run: python train_ppo.py --episodes 1000"
        )
    return plot_all_scenario_rewards(logs)


if __name__ == "__main__":
    replot_from_saved_logs()
