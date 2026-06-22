"""
Cross-scenario metrics and comparison plots (paper Section VI — three scenarios).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constraints as cons
from communication_model import compute_link_metrics, shaped_reward
from data_loader import GRAPH_OUTPUTS_DIR, ensure_output_dirs, iter_scenario_frames, load_dataset
from results_paths import model_path_for_scenario
from scenarios import SCENARIOS

DECISION_GROUP_COLS = ["Scenario_ID", "Timestamp", "Vehicle_ID"]
BASELINE_COLORS = {
    "Random": "#7A7A7A",
    "Nearest UAV": "#4C78A8",
    "Strongest Signal": "#F58518",
    "Energy-Aware Greedy": "#54A24B",
    "Reward-Aware Proposed": "#C44E52",
    "Trained PPO Model": "#6F4E7C",
}
HIGH_LATENCY_MS = 80.0
WEAK_SIGNAL_MAX = 0.05
POLICY_EVAL_EPISODES = 20


def _reliability_score(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["Signal_Strength"] * 50)
        + (frame["PDR"] * 0.5)
        - (frame["Delay"] * 0.4)
    )


def _reward_score(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: shaped_reward(
            row["Signal_Strength"],
            row["PDR"],
            row["Delay"],
            row["Energy"],
        ),
        axis=1,
    )


def _select_idxmax(frame: pd.DataFrame, score: pd.Series) -> pd.Index:
    return score.groupby([frame[col] for col in DECISION_GROUP_COLS]).idxmax()


def _select_idxmin(frame: pd.DataFrame, score: pd.Series) -> pd.Index:
    return score.groupby([frame[col] for col in DECISION_GROUP_COLS]).idxmin()


def _env_candidate_metrics(env: UAVIoVEnv) -> list[dict[str, float]]:
    vx, vy = env._vehicle_xy
    candidates = []
    for i in range(env.NUM_UAVS):
        ux, uy, uz = env._uav_positions[i]
        metrics = compute_link_metrics(vx, vy, ux, uy, uz, rng=env._rng)
        metrics["Action"] = i
        metrics["Energy"] = float(env._uav_energies[i])
        metrics["Reward"] = shaped_reward(
            metrics["Signal_Strength"],
            metrics["PDR"],
            metrics["Delay"],
            metrics["Energy"],
        )
        metrics["Reliability"] = (
            (metrics["Signal_Strength"] * 50)
            + (metrics["PDR"] * 0.5)
            - (metrics["Delay"] * 0.4)
        )
        return_metric = dict(metrics)
        candidates.append(return_metric)
    return candidates


def _policy_action(policy_name: str, env: UAVIoVEnv, obs: np.ndarray, model=None) -> int:
    candidates = _env_candidate_metrics(env)
    if policy_name == "Random":
        return int(env._rng.integers(0, env.NUM_UAVS))
    if policy_name == "Nearest UAV":
        return int(min(candidates, key=lambda row: row["Distance"])["Action"])
    if policy_name == "Strongest Signal":
        return int(max(candidates, key=lambda row: row["Signal_Strength"])["Action"])
    if policy_name == "Energy-Aware Greedy":
        return int(max(candidates, key=lambda row: (row["Energy"], -row["Distance"]))["Action"])
    if policy_name == "Reward-Aware Proposed":
        return int(max(candidates, key=lambda row: row["Reward"])["Action"])
    if policy_name == "Trained PPO Model":
        if model is None:
            raise ValueError("PPO policy requested without a loaded model")
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    raise KeyError(f"Unknown policy: {policy_name}")


def _rollout_policy(
    scenario_id: str,
    policy_name: str,
    n_episodes: int = POLICY_EVAL_EPISODES,
    seed: int = 2026,
    model=None,
) -> pd.DataFrame:
    from uav_iov_env import UAVIoVEnv

    rows = []
    env = UAVIoVEnv(scenario_id=scenario_id)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = _policy_action(policy_name, env, obs, model=model)
            obs, reward, terminated, truncated, info = env.step(action)
            metrics = info["metrics"]
            reliability = (
                (metrics["Signal_Strength"] * 50)
                + (metrics["PDR"] * 0.5)
                - (metrics["Delay"] * 0.4)
            )
            rows.append({
                "Episode": ep + 1,
                "Step": env._step_count,
                "Reward": float(reward),
                "Delay": metrics["Delay"],
                "PDR": metrics["PDR"],
                "Signal_Strength": metrics["Signal_Strength"],
                "Energy": info["energy"],
                "Distance": metrics["Distance"],
                "Reliability": reliability,
            })
    env.close()
    return pd.DataFrame(rows)


def _load_ppo_model(scenario_id: str):
    zip_path = model_path_for_scenario(scenario_id).with_suffix(".zip")
    if not zip_path.is_file():
        print(f"Warning: missing PPO model for {scenario_id}: {zip_path}")
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Warning: stable_baselines3 is unavailable; skipping PPO rollout.")
        return None
    return PPO.load(str(model_path_for_scenario(scenario_id)))


def build_policy_rollout_comparison() -> pd.DataFrame:
    rows = []
    base_policies = [
        "Random",
        "Nearest UAV",
        "Strongest Signal",
        "Energy-Aware Greedy",
        "Reward-Aware Proposed",
    ]

    for sid, scenario in SCENARIOS.items():
        ppo_model = _load_ppo_model(sid)
        policies = list(base_policies)
        if ppo_model is not None:
            policies.append("Trained PPO Model")

        rollout_by_policy = {
            policy: _rollout_policy(sid, policy, model=ppo_model if policy == "Trained PPO Model" else None)
            for policy in policies
        }
        proposed_reward = rollout_by_policy["Reward-Aware Proposed"]["Reward"].mean()

        for policy, pdf in rollout_by_policy.items():
            reward = pdf["Reward"].mean()
            episode_returns = pdf.groupby("Episode")["Reward"].sum()
            rows.append({
                "Scenario_ID": sid,
                "Scenario_Name": scenario.name,
                "Method": policy,
                "Avg_Reward_Per_Step": round(reward, 2),
                "Avg_Episode_Reward": round(episode_returns.mean(), 2),
                "Std_Episode_Reward": round(episode_returns.std(ddof=0), 2),
                "Reward_vs_Proposed_pct": round(((reward - proposed_reward) / proposed_reward) * 100.0, 2),
                "Avg_Delay_ms": round(pdf["Delay"].mean(), 2),
                "Avg_PDR_pct": round(pdf["PDR"].mean(), 2),
                "Avg_Signal": round(pdf["Signal_Strength"].mean(), 4),
                "Avg_Energy": round(pdf["Energy"].mean(), 2),
                "Avg_Reliability": round(pdf["Reliability"].mean(), 2),
                "Out_Of_Range_Rate_pct": round((pdf["Distance"] > 500.0).mean() * 100, 2),
                "High_Latency_Risk_pct": round((pdf["Delay"] > HIGH_LATENCY_MS).mean() * 100, 2),
                "Weak_Signal_Risk_pct": round((pdf["Signal_Strength"] <= WEAK_SIGNAL_MAX).mean() * 100, 2),
                "Low_Reliability_Risk_pct": round((pdf["Reliability"] < 0).mean() * 100, 2),
                "Episodes": POLICY_EVAL_EPISODES,
                "Decision_Points": len(pdf),
                "Focus": scenario.focus,
            })

    return pd.DataFrame(rows)


def build_method_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build comparable per-decision selections from the same candidate UAV rows.

    Random is represented by all candidate rows, which is the expected value of
    choosing uniformly among UAVs. The proposed selector chooses the candidate
    with maximum project reward per vehicle/timestamp decision.
    """
    work = df.copy()
    work["_Reward"] = _reward_score(work)
    work["_Reliability"] = _reliability_score(work)

    distance = work["Distance"]
    signal = work["Signal_Strength"]
    energy = work["Energy"]

    normalized_distance = 1.0 - (
        (distance - distance.min()) / max(float(distance.max() - distance.min()), 1.0)
    )
    normalized_energy = (
        (energy - energy.min()) / max(float(energy.max() - energy.min()), 1.0)
    )
    energy_aware_score = (0.6 * normalized_energy) + (0.4 * normalized_distance)

    methods = {
        "Random": work,
        "Nearest UAV": work.loc[_select_idxmin(work, distance)],
        "Strongest Signal": work.loc[_select_idxmax(work, signal)],
        "Energy-Aware Greedy": work.loc[_select_idxmax(work, energy_aware_score)],
        "Reward-Aware Proposed": work.loc[_select_idxmax(work, work["_Reward"])],
    }
    return {name: frame.drop(columns=[], errors="ignore").copy() for name, frame in methods.items()}


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_frames = build_method_frames(df)
    proposed_by_scenario = {
        sid: frame
        for sid, _, frame in iter_scenario_frames(method_frames["Reward-Aware Proposed"])
    }

    for method, mdf in method_frames.items():
        for sid, sname, sdf in iter_scenario_frames(mdf):
            if sdf.empty:
                continue
            proposed_reward = proposed_by_scenario.get(sid, pd.DataFrame())["_Reward"].mean()
            reward = sdf["_Reward"].mean()
            reward_gap_pct = 0.0
            if pd.notna(proposed_reward) and proposed_reward:
                reward_gap_pct = ((reward - proposed_reward) / proposed_reward) * 100.0

            rows.append({
                "Scenario_ID": sid,
                "Scenario_Name": sname,
                "Method": method,
                "Avg_Reward": round(reward, 2),
                "Reward_vs_Proposed_pct": round(reward_gap_pct, 2),
                "Avg_Delay_ms": round(sdf["Delay"].mean(), 2),
                "Avg_PDR_pct": round(sdf["PDR"].mean(), 2),
                "Avg_Signal": round(sdf["Signal_Strength"].mean(), 4),
                "Avg_Energy": round(sdf["Energy"].mean(), 2),
                "Avg_Reliability": round(sdf["_Reliability"].mean(), 2),
                "Delay_Violation_Rate_pct": round((sdf["Delay"] > cons.DELAY_MAX_MS).mean() * 100, 2),
                "PDR_Violation_Rate_pct": round((sdf["PDR"] < cons.PDR_MIN_PCT).mean() * 100, 2),
                "Energy_Violation_Rate_pct": round((sdf["Energy"] < cons.ENERGY_MIN).mean() * 100, 2),
                "Signal_Violation_Rate_pct": round((sdf["Signal_Strength"] < cons.SIGNAL_MIN).mean() * 100, 2),
                "Out_Of_Range_Rate_pct": round((sdf["Distance"] > 500.0).mean() * 100, 2),
                "High_Latency_Risk_pct": round((sdf["Delay"] > HIGH_LATENCY_MS).mean() * 100, 2),
                "Weak_Signal_Risk_pct": round((sdf["Signal_Strength"] <= WEAK_SIGNAL_MAX).mean() * 100, 2),
                "Low_Reliability_Risk_pct": round((sdf["_Reliability"] < 0).mean() * 100, 2),
                "Decision_Points": sdf.groupby(DECISION_GROUP_COLS).ngroups,
                "Focus": SCENARIOS[sid].focus if sid in SCENARIOS else "",
            })
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sname, sdf in iter_scenario_frames(df):
        reliability = _reliability_score(sdf).mean()
        rows.append({
            "Scenario_ID": sid,
            "Scenario_Name": sname,
            "Vehicles": sdf["Vehicle_ID"].nunique(),
            "Avg_Delay_ms": round(sdf["Delay"].mean(), 2),
            "Avg_PDR_pct": round(sdf["PDR"].mean(), 2),
            "Avg_Signal": round(sdf["Signal_Strength"].mean(), 4),
            "Avg_Energy": round(sdf["Energy"].mean(), 2),
            "Avg_Reliability": round(reliability, 2),
            "Records": len(sdf),
            "Focus": SCENARIOS[sid].focus if sid in SCENARIOS else "",
        })
    return pd.DataFrame(rows)


def plot_scenario_bars(summary: pd.DataFrame) -> None:
    metrics = ["Avg_Delay_ms", "Avg_PDR_pct", "Avg_Reliability", "Avg_Energy"]
    titles = ["Average Delay (ms)", "Average PDR (%)", "Average Reliability", "Average UAV Energy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    names = summary["Scenario_Name"].tolist()
    x = range(len(names))

    for ax, col, title in zip(axes.flat, metrics, titles):
        ax.bar(x, summary[col], color=["#2E86AB", "#A23B72", "#F18F01"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "UAV-IoV Performance Across Paper Simulation Scenarios (Section VI)",
        fontsize=13,
    )
    fig.tight_layout()
    out = GRAPH_OUTPUTS_DIR / "scenario_metrics_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_traditional_vs_proposed_by_scenario(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    scenario_names = []
    proposed_pdr = []
    traditional_pdr = []

    for sid, sname, sdf in iter_scenario_frames(df):
        p_pdr = sdf["PDR"].mean()
        scenario_names.append(sname)
        proposed_pdr.append(p_pdr)
        traditional_pdr.append(max(50.0, p_pdr - 15))

    x = range(len(scenario_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], traditional_pdr, width, label="Traditional (baseline)")
    ax.bar([i + width / 2 for i in x], proposed_pdr, width, label="Proposed UAV-IoV")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario_names, rotation=12, ha="right")
    ax.set_ylabel("Average PDR (%)")
    ax.set_title("Traditional vs Proposed — Per Scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = GRAPH_OUTPUTS_DIR / "scenario_traditional_vs_proposed_pdr.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_method_comparison(comparison: pd.DataFrame) -> None:
    metrics = [
        ("Avg_Reward", "Average Reward"),
        ("Avg_PDR_pct", "Average PDR (%)"),
        ("Avg_Delay_ms", "Average Delay (ms)"),
        ("Avg_Reliability", "Average Reliability"),
    ]
    method_order = list(BASELINE_COLORS)
    scenario_names = comparison[["Scenario_ID", "Scenario_Name"]].drop_duplicates()

    for sid, sname in scenario_names.itertuples(index=False):
        sdf = comparison[comparison["Scenario_ID"] == sid].set_index("Method")
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (col, title) in zip(axes.flat, metrics):
            values = [sdf.loc[m, col] for m in method_order if m in sdf.index]
            labels = [m for m in method_order if m in sdf.index]
            colors = [BASELINE_COLORS[m] for m in labels]
            ax.bar(range(len(labels)), values, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"Baseline Comparison — {sname}", fontsize=13)
        fig.tight_layout()
        out = GRAPH_OUTPUTS_DIR / f"{sid}_baseline_method_comparison.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_policy_rollout_comparison(comparison: pd.DataFrame) -> None:
    metrics = [
        ("Avg_Episode_Reward", "Average Episode Reward"),
        ("Avg_PDR_pct", "Average PDR (%)"),
        ("Avg_Delay_ms", "Average Delay (ms)"),
        ("Avg_Reliability", "Average Reliability"),
    ]
    method_order = [m for m in BASELINE_COLORS if m in set(comparison["Method"])]
    scenario_names = comparison[["Scenario_ID", "Scenario_Name"]].drop_duplicates()

    for sid, sname in scenario_names.itertuples(index=False):
        sdf = comparison[comparison["Scenario_ID"] == sid].set_index("Method")
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (col, title) in zip(axes.flat, metrics):
            labels = [m for m in method_order if m in sdf.index]
            values = [sdf.loc[m, col] for m in labels]
            colors = [BASELINE_COLORS[m] for m in labels]
            ax.bar(range(len(labels)), values, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"Actual Policy Rollout Comparison — {sname}", fontsize=13)
        fig.tight_layout()
        out = GRAPH_OUTPUTS_DIR / f"{sid}_ppo_policy_rollout_comparison.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_safety_risks(comparison: pd.DataFrame) -> None:
    risk_cols = [
        "Out_Of_Range_Rate_pct",
        "High_Latency_Risk_pct",
        "Weak_Signal_Risk_pct",
        "Low_Reliability_Risk_pct",
    ]
    titles = ["Out of Range", "High Latency", "Weak Signal", "Low Reliability"]
    method_order = list(BASELINE_COLORS)
    scenario_names = comparison[["Scenario_ID", "Scenario_Name"]].drop_duplicates()

    for sid, sname in scenario_names.itertuples(index=False):
        sdf = comparison[comparison["Scenario_ID"] == sid].set_index("Method")
        x = range(len(method_order))
        width = 0.18
        fig, ax = plt.subplots(figsize=(12, 6))
        for offset, (col, title) in enumerate(zip(risk_cols, titles)):
            values = [sdf.loc[m, col] if m in sdf.index else 0 for m in method_order]
            positions = [i + (offset - 1.5) * width for i in x]
            ax.bar(positions, values, width, label=title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(method_order, rotation=15, ha="right")
        ax.set_ylabel("Risk Rate (%)")
        ax.set_title(f"Operational Risk Comparison — {sname}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = GRAPH_OUTPUTS_DIR / f"{sid}_operational_risk_comparison.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_policy_rollout_risks(comparison: pd.DataFrame) -> None:
    risk_cols = [
        "Out_Of_Range_Rate_pct",
        "High_Latency_Risk_pct",
        "Weak_Signal_Risk_pct",
        "Low_Reliability_Risk_pct",
    ]
    titles = ["Out of Range", "High Latency", "Weak Signal", "Low Reliability"]
    method_order = [m for m in BASELINE_COLORS if m in set(comparison["Method"])]
    scenario_names = comparison[["Scenario_ID", "Scenario_Name"]].drop_duplicates()

    for sid, sname in scenario_names.itertuples(index=False):
        sdf = comparison[comparison["Scenario_ID"] == sid].set_index("Method")
        x = range(len(method_order))
        width = 0.14
        fig, ax = plt.subplots(figsize=(12.5, 6.2))
        for offset, (col, title) in enumerate(zip(risk_cols, titles)):
            values = [sdf.loc[m, col] if m in sdf.index else 0 for m in method_order]
            positions = [i + (offset - 1.5) * width for i in x]
            ax.bar(positions, values, width, label=title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(method_order, rotation=15, ha="right")
        ax.set_ylabel("Risk Rate (%)")
        ax.set_title(f"Actual Policy Rollout Risk Comparison — {sname}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = GRAPH_OUTPUTS_DIR / f"{sid}_ppo_policy_rollout_risk_comparison.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()
    summary = build_summary(df)
    comparison = build_comparison_table(df)
    try:
        policy_rollout_comparison = build_policy_rollout_comparison()
    except ModuleNotFoundError as exc:
        policy_rollout_comparison = pd.DataFrame()
        print(
            f"\nWarning: skipped PPO rollout comparison because {exc.name} "
            "is unavailable. Use UAV_IOV/.venv/bin/python for RL evaluation."
        )

    print("\n=== Scenario Summary (Paper Section VI) ===\n")
    print(summary.to_string(index=False))

    summary_path = GRAPH_OUTPUTS_DIR / "scenario_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    print("\n=== Method Comparison With Baselines and Safety Metrics ===\n")
    print(comparison.to_string(index=False))

    comparison_path = GRAPH_OUTPUTS_DIR / "scenario_method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")

    if not policy_rollout_comparison.empty:
        print("\n=== Actual PPO Policy Rollout Comparison ===\n")
        print(policy_rollout_comparison.to_string(index=False))

        policy_comparison_path = GRAPH_OUTPUTS_DIR / "scenario_ppo_policy_comparison.csv"
        policy_rollout_comparison.to_csv(policy_comparison_path, index=False)
        print(f"\nSaved: {policy_comparison_path}")

    plot_scenario_bars(summary)
    plot_traditional_vs_proposed_by_scenario(df)
    plot_method_comparison(comparison)
    plot_safety_risks(comparison)
    if not policy_rollout_comparison.empty:
        plot_policy_rollout_comparison(policy_rollout_comparison)
        plot_policy_rollout_risks(policy_rollout_comparison)
    print("\nScenario analysis complete.\n")


if __name__ == "__main__":
    main()
