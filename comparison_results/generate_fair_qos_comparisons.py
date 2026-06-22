#!/usr/bin/env python3
"""Generate fair shared-QoS comparisons for LP-PPO and baseline algorithms.

This avoids comparing raw rewards from different environments. Instead, every
method is evaluated with the same QoS score formula using common metrics:
delay, PDR, signal, and energy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "comparison_results" / "fair_graphs"
CSV_OUT = ROOT / "comparison_results" / "fair_qos_comparison.csv"
MPL_CACHE = ROOT / "comparison_results" / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCENARIOS = {
    "emergency": {
        "title": "Emergency Scenario Fair QoS Comparison",
        "lp_scenario_id": "emergency_response",
        "baseline_tokens": ("emergency",),
        "output": "emergency_fair_qos_comparison.png",
    },
    "suburban": {
        "title": "Suburban Scenario Fair QoS Comparison",
        "lp_scenario_id": "suburban_crossroads",
        "baseline_tokens": ("suburban",),
        "output": "suburban_fair_qos_comparison.png",
    },
    "urban": {
        "title": "Urban Scenario Fair QoS Comparison",
        "lp_scenario_id": "urban_canyon",
        "baseline_tokens": ("urban",),
        "output": "urban_fair_qos_comparison.png",
    },
}

BASELINE_ALGORITHMS = {
    "Greedy PPO": ("greedy", "ppo"),
    "LC-MAPO": ("lc", "mapo"),
    "MADDPG": ("maddpg",),
}

METHOD_ORDER = [
    "Fuzzy LP-PPO",
    "Greedy PPO",
    "LC-MAPO",
    "MADDPG",
]

COLORS = {
    "Fuzzy LP-PPO": "#C44E52",
    "Greedy PPO": "#4C72B0",
    "LC-MAPO": "#55A868",
    "MADDPG": "#8172B2",
}

# Shared QoS thresholds and weights.
DELAY_MAX_MS = 100.0
PDR_MIN_PCT = 48.0
ENERGY_MIN_PCT = 8.0
SIGNAL_MIN = 0.05
PENALTY_DELAY = 0.25
PENALTY_PDR = 0.15
PENALTY_ENERGY = 0.15
PENALTY_SIGNAL = 0.08
SIGNAL_WEIGHT = 120.0
PDR_WEIGHT = 1.5
DELAY_WEIGHT = 1.2
ENERGY_WEIGHT = 0.8


@dataclass(frozen=True)
class FairScoreRow:
    scenario: str
    method: str
    shared_qos_score: float
    avg_delay_ms: float
    avg_pdr_pct: float
    avg_signal: float
    avg_energy_pct: float
    source: Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tokens_in_name(path: Path, tokens: tuple[str, ...]) -> bool:
    parts = {
        part
        for part in path.name.lower().replace("-", "_").split("_")
        if part
    }
    return all(token in parts for token in tokens)


def shared_qos_score(
    *,
    delay_ms: float,
    pdr_pct: float,
    signal: float,
    energy_pct: float,
) -> float:
    penalty = 0.0
    penalty += PENALTY_DELAY * max(0.0, delay_ms - DELAY_MAX_MS)
    penalty += PENALTY_PDR * max(0.0, PDR_MIN_PCT - pdr_pct)
    penalty += PENALTY_ENERGY * max(0.0, ENERGY_MIN_PCT - energy_pct)
    penalty += PENALTY_SIGNAL * max(0.0, SIGNAL_MIN - signal)

    return (
        SIGNAL_WEIGHT * signal
        + PDR_WEIGHT * pdr_pct
        - DELAY_WEIGHT * delay_ms
        + ENERGY_WEIGHT * energy_pct
        - penalty
    )


def score_baseline_path(path: Path) -> tuple[int, int, str]:
    name = path.parent.name.lower()
    qos_rank = 0 if name.startswith("qos_") else 1
    new_rank = 1 if name.startswith("new_") else 2
    return (qos_rank, new_rank, name)


def discover_baseline_log(scenario_key: str, algorithm: str) -> Path:
    results_dir = ROOT / "UAV_MEC_Project" / "results"
    scenario_tokens = SCENARIOS[scenario_key]["baseline_tokens"]
    algorithm_tokens = BASELINE_ALGORITHMS[algorithm]
    matches = [
        path
        for path in sorted(results_dir.glob("*/training_log.json"))
        if tokens_in_name(path.parent, scenario_tokens)
        and tokens_in_name(path.parent, algorithm_tokens)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No {algorithm} baseline log found for {scenario_key} in {results_dir}"
        )
    return sorted(matches, key=score_baseline_path)[0]


def load_baseline_row(scenario_key: str, algorithm: str) -> FairScoreRow | None:
    path = discover_baseline_log(scenario_key, algorithm)
    rows = read_json(path)
    required = {"avg_delay_ms", "avg_pdr_pct", "avg_signal", "avg_energy_pct"}
    if not rows or not required.issubset(rows[0]):
        print(
            f"  {scenario_key:9s} | {algorithm:10s} | SKIP: "
            f"missing QoS metrics in {path}"
        )
        return None

    avg_delay = sum(float(row["avg_delay_ms"]) for row in rows) / len(rows)
    avg_pdr = sum(float(row["avg_pdr_pct"]) for row in rows) / len(rows)
    avg_signal = sum(float(row["avg_signal"]) for row in rows) / len(rows)
    avg_energy = sum(float(row["avg_energy_pct"]) for row in rows) / len(rows)

    return FairScoreRow(
        scenario=scenario_key,
        method=algorithm,
        shared_qos_score=shared_qos_score(
            delay_ms=avg_delay,
            pdr_pct=avg_pdr,
            signal=avg_signal,
            energy_pct=avg_energy,
        ),
        avg_delay_ms=avg_delay,
        avg_pdr_pct=avg_pdr,
        avg_signal=avg_signal,
        avg_energy_pct=avg_energy,
        source=path,
    )


def load_lp_ppo_row(scenario_key: str) -> FairScoreRow:
    path = ROOT / "UAV_IOV" / "graph_outputs" / "scenario_ppo_policy_comparison.csv"
    df = pd.read_csv(path)
    scenario_id = SCENARIOS[scenario_key]["lp_scenario_id"]
    match = df[
        (df["Scenario_ID"] == scenario_id)
        & (df["Method"] == "Trained PPO Model")
    ]
    if match.empty:
        raise FileNotFoundError(
            f"No Trained PPO Model row found for {scenario_id} in {path}"
        )
    row = match.iloc[0]

    # UAV_IOV stores energy on a 0-50 scale in this CSV; convert to percent.
    energy_pct = min(100.0, float(row["Avg_Energy"]) * 2.0)
    avg_delay = float(row["Avg_Delay_ms"])
    avg_pdr = float(row["Avg_PDR_pct"])
    avg_signal = float(row["Avg_Signal"])

    return FairScoreRow(
        scenario=scenario_key,
        method="Fuzzy LP-PPO",
        shared_qos_score=shared_qos_score(
            delay_ms=avg_delay,
            pdr_pct=avg_pdr,
            signal=avg_signal,
            energy_pct=energy_pct,
        ),
        avg_delay_ms=avg_delay,
        avg_pdr_pct=avg_pdr,
        avg_signal=avg_signal,
        avg_energy_pct=energy_pct,
        source=path,
    )


def plot_scenario(scenario_key: str, rows: list[FairScoreRow]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: METHOD_ORDER.index(row.method))
    methods = [row.method for row in ordered]
    scores = [row.shared_qos_score for row in ordered]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.bar(methods, scores, color=[COLORS[method] for method in methods])
    ax.set_title(SCENARIOS[scenario_key]["title"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Shared QoS Score")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()

    out_path = OUT_DIR / SCENARIOS[scenario_key]["output"]
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    all_rows: list[FairScoreRow] = []
    generated: list[Path] = []

    print("Fair comparison sources:")
    for scenario_key in SCENARIOS:
        rows = [load_lp_ppo_row(scenario_key)]
        for algorithm in BASELINE_ALGORITHMS:
            row = load_baseline_row(scenario_key, algorithm)
            if row is not None:
                rows.append(row)
        all_rows.extend(rows)
        generated.append(plot_scenario(scenario_key, rows))
        for row in rows:
            print(
                f"  {scenario_key:9s} | {row.method:10s} | "
                f"score={row.shared_qos_score:8.2f} | {row.source}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.__dict__ for row in all_rows]).to_csv(CSV_OUT, index=False)

    print("\nGenerated fair comparison graphs:")
    for path in generated:
        print(f"  {path}")
    print(f"\nSaved fair comparison table: {CSV_OUT}")


if __name__ == "__main__":
    main()
