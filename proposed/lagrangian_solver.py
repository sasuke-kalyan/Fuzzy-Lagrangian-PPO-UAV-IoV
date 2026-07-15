"""
Lagrangian penalized objective for maximization:
    L = objective - sum_i lambda_i * violation_i

Dual update (after seeing violations on a batch):
    lambda_i <- max(0, lambda_i + lr_lambda * mean_violation_i)
"""

from dataclasses import dataclass, field
import pandas as pd

import constraints as cons
from communication_model import base_reward as scalar_base_reward


@dataclass
class LagrangianMultipliers:
    delay: float = 1.0
    pdr: float = 1.0
    energy: float = 1.0
    signal: float = 0.5


@dataclass
class LagrangianConfig:
    lr_lambda: float = 0.1      # dual step size
    n_iterations: int = 5       # primal-dual rounds on full dataframe


def base_reward(row) -> float:
    """Per-row reward (shared with PPO objective)."""
    return scalar_base_reward(
        row["Signal_Strength"], row["PDR"], row["Delay"]
    )


def base_routing_score(row) -> float:
    """Same idea as smart_routing_optimization.py."""
    return (
        float(row["Signal_Strength"]) * 40.0
        + float(row["PDR"]) * 0.3
        + float(row["Energy"]) * 0.2
        - float(row["Delay"]) * 0.5
    )


def lagrangian_value(objective: float, v: dict, lam: LagrangianMultipliers) -> float:
    penalty = (
        lam.delay * v["delay"]
        + lam.pdr * v["pdr"]
        + lam.energy * v["energy"]
        + lam.signal * v["signal"]
    )
    return objective - penalty


def add_lagrangian_columns(df: pd.DataFrame, lam: LagrangianMultipliers, objective_col: str) -> pd.DataFrame:
    """
    objective_col: name of per-row objective already in df, OR pass None and use base_reward.
    Here we compute from row each time via objective_fn in caller — simplified below.
    """
    out = df.copy()
    L_vals = []
    viol_delay = []
    viol_pdr = []
    viol_energy = []
    viol_signal = []

    for _, row in out.iterrows():
        v = cons.all_violations(row)
        if objective_col == "Routing_Score":
            obj = base_routing_score(row)
        else:
            obj = base_reward(row)

        L_vals.append(lagrangian_value(obj, v, lam))
        viol_delay.append(v["delay"])
        viol_pdr.append(v["pdr"])
        viol_energy.append(v["energy"])
        viol_signal.append(v["signal"])

    out["Violation_Delay"] = viol_delay
    out["Violation_PDR"] = viol_pdr
    out["Violation_Energy"] = viol_energy
    out["Violation_Signal"] = viol_signal
    out["Lagrangian_Score"] = L_vals
    return out


def mean_violations(df: pd.DataFrame) -> dict:
    return {
        "delay": df["Violation_Delay"].mean(),
        "pdr": df["Violation_PDR"].mean(),
        "energy": df["Violation_Energy"].mean(),
        "signal": df["Violation_Signal"].mean(),
    }


def update_multipliers(lam: LagrangianMultipliers, mean_v: dict, lr: float) -> LagrangianMultipliers:
    return LagrangianMultipliers(
        delay=max(0.0, lam.delay + lr * mean_v["delay"]),
        pdr=max(0.0, lam.pdr + lr * mean_v["pdr"]),
        energy=max(0.0, lam.energy + lr * mean_v["energy"]),
        signal=max(0.0, lam.signal + lr * mean_v["signal"]),
    )


def run_primal_dual(df: pd.DataFrame, objective_col: str = "Reward") -> tuple[pd.DataFrame, LagrangianMultipliers]:
    """
    Iteratively re-score rows and increase lambdas when constraints are violated on average.
    """
    cfg = LagrangianConfig()
    lam = LagrangianMultipliers()
    working = df.copy()

    for it in range(cfg.n_iterations):
        working = add_lagrangian_columns(working, lam, objective_col)
        mv = mean_violations(working)
        lam = update_multipliers(lam, mv, cfg.lr_lambda)
        print(f"[Lagrangian iter {it + 1}] mean violations: {mv}")
        print(f"[Lagrangian iter {it + 1}] lambdas: delay={lam.delay:.3f}, pdr={lam.pdr:.3f}, "
              f"energy={lam.energy:.3f}, signal={lam.signal:.3f}")

    return working, lam
