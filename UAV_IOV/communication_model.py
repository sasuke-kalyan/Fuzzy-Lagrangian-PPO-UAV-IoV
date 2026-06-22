"""
Shared UAV–vehicle link physics and reward (used by dataset, env, and RL).
"""

from __future__ import annotations

import numpy as np

import constraints as cons

COMMUNICATION_RANGE = 500.0
AREA_SIZE = 2000.0  # Paper Table III / Section VI: 2000 x 2000 m^2
ENERGY_DRAIN_MIN = 1
ENERGY_DRAIN_MAX = 3

# Shared QoS reward weights and penalties used for fair comparison.
PENALTY_DELAY = 0.35
PENALTY_PDR = 0.20
PENALTY_ENERGY = 0.60
PENALTY_SIGNAL = 0.35
SIGNAL_WEIGHT = 190.0
PDR_WEIGHT = 1.5
DELAY_WEIGHT = 1.2
ENERGY_WEIGHT = 2.2
W_COMPLETION = 15.0
W_DISTANCE = 0.002
W_ENERGY_USED = 0.08
TIME_PENALTY = -0.05
FUZZY_PRIORITY_WEIGHT = 20.0


def _triangular_membership(value: float, left: float, peak: float, right: float) -> float:
    value = float(value)
    if value <= left or value >= right:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return (value - left) / max(peak - left, 1e-9)
    return (right - value) / max(right - peak, 1e-9)


def fuzzy_priority(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
    scenario_focus: str = "",
) -> float:
    """Rule-based priority in [0, 1] for urgent or weak-QoS decisions."""
    delay_urgency = max(
        0.0,
        min(1.0, (float(delay) - 55.0) / 45.0),
    )
    pdr_risk = max(0.0, min(1.0, (55.0 - float(pdr)) / 7.0))
    signal_risk = max(0.0, min(1.0, (0.18 - float(signal)) / 0.13))
    energy_risk = max(0.0, min(1.0, (35.0 - energy_to_percent(energy)) / 35.0))
    emergency_boost = 1.0 if scenario_focus == "dynamic_adaptation" else 0.0

    priority = (
        0.35 * delay_urgency
        + 0.25 * pdr_risk
        + 0.25 * signal_risk
        + 0.15 * energy_risk
        + 0.10 * emergency_boost
    )
    return max(0.0, min(1.0, priority))


def compute_3d_distance(
    vehicle_x: float,
    vehicle_y: float,
    uav_x: float,
    uav_y: float,
    uav_z: float,
) -> float:
    return float(
        np.sqrt(
            (vehicle_x - uav_x) ** 2
            + (vehicle_y - uav_y) ** 2
            + uav_z**2
        )
    )


def compute_link_metrics(
    vehicle_x: float,
    vehicle_y: float,
    uav_x: float,
    uav_y: float,
    uav_z: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Distance, signal, delay, and PDR for one vehicle–UAV pair."""
    if rng is None:
        rng = np.random.default_rng()

    distance = compute_3d_distance(vehicle_x, vehicle_y, uav_x, uav_y, uav_z)

    if distance <= COMMUNICATION_RANGE:
        signal = max(0.1, 1.0 - (distance / COMMUNICATION_RANGE))
    else:
        signal = 0.05

    delay = min(100.0, distance / 10.0 + rng.integers(1, 10))
    pdr = max(50.0, signal * 100.0 - delay / 2.0)

    return {
        "Distance": round(distance, 2),
        "Signal_Strength": round(signal, 4),
        "Delay": round(delay, 2),
        "PDR": round(pdr, 2),
    }


def base_reward(signal: float, pdr: float, delay: float) -> float:
    """Legacy linear score (may be negative); used by Lagrangian / CSV analysis."""
    return float(signal) * 100.0 + float(pdr) - float(delay)


def energy_to_percent(energy: float) -> float:
    """UAV_IOV energy is stored on a 0-50 scale; baselines use 0-100 pct."""
    return min(100.0, max(0.0, float(energy) * 2.0))


def qos_score(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
) -> float:
    """Same QoS reward equation used by the baseline project."""
    energy_pct = energy_to_percent(energy)
    return (
        SIGNAL_WEIGHT * float(signal)
        + PDR_WEIGHT * float(pdr)
        - DELAY_WEIGHT * float(delay)
        + ENERGY_WEIGHT * energy_pct
    )


def constraint_penalty(signal: float, pdr: float, delay: float, energy: float) -> float:
    energy_pct = energy_to_percent(energy)
    return (
        PENALTY_DELAY * cons.violation_delay(delay)
        + PENALTY_PDR * cons.violation_pdr(pdr)
        + PENALTY_ENERGY * cons.violation_energy(energy_pct)
        + PENALTY_SIGNAL * cons.violation_signal(signal)
    )


def lagrangian_constraint_penalty(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
    multipliers: dict[str, float] | None = None,
) -> float:
    if multipliers is None:
        return constraint_penalty(signal, pdr, delay, energy)

    energy_pct = energy_to_percent(energy)
    return (
        float(multipliers.get("delay", PENALTY_DELAY)) * cons.violation_delay(delay)
        + float(multipliers.get("pdr", PENALTY_PDR)) * cons.violation_pdr(pdr)
        + float(multipliers.get("energy", PENALTY_ENERGY)) * cons.violation_energy(energy_pct)
        + float(multipliers.get("signal", PENALTY_SIGNAL)) * cons.violation_signal(signal)
    )


def shaped_reward(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
) -> float:
    """Shared QoS reward: quality score minus the same soft penalties."""
    score = qos_score(signal, pdr, delay, energy)
    penalty = constraint_penalty(signal, pdr, delay, energy)
    return score - penalty


def baseline_task_reward(
    completed_count: int,
    distance: float,
    energy_used: float,
) -> float:
    """Task reward equation used by the baseline project."""
    task_reward = W_COMPLETION * int(completed_count)
    distance_penalty = W_DISTANCE * float(distance)
    energy_penalty = W_ENERGY_USED * float(energy_used)
    return task_reward - distance_penalty - energy_penalty + TIME_PENALTY


def baseline_compatible_reward(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
    distance: float,
    energy_used: float,
    completed_count: int = 1,
    scenario_focus: str = "",
    lagrangian_multipliers: dict[str, float] | None = None,
) -> float:
    """
    Baseline-compatible reward:
    task reward - distance penalty - energy penalty + time penalty + QoS reward.
    """
    priority_bonus = FUZZY_PRIORITY_WEIGHT * fuzzy_priority(
        signal,
        pdr,
        delay,
        energy,
        scenario_focus,
    )
    qos = qos_score(signal, pdr, delay, energy) - lagrangian_constraint_penalty(
        signal,
        pdr,
        delay,
        energy,
        lagrangian_multipliers,
    )
    return baseline_task_reward(
        completed_count,
        distance,
        energy_used,
    ) + qos + priority_bonus


def traditional_step_reward(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
) -> float:
    """
    Proxy for fixed / greedy traditional UAV selection (lower than trained PPO).
    Used only for reference lines on Episode vs Reward plots.
    """
    return shaped_reward(signal, pdr, delay, energy) * 0.72 - 8.0
