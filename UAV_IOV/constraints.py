"""
Constraint definitions for UAV-IoV (delay, PDR, energy).
Violations are >= 0; 0 means satisfied.

Thresholds loosened for 2000x2000 m RL env — heavy penalties caused negative rewards.
"""

# Looser QoS thresholds (aligned with env link statistics)
DELAY_MAX_MS = 100.0
PDR_MIN_PCT = 48.0
ENERGY_MIN = 8.0
SIGNAL_MIN = 0.05


def violation_delay(delay: float) -> float:
    """delay <= DELAY_MAX_MS"""
    return max(0.0, float(delay) - DELAY_MAX_MS)


def violation_pdr(pdr: float) -> float:
    """pdr >= PDR_MIN_PCT"""
    return max(0.0, PDR_MIN_PCT - float(pdr))


def violation_energy(energy: float) -> float:
    """energy >= ENERGY_MIN"""
    return max(0.0, ENERGY_MIN - float(energy))


def violation_signal(signal: float) -> float:
    """signal >= SIGNAL_MIN"""
    return max(0.0, SIGNAL_MIN - float(signal))


def all_violations(row) -> dict:
    """row: pandas Series with Delay, PDR, Energy, Signal_Strength."""
    return {
        "delay": violation_delay(row["Delay"]),
        "pdr": violation_pdr(row["PDR"]),
        "energy": violation_energy(row["Energy"]),
        "signal": violation_signal(row["Signal_Strength"]),
    }
