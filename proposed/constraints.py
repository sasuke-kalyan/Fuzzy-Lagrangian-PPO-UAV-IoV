"""Active UAV-IoV QoS constraints used by the Lagrangian updates.

Violations are non-negative and are deliberately inside the communication
model's attainable range, so every multiplier can receive a learning signal.
The values match the hard candidate-screening thresholds.
"""

# Active QoS thresholds aligned with hard candidate screening.
DELAY_MAX_MS = 95.0
PDR_MIN_PCT = 55.0
ENERGY_MIN = 20.0
SIGNAL_MIN = 0.10


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
