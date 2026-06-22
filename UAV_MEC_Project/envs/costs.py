from __future__ import annotations

from typing import Dict, List

import numpy as np

from .entities import Drone


class CostCalculator:
    """Safety costs from equations (12) and (13)."""

    def __init__(self, safe_distance_m: float, critical_battery_j: float):
        self.safe_distance_m = float(safe_distance_m)
        self.critical_battery_j = float(critical_battery_j)

    def compute(self, drones: List[Drone]) -> tuple[np.ndarray, Dict[str, float]]:
        collision = 0.0
        min_pair_dist = float("inf")
        for i, drone_i in enumerate(drones):
            for drone_j in drones[i + 1 :]:
                dist = float(np.linalg.norm(drone_i.position - drone_j.position))
                min_pair_dist = min(min_pair_dist, dist)
                if dist < self.safe_distance_m:
                    collision = 1.0
        low_battery = 1.0 if any(d.battery_j < self.critical_battery_j for d in drones) else 0.0
        return np.array([collision, low_battery], dtype=np.float32), {
            "collision": collision,
            "low_battery": low_battery,
            "min_inter_drone_distance": min_pair_dist if min_pair_dist != float("inf") else 0.0,
        }
