from __future__ import annotations

from typing import List, Optional

import numpy as np

from .entities import Drone, Task


class CommunicationModel:
    """Distance-based V2D assignment from equation (4)."""

    def __init__(self, comm_range_m: float):
        self.comm_range_m = float(comm_range_m)

    def nearest_drone(self, task: Task, drones: List[Drone]) -> tuple[Optional[int], float]:
        best_id = None
        best_dist = float("inf")
        src3 = np.array([task.source_position[0], task.source_position[1], drones[0].position[2]], dtype=np.float32)
        for drone in drones:
            dist = float(np.linalg.norm(src3 - drone.position))
            if dist <= self.comm_range_m and dist < best_dist:
                best_id = drone.drone_id
                best_dist = dist
        return best_id, best_dist
