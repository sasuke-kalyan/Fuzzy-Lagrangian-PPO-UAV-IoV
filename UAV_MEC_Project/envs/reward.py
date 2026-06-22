from __future__ import annotations

from typing import Dict, List

import numpy as np

from .entities import Drone, Task


class RewardCalculator:
    """Reward shaping from equation (11).

    Missing paper details: reward weights are assumed configurable.
    """

    def __init__(self, w_comp: float, w_dist: float, w_energy: float, time_penalty: float):
        self.w_comp = float(w_comp)
        self.w_dist = float(w_dist)
        self.w_energy = float(w_energy)
        self.time_penalty = float(time_penalty)

    def nearest_task_distance(self, drones: List[Drone], pending_tasks: List[Task]) -> float:
        if not drones or not pending_tasks:
            return 0.0
        distances = []
        for drone in drones:
            best = min(
                float(np.linalg.norm(task.source_position - drone.position[:2]))
                for task in pending_tasks
                if not task.completed and not task.failed
            ) if any(not t.completed and not t.failed for t in pending_tasks) else 0.0
            distances.append(best)
        return float(np.mean(distances)) if distances else 0.0

    def compute(
        self,
        completed_count: int,
        drones: List[Drone],
        pending_tasks: List[Task],
        step_energy_j: float,
    ) -> tuple[float, Dict[str, float]]:
        d_min = self.nearest_task_distance(drones, pending_tasks)
        task_reward = self.w_comp * completed_count
        distance_penalty = self.w_dist * d_min
        energy_penalty = self.w_energy * step_energy_j
        reward = task_reward - distance_penalty - energy_penalty + self.time_penalty
        return float(reward), {
            "task_reward": float(task_reward),
            "distance_penalty": float(distance_penalty),
            "energy_penalty": float(energy_penalty),
            "time_penalty": float(self.time_penalty),
            "d_min": float(d_min),
        }
