from __future__ import annotations

import numpy as np


class GreedyPolicy:
    """Nearest-task heuristic from equations (22)-(23)."""

    def __init__(self, max_speed_mps: float):
        self.max_speed_mps = float(max_speed_mps)

    def act(self, env) -> np.ndarray:
        actions = []
        active_tasks = [task for task in env.pending_tasks if not task.completed and not task.failed]
        for drone in env.drones:
            if not active_tasks:
                actions.append(np.zeros(2, dtype=np.float32))
                continue
            target = min(active_tasks, key=lambda t: float(np.linalg.norm(t.source_position - drone.position[:2])))
            direction = target.source_position - drone.position[:2]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                actions.append(np.zeros(2, dtype=np.float32))
            else:
                actions.append((direction / norm * self.max_speed_mps).astype(np.float32))
        return np.stack(actions, axis=0)
