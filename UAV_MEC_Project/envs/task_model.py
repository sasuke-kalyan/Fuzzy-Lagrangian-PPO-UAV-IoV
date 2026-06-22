from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .entities import Task, Vehicle


@dataclass
class TaskGenerator:
    data_size_range_mbits: tuple[float, float]
    compute_range_gcycles: tuple[float, float]
    deadline_s: float
    base_probability: float

    def maybe_generate(
        self,
        rng: np.random.Generator,
        vehicle: Vehicle,
        time_s: float,
        next_task_id: int,
        probability_multiplier: float = 1.0,
    ) -> List[Task]:
        probability = min(1.0, self.base_probability * probability_multiplier)
        if rng.random() > probability:
            return []
        return [
            Task(
                task_id=next_task_id,
                vehicle_id=vehicle.vehicle_id,
                created_at=time_s,
                data_size_mbits=float(rng.uniform(*self.data_size_range_mbits)),
                compute_demand_gcycles=float(rng.uniform(*self.compute_range_gcycles)),
                source_position=vehicle.position.copy(),
                deadline_s=self.deadline_s,
            )
        ]
