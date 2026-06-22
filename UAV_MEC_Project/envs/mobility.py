from __future__ import annotations

from typing import Tuple

import numpy as np

from .entities import Drone, Vehicle


class MobilityModel:
    """Updates drone and vehicle positions in discrete time."""

    def __init__(self, dt: float, area_size: Tuple[float, float], altitude_m: float):
        self.dt = float(dt)
        self.area_size = area_size
        self.altitude_m = float(altitude_m)

    def update_drone(self, drone: Drone, action: np.ndarray) -> None:
        action = drone.apply_action(action)
        old = drone.position.copy()
        drone.position[:2] = drone.position[:2] + action * self.dt
        drone.position[0] = float(np.clip(drone.position[0], 0.0, self.area_size[0]))
        drone.position[1] = float(np.clip(drone.position[1], 0.0, self.area_size[1]))
        drone.position[2] = self.altitude_m
        drone.total_distance_m += float(np.linalg.norm(drone.position - old))

    def update_vehicle(self, vehicle: Vehicle) -> None:
        if not vehicle.route:
            return
        target = vehicle.route[vehicle.route_index]
        direction = target - vehicle.position
        distance = float(np.linalg.norm(direction))
        travel = vehicle.speed_mps * self.dt
        if distance <= travel:
            vehicle.position = target.copy()
            vehicle.route_index = (vehicle.route_index + 1) % len(vehicle.route)
        else:
            vehicle.position = vehicle.position + direction / (distance + 1e-8) * travel
