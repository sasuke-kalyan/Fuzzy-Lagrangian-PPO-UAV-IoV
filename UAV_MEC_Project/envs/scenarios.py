from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .entities import Drone, Vehicle


class ScenarioBuilder:
    """Builds simplified research-grade scenarios from the paper.

    Assumption: the paper describes road layouts qualitatively but does not
    give exact coordinates, so deterministic synthetic roads are used.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.area = tuple(config["env"]["area_size"])

    def build(self, rng: np.random.Generator) -> tuple[List[Drone], List[Vehicle]]:
        scenario = self.config["scenario"]["name"]
        drones = self._build_drones()
        if scenario == "suburban_crossroads":
            vehicles = self._crossroads_vehicles(15)
        elif scenario == "emergency_response":
            vehicles = self._crossroads_vehicles(5)
        else:
            vehicles = self._urban_canyon_vehicles(20)
        return drones, vehicles

    def emergency_vehicles(self, start_id: int, count: int = 10) -> List[Vehicle]:
        vehicles = []
        center = np.array([self.area[0] * 0.65, self.area[1] * 0.55], dtype=np.float32)
        for i in range(count):
            offset = (i - count / 2) * 12.0
            route = [
                center + np.array([-250.0, offset], dtype=np.float32),
                center + np.array([250.0, offset], dtype=np.float32),
            ]
            vehicles.append(Vehicle(start_id + i, route[0].copy(), route, self.config["env"]["vehicle_speed_mps"]))
        return vehicles

    def _build_drones(self) -> List[Drone]:
        n = self.config["env"]["num_drones"]
        altitude = self.config["env"]["altitude_m"]
        battery = self.config["env"]["initial_battery_j"]
        max_speed = self.config["env"]["max_drone_speed_mps"]
        xs = np.linspace(self.area[0] * 0.25, self.area[0] * 0.75, n)
        return [
            Drone(i, np.array([xs[i], self.area[1] * 0.5, altitude], dtype=np.float32), battery, max_speed)
            for i in range(n)
        ]

    def _urban_canyon_vehicles(self, n: int) -> List[Vehicle]:
        lanes = [self.area[1] * y for y in (0.35, 0.5, 0.65)]
        vehicles = []
        for i in range(n):
            y = lanes[i % len(lanes)]
            route = [np.array([0.0, y], dtype=np.float32), np.array([self.area[0], y], dtype=np.float32)]
            pos = route[i % 2].copy()
            vehicles.append(Vehicle(i, pos, route, self.config["env"]["vehicle_speed_mps"]))
        return vehicles

    def _crossroads_vehicles(self, n: int) -> List[Vehicle]:
        xs = [self.area[0] * x for x in (0.25, 0.5, 0.75)]
        ys = [self.area[1] * y for y in (0.25, 0.5, 0.75)]
        vehicles = []
        for i in range(n):
            if i % 2 == 0:
                x = xs[(i // 2) % len(xs)]
                route = [np.array([x, 0.0], dtype=np.float32), np.array([x, self.area[1]], dtype=np.float32)]
            else:
                y = ys[(i // 2) % len(ys)]
                route = [np.array([0.0, y], dtype=np.float32), np.array([self.area[0], y], dtype=np.float32)]
            vehicles.append(Vehicle(i, route[0].copy(), route, self.config["env"]["vehicle_speed_mps"]))
        return vehicles
