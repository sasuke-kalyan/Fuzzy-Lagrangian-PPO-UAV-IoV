import numpy as np

from envs.costs import CostCalculator
from envs.entities import Drone


def test_cost_collision_and_battery():
    drones = [
        Drone(0, np.array([0.0, 0.0, 100.0], dtype=np.float32), 100000.0, 20.0),
        Drone(1, np.array([10.0, 0.0, 100.0], dtype=np.float32), 1000.0, 20.0),
    ]
    calc = CostCalculator(safe_distance_m=50.0, critical_battery_j=50000.0)
    costs, info = calc.compute(drones)
    assert costs.tolist() == [1.0, 1.0]
    assert info["collision"] == 1.0
    assert info["low_battery"] == 1.0
