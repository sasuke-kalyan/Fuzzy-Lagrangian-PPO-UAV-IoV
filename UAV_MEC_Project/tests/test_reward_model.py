import numpy as np

from envs.entities import Drone, Task
from envs.reward import RewardCalculator


def test_reward_equation():
    drones = [Drone(0, np.array([0.0, 0.0, 100.0], dtype=np.float32), 100.0, 20.0)]
    tasks = [Task(0, 0, 0.0, 1.0, 1.0, np.array([3.0, 4.0], dtype=np.float32))]
    calc = RewardCalculator(w_comp=10.0, w_dist=1.0, w_energy=0.5, time_penalty=-1.0)
    reward, info = calc.compute(completed_count=2, drones=drones, pending_tasks=tasks, step_energy_j=4.0)
    assert abs(info["d_min"] - 5.0) < 1e-6
    assert abs(reward - (20.0 - 5.0 - 2.0 - 1.0)) < 1e-6
