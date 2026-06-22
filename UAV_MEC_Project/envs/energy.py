from __future__ import annotations

import numpy as np


class EnergyModel:
    """Drone energy model from the paper.

    Equation (5): E_d(dt) = (P_idle + c_move * s_d(t)^2) * dt
    Equation (6): B_d(t+dt) = B_d(t) - E_d(dt)
    """

    def __init__(self, idle_power_w: float, move_coeff: float, dt: float):
        self.idle_power_w = float(idle_power_w)
        self.move_coeff = float(move_coeff)
        self.dt = float(dt)

    def step_energy(self, action: np.ndarray) -> float:
        speed = float(np.linalg.norm(action))
        return (self.idle_power_w + self.move_coeff * speed * speed) * self.dt

    def update_battery(self, battery_j: float, action: np.ndarray) -> tuple[float, float]:
        energy = self.step_energy(action)
        return max(0.0, battery_j - energy), energy
