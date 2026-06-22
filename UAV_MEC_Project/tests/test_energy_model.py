import numpy as np

from envs.energy import EnergyModel


def test_energy_model_equation():
    model = EnergyModel(idle_power_w=20.0, move_coeff=0.1, dt=0.1)
    action = np.array([3.0, 4.0], dtype=np.float32)
    energy = model.step_energy(action)
    assert abs(energy - ((20.0 + 0.1 * 25.0) * 0.1)) < 1e-6
    battery, used = model.update_battery(100.0, action)
    assert abs(used - energy) < 1e-6
    assert abs(battery - (100.0 - energy)) < 1e-6
