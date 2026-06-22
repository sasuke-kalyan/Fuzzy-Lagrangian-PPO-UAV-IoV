import numpy as np

from common.config import load_config
from envs import UAVIoVEnv


def test_env_step_smoke():
    cfg = load_config("configs/default_lc_mapo.json")
    env = UAVIoVEnv(cfg)
    obs, _ = env.reset(seed=1)
    action = np.zeros((env.num_drones, env.action_dim), dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert obs["global_state"].shape == (env.state_dim,)
    assert next_obs["local_obs"].shape == (env.num_drones, env.obs_dim)
    assert isinstance(reward, float)
    assert "costs" in info
