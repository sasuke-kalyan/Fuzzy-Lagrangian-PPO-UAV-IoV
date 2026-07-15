"""Episode-based RL training defaults (configurable via CLI)."""

from __future__ import annotations

from scenarios import NUM_SAMPLES

# One RL episode = one full scenario timestep horizon (paper-scaled dataset length)
STEPS_PER_EPISODE = NUM_SAMPLES

DEFAULT_TRAINING_EPISODES = 100
DEFAULT_PPO_LEARNING_RATE = 3e-4
DEFAULT_PPO_N_STEPS = 512
DEFAULT_PPO_BATCH_SIZE = 128
DEFAULT_PPO_N_EPOCHS = 10
