"""Episode horizon = steps per episode (aligned with dataset timestamps)."""

from __future__ import annotations

from scenarios import NUM_SAMPLES
from training_config import DEFAULT_TRAINING_EPISODES

EPISODE_HORIZON_STEPS = NUM_SAMPLES
TRAINING_EPISODES = DEFAULT_TRAINING_EPISODES
