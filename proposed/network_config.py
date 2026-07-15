"""Shared fleet sizes for dataset generation, RL env, and analysis."""

from scenarios import NUM_SAMPLES, NUM_UAVS, list_scenario_ids

# Default RL env uses urban canyon scale (largest fleet in paper experiments)
NUM_VEHICLES = 20
SCENARIO_IDS = list_scenario_ids()
