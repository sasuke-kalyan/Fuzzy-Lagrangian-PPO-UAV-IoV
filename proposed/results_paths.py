"""
Central paths for RL training outputs and evaluation graphs (results/).
"""

from __future__ import annotations

from pathlib import Path

from scenarios import list_scenario_ids

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
EPISODE_HORIZON_GRAPHS_DIR = RESULTS_DIR / "episode_horizon"
REWARD_GRAPHS_DIR = RESULTS_DIR / "reward_graphs"
DELAY_GRAPHS_DIR = RESULTS_DIR / "delay_graphs"
THROUGHPUT_GRAPHS_DIR = RESULTS_DIR / "throughput_graphs"
PDR_GRAPHS_DIR = RESULTS_DIR / "pdr_graphs"
ENERGY_GRAPHS_DIR = RESULTS_DIR / "energy_graphs"
UTILIZATION_GRAPHS_DIR = RESULTS_DIR / "utilization_graphs"
LIFETIME_GRAPHS_DIR = RESULTS_DIR / "lifetime_graphs"
TRAINING_TABLES_DIR = RESULTS_DIR / "training_tables"
TRAINING_LOGS_DIR = RESULTS_DIR / "training_logs"
MODELS_DIR = PROJECT_DIR / "models"

# Short filenames for publication-style outputs (maps project scenario IDs)
SCENARIO_FILE_ALIASES: dict[str, str] = {
    "urban_canyon": "urban",
    "suburban_crossroads": "suburban",
    "emergency_response": "emergency",
}


def scenario_alias(scenario_id: str) -> str:
    return SCENARIO_FILE_ALIASES.get(scenario_id, scenario_id)


def reward_graph_path(scenario_id: str) -> Path:
    return REWARD_GRAPHS_DIR / f"{scenario_alias(scenario_id)}_episode_reward.png"


def training_xlsx_path(scenario_id: str) -> Path:
    return TRAINING_TABLES_DIR / f"{scenario_alias(scenario_id)}_training_results.xlsx"


def training_table_png_path(scenario_id: str) -> Path:
    return TRAINING_TABLES_DIR / f"{scenario_alias(scenario_id)}_training_table.png"


def training_log_path(scenario_id: str) -> Path:
    return TRAINING_LOGS_DIR / f"{scenario_id}_episode_rewards.json"


def model_path_for_scenario(scenario_id: str) -> Path:
    return MODELS_DIR / f"ppo_{scenario_id}"


def ensure_results_dirs() -> None:
    for path in (
        RESULTS_DIR,
        EPISODE_HORIZON_GRAPHS_DIR,
        REWARD_GRAPHS_DIR,
        DELAY_GRAPHS_DIR,
        THROUGHPUT_GRAPHS_DIR,
        PDR_GRAPHS_DIR,
        ENERGY_GRAPHS_DIR,
        UTILIZATION_GRAPHS_DIR,
        LIFETIME_GRAPHS_DIR,
        TRAINING_TABLES_DIR,
        TRAINING_LOGS_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def all_scenario_ids() -> list[str]:
    return list_scenario_ids()
