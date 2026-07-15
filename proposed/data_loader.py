"""
Load UAV-IoV datasets (combined or per-scenario) and resolve output paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scenarios import list_scenario_ids

PROJECT_DIR = Path(__file__).resolve().parent
COMBINED_DATASET = PROJECT_DIR / "uav_iov_dataset.csv"
SCENARIO_DATASETS_DIR = PROJECT_DIR / "datasets"
GRAPH_OUTPUTS_DIR = PROJECT_DIR / "graph_outputs"
SCENARIO_GRAPHS_DIR = GRAPH_OUTPUTS_DIR / "scenarios"


def ensure_output_dirs() -> None:
    SCENARIO_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for sid in list_scenario_ids():
        (SCENARIO_GRAPHS_DIR / sid).mkdir(parents=True, exist_ok=True)


def scenario_dataset_path(scenario_id: str) -> Path:
    return SCENARIO_DATASETS_DIR / f"{scenario_id}.csv"


def load_dataset(scenario_id: str | None = None) -> pd.DataFrame:
    """Load combined CSV or one scenario slice."""
    if scenario_id is not None:
        path = scenario_dataset_path(scenario_id)
        if path.is_file():
            return pd.read_csv(path)
        df = pd.read_csv(COMBINED_DATASET)
        return df[df["Scenario_ID"] == scenario_id].copy()

    if COMBINED_DATASET.is_file():
        return pd.read_csv(COMBINED_DATASET)

    frames = []
    for sid in list_scenario_ids():
        path = scenario_dataset_path(sid)
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            "No dataset found. Run dataset_generation.py first."
        )
    return pd.concat(frames, ignore_index=True)


def iter_scenario_frames(
    df: pd.DataFrame | None = None,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Yield (scenario_id, scenario_name, dataframe) for each scenario."""
    if df is None:
        df = load_dataset()
    if "Scenario_ID" not in df.columns:
        return [("default", "Default", df)]

    out: list[tuple[str, str, pd.DataFrame]] = []
    for sid in list_scenario_ids():
        part = df[df["Scenario_ID"] == sid]
        if part.empty:
            continue
        name = part["Scenario_Name"].iloc[0] if "Scenario_Name" in part.columns else sid
        out.append((sid, str(name), part.copy()))
    return out


def graph_path(scenario_id: str | None, filename: str) -> Path:
    """Per-scenario graph dir, or project root for legacy combined plots."""
    ensure_output_dirs()
    if scenario_id:
        return SCENARIO_GRAPHS_DIR / scenario_id / filename
    return PROJECT_DIR / filename


def sort_vehicle_index(series: pd.Series) -> pd.Series:
    return series.reindex(sorted(series.index, key=lambda x: int(str(x)[1:])))
