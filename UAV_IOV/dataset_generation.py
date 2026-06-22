"""
Generate UAV-IoV datasets for all three paper simulation scenarios (Section VI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from communication_model import compute_link_metrics
from data_loader import COMBINED_DATASET, SCENARIO_DATASETS_DIR, ensure_output_dirs
from scenarios import NUM_SAMPLES, NUM_UAVS, SCENARIOS, ScenarioConfig


def _init_uav_positions(area: int, rng: np.random.Generator) -> dict[str, list[float]]:
    return {
        f"U{i}": [
            float(rng.integers(0, area)),
            float(rng.integers(0, area)),
            float(rng.integers(80, 201)),
        ]
        for i in range(1, NUM_UAVS + 1)
    }


def _init_vehicle_positions(
    cfg: ScenarioConfig, area: int, rng: np.random.Generator
) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    margin = 80

    if cfg.layout == "corridor":
        roads = [300, 700, 1100, 1500]
        for i in range(1, cfg.num_vehicles + 1):
            y = roads[(i - 1) % len(roads)]
            x = float(rng.integers(margin, area - margin))
            positions[f"V{i}"] = [x, float(y)]

    elif cfg.layout == "sparse_grid":
        xs = np.linspace(margin, area - margin, 5)
        ys = np.linspace(margin, area - margin, 5)
        grid = [(float(x), float(y)) for x in xs for y in ys]
        rng.shuffle(grid)
        for i in range(1, cfg.num_vehicles + 1):
            positions[f"V{i}"] = list(grid[i - 1])

    else:
        for i in range(1, cfg.num_vehicles + 1):
            positions[f"V{i}"] = [
                float(rng.integers(margin, area - margin)),
                float(rng.integers(margin, area - margin)),
            ]

    return positions


def _active_vehicles(cfg: ScenarioConfig, timestep: int) -> list[str]:
    if cfg.layout != "emergency_dynamic":
        return [f"V{i}" for i in range(1, cfg.num_vehicles + 1)]

    start = cfg.emergency_start_step or 0
    base = [f"V{i}" for i in range(1, cfg.num_vehicles + 1)]
    if timestep < start:
        return base

    extra = cfg.emergency_extra_vehicles
    return base + [f"V{cfg.num_vehicles + j}" for j in range(1, extra + 1)]


def _move_vehicles(
    cfg: ScenarioConfig,
    positions: dict[str, list[float]],
    active: list[str],
    timestep: int,
    area: int,
    rng: np.random.Generator,
) -> None:
    incident_x, incident_y = area / 2, area / 2
    start = cfg.emergency_start_step or 0

    for vid in active:
        if vid not in positions:
            if cfg.layout == "emergency_dynamic" and timestep >= start:
                positions[vid] = [
                    float(incident_x + rng.integers(-120, 121)),
                    float(incident_y + rng.integers(-120, 121)),
                ]
            else:
                continue

        x, y = positions[vid]
        dx = int(rng.integers(cfg.vehicle_step_x[0], cfg.vehicle_step_x[1]))
        dy = int(rng.integers(cfg.vehicle_step_y[0], cfg.vehicle_step_y[1]))

        if cfg.layout == "emergency_dynamic" and vid.startswith("V") and int(vid[1:]) > cfg.num_vehicles:
            dx = int(rng.integers(-30, 31))
            dy = int(rng.integers(-30, 31))
            x = float(np.clip(x + dx, incident_x - 200, incident_x + 200))
            y = float(np.clip(y + dy, incident_y - 200, incident_y + 200))
        else:
            x = float(np.clip(x + dx, 0, area))
            y = float(np.clip(y + dy, 0, area))

        positions[vid] = [x, y]


def _scenario_metrics_adjust(
    cfg: ScenarioConfig,
    metrics: dict[str, float],
    timestep: int,
    vehicle_id: str,
) -> dict[str, float]:
    delay = min(100.0, metrics["Delay"] * cfg.delay_multiplier)
    pdr = max(50.0, metrics["PDR"] + cfg.pdr_offset)

    if cfg.layout == "emergency_dynamic":
        start = cfg.emergency_start_step or 0
        if timestep >= start:
            vid_num = int(vehicle_id[1:])
            if vid_num > cfg.num_vehicles:
                delay = min(100.0, delay * 1.35 + 8.0)
                pdr = max(50.0, pdr - 4.0)
            elif cfg.task_rate == "low":
                delay = max(14.0, delay * 0.92)

    return {
        "Distance": metrics["Distance"],
        "Signal_Strength": metrics["Signal_Strength"],
        "Delay": round(delay, 2),
        "PDR": round(pdr, 2),
    }


def generate_scenario_dataframe(
    cfg: ScenarioConfig,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(cfg.scenario_id) % 1000)
    area = cfg.area_size
    vehicle_positions = _init_vehicle_positions(cfg, area, rng)
    uav_positions = _init_uav_positions(area, rng)
    uav_energy = {f"U{i}": int(rng.integers(35, 51)) for i in range(1, NUM_UAVS + 1)}

    rows: list[list] = []

    for t in range(NUM_SAMPLES):
        active = _active_vehicles(cfg, t)
        _move_vehicles(cfg, vehicle_positions, active, t, area, rng)

        for uav_id, (uav_x, uav_y, uav_z) in list(uav_positions.items()):
            uav_x += float(rng.integers(cfg.uav_step_xy[0], cfg.uav_step_xy[1]))
            uav_y += float(rng.integers(cfg.uav_step_xy[0], cfg.uav_step_xy[1]))
            uav_x = float(np.clip(uav_x, 0, area))
            uav_y = float(np.clip(uav_y, 0, area))
            uav_positions[uav_id] = [uav_x, uav_y, uav_z]
            drain = int(rng.integers(0, 2)) + cfg.energy_drain_extra
            uav_energy[uav_id] = max(10, uav_energy[uav_id] - drain)

        emergency_phase = (
            cfg.layout == "emergency_dynamic"
            and cfg.emergency_start_step is not None
            and t >= cfg.emergency_start_step
        )
        task_rate = cfg.task_rate
        if emergency_phase:
            task_rate = "burst"

        for vehicle_id in active:
            if vehicle_id not in vehicle_positions:
                continue
            vehicle_x, vehicle_y = vehicle_positions[vehicle_id]
            speed = int(rng.integers(cfg.speed_range[0], cfg.speed_range[1] + 1))
            is_emergency_vehicle = (
                cfg.layout == "emergency_dynamic"
                and int(vehicle_id[1:]) > cfg.num_vehicles
            )

            for uav_id in uav_positions:
                uav_x, uav_y, uav_z = uav_positions[uav_id]
                raw = compute_link_metrics(
                    vehicle_x, vehicle_y, uav_x, uav_y, uav_z, rng=rng
                )
                metrics = _scenario_metrics_adjust(cfg, raw, t, vehicle_id)

                rows.append([
                    t,
                    cfg.scenario_id,
                    cfg.name,
                    vehicle_id,
                    round(vehicle_x, 2),
                    round(vehicle_y, 2),
                    speed,
                    uav_id,
                    round(uav_x, 2),
                    round(uav_y, 2),
                    uav_z,
                    metrics["Distance"],
                    metrics["Signal_Strength"],
                    metrics["Delay"],
                    metrics["PDR"],
                    uav_energy[uav_id],
                    task_rate,
                    int(emergency_phase),
                    int(is_emergency_vehicle),
                    cfg.focus,
                ])

    return pd.DataFrame(
        rows,
        columns=[
            "Timestamp",
            "Scenario_ID",
            "Scenario_Name",
            "Vehicle_ID",
            "Vehicle_X",
            "Vehicle_Y",
            "Speed",
            "UAV_ID",
            "UAV_X",
            "UAV_Y",
            "UAV_Z",
            "Distance",
            "Signal_Strength",
            "Delay",
            "PDR",
            "Energy",
            "Task_Rate",
            "Emergency_Phase",
            "Is_Emergency_Vehicle",
            "Scenario_Focus",
        ],
    )


def main() -> None:
    ensure_output_dirs()
    frames = []

    print("\n=== UAV-IoV Multi-Scenario Dataset Generation (Paper Section VI) ===\n")

    for sid, cfg in SCENARIOS.items():
        df = generate_scenario_dataframe(cfg)
        out_path = SCENARIO_DATASETS_DIR / f"{sid}.csv"
        df.to_csv(out_path, index=False)
        frames.append(df)
        print(f"[{cfg.name}]")
        print(f"  File      : {out_path.name}")
        print(f"  Vehicles  : {df['Vehicle_ID'].nunique()} (peak)")
        print(f"  Rows      : {len(df)}")
        print(f"  Avg Delay : {df['Delay'].mean():.2f} ms")
        print(f"  Avg PDR   : {df['PDR'].mean():.2f} %")
        print(f"  Focus     : {cfg.focus}\n")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_DATASET, index=False)
    print(f"Combined dataset : {COMBINED_DATASET.name} ({len(combined)} rows)")
    print("Dataset generation completed.\n")


if __name__ == "__main__":
    main()
