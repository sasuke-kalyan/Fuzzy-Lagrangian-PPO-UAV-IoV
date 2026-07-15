"""
Simulation scenarios from the base paper (LC-MAPO / drone-aided IoV, Section VI).

1. urban_canyon       — high-density, 20 vehicles on parallel roads (collision stress)
2. suburban_crossroads — sparse 15-vehicle layout, long UAV flights (energy stress)
3. emergency_response  — stable low load, then sudden incident influx (adaptability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Shared fair-comparison parameters.
PAPER_AREA_SIZE = 2000
NUM_UAVS = 8
# Steps per episode / simulation horizon.
NUM_SAMPLES = 50

LayoutType = Literal["corridor", "sparse_grid", "emergency_dynamic"]


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    description: str
    num_vehicles: int
    vehicle_speed_mps: float
    layout: LayoutType
    speed_range: tuple[int, int]
    vehicle_step_x: tuple[int, int]
    vehicle_step_y: tuple[int, int]
    uav_step_xy: tuple[int, int]
    energy_drain_extra: int
    delay_multiplier: float
    pdr_offset: float
    task_rate: str
    emergency_start_step: int | None = None
    emergency_extra_vehicles: int = 0
    focus: str = ""

    @property
    def area_size(self) -> int:
        return PAPER_AREA_SIZE


SCENARIOS: dict[str, ScenarioConfig] = {
    "urban_canyon": ScenarioConfig(
        scenario_id="urban_canyon",
        name="Urban Canyon",
        description=(
            "Dense IoV (NV=20): vehicles on parallel main roads, high speed and "
            "task rate — tests throughput and collision-avoidance pressure (C1)."
        ),
        num_vehicles=20,
        vehicle_speed_mps=22.0,
        layout="corridor",
        speed_range=(60, 100),
        vehicle_step_x=(-25, 26),
        vehicle_step_y=(-8, 9),
        uav_step_xy=(-12, 13),
        energy_drain_extra=1,
        delay_multiplier=1.25,
        pdr_offset=-3.0,
        task_rate="high",
        focus="collision_avoidance",
    ),
    "suburban_crossroads": ScenarioConfig(
        scenario_id="suburban_crossroads",
        name="Suburban Crossroads",
        description=(
            "Sprawling suburban IoV (NV=15): sparse road grid, wide dispersion — "
            "tests UAV endurance and path efficiency under energy cost C2."
        ),
        num_vehicles=15,
        vehicle_speed_mps=10.0,
        layout="sparse_grid",
        speed_range=(20, 50),
        vehicle_step_x=(-18, 19),
        vehicle_step_y=(-18, 19),
        uav_step_xy=(-18, 19),
        energy_drain_extra=2,
        delay_multiplier=1.45,
        pdr_offset=-5.0,
        task_rate="medium",
        focus="energy_efficiency",
    ),
    "emergency_response": ScenarioConfig(
        scenario_id="emergency_response",
        name="Emergency Response",
        description=(
            "Dynamic incident response with 15 emergency-zone vehicles and burst "
            "task load — tests non-stationary adaptation."
        ),
        num_vehicles=15,
        vehicle_speed_mps=14.0,
        layout="emergency_dynamic",
        speed_range=(30, 70),
        vehicle_step_x=(-20, 21),
        vehicle_step_y=(-20, 21),
        uav_step_xy=(-14, 15),
        energy_drain_extra=1,
        delay_multiplier=1.0,
        pdr_offset=0.0,
        task_rate="low",
        emergency_start_step=max(1, int(NUM_SAMPLES * (120 / 360))),
        emergency_extra_vehicles=0,
        focus="dynamic_adaptation",
    ),
}


def list_scenario_ids() -> list[str]:
    return list(SCENARIOS.keys())


def get_scenario(scenario_id: str) -> ScenarioConfig:
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Choose from: {', '.join(list_scenario_ids())}"
        )
    return SCENARIOS[scenario_id]
