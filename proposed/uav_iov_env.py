"""
Gymnasium environment: vehicle chooses among fixed UAVs each timestep.
Observation includes both UAV link previews; reward uses communication_model.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from communication_model import (
    AREA_SIZE,
    COMMUNICATION_RANGE,
    ENERGY_DRAIN_MAX,
    ENERGY_DRAIN_MIN,
    baseline_compatible_reward,
    compute_link_metrics,
    energy_to_percent,
    fuzzy_priority,
)
import constraints as cons
import network_config as net
from scenarios import get_scenario
from training_config import STEPS_PER_EPISODE


class UAVIoVEnv(gym.Env):
    """UAV selection for a moving ground vehicle."""

    metadata = {"render_modes": []}

    NUM_UAVS = net.NUM_UAVS
    # One RL episode = full scenario timestamp horizon (dataset steps)
    MAX_EPISODE_STEPS = STEPS_PER_EPISODE

    def __init__(self, render_mode=None, scenario_id: str = "urban_canyon"):
        super().__init__()
        self.render_mode = render_mode
        self.scenario_id = scenario_id
        self._scenario = get_scenario(scenario_id)
        self._area_size = float(self._scenario.area_size)

        self.action_space = spaces.Discrete(self.NUM_UAVS)
        # [vx, vy] normalized + per-UAV (rel_x, rel_y, signal, energy_norm)
        obs_dim = 2 + self.NUM_UAVS * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._vehicle_xy = np.zeros(2, dtype=np.float64)
        self._uav_positions = np.zeros((self.NUM_UAVS, 3), dtype=np.float64)
        self._uav_energies = np.zeros(self.NUM_UAVS, dtype=np.float64)
        self._lagrangian = {
            "delay": 0.35,
            "pdr": 0.20,
            "energy": 0.60,
            "signal": 0.35,
        }
        self._lambda_lr = 0.03

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "scenario_id" in options:
            self.scenario_id = options["scenario_id"]
            self._scenario = get_scenario(self.scenario_id)
            self._area_size = float(self._scenario.area_size)

        self._step_count = 0
        self._lagrangian = {
            "delay": 0.35,
            "pdr": 0.20,
            "energy": 0.60,
            "signal": 0.35,
        }
        area = self._area_size
        self._vehicle_xy = self._rng.uniform(0, area, size=2)

        for i in range(self.NUM_UAVS):
            self._uav_positions[i] = [
                self._rng.uniform(0, area),
                self._rng.uniform(0, area),
                self._rng.integers(80, 201),
            ]
            self._uav_energies[i] = self._rng.integers(30, 51)

        return self._get_obs(), {"scenario_id": self.scenario_id}

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.NUM_UAVS:
            raise ValueError(f"Invalid action {action}")

        requested_action = action
        ranked_candidates = self._ranked_action_candidates()
        effective = ranked_candidates[action % len(ranked_candidates)]
        action = int(effective["action"])

        ux, uy, uz = self._uav_positions[action]
        metrics = compute_link_metrics(
            self._vehicle_xy[0],
            self._vehicle_xy[1],
            ux,
            uy,
            uz,
            rng=self._rng,
        )
        energy = float(self._uav_energies[action])
        drain = self._rng.integers(ENERGY_DRAIN_MIN, ENERGY_DRAIN_MAX + 1)
        reward = baseline_compatible_reward(
            metrics["Signal_Strength"],
            metrics["PDR"],
            metrics["Delay"],
            energy,
            metrics["Distance"],
            float(drain),
            completed_count=1,
            scenario_focus=self._scenario.focus,
            lagrangian_multipliers=self._lagrangian,
        )
        self._update_lagrangian(metrics, energy)

        self._uav_energies[action] = max(0.0, energy - drain)

        self._move_vehicle()

        self._step_count += 1
        terminated = self._step_count >= self.MAX_EPISODE_STEPS
        truncated = False

        info = {
            "uav_id": f"U{action + 1}",
            "requested_action": requested_action,
            "effective_action": action,
            "action_mask": self.action_mask().tolist(),
            "metrics": metrics,
            "energy": float(self._uav_energies[action]),
            "lagrangian": self._lagrangian.copy(),
            "raw_reward": reward,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """True means the UAV passes hard QoS checks for direct selection."""
        candidates = self._candidate_rows()
        mask = np.array([row["valid"] for row in candidates], dtype=bool)
        if mask.any():
            return mask
        return np.ones(self.NUM_UAVS, dtype=bool)

    def _candidate_rows(self) -> list[dict[str, float | int | bool]]:
        rows: list[dict[str, float | int | bool]] = []
        for i in range(self.NUM_UAVS):
            ux, uy, uz = self._uav_positions[i]
            metrics = compute_link_metrics(
                self._vehicle_xy[0],
                self._vehicle_xy[1],
                ux,
                uy,
                uz,
                rng=self._rng,
            )
            energy = float(self._uav_energies[i])
            energy_pct = energy_to_percent(energy)
            valid = (
                metrics["Distance"] <= COMMUNICATION_RANGE
                and energy_pct >= 20.0
                and metrics["Signal_Strength"] >= 0.10
                and metrics["Delay"] <= 95.0
            )
            priority = fuzzy_priority(
                metrics["Signal_Strength"],
                metrics["PDR"],
                metrics["Delay"],
                energy,
                self._scenario.focus,
            )
            score = (
                240.0 * metrics["Signal_Strength"]
                + 2.0 * metrics["PDR"]
                - 1.3 * metrics["Delay"]
                + 2.5 * energy_pct
                + 35.0 * priority
                - 0.02 * metrics["Distance"]
            )
            rows.append(
                {
                    "action": i,
                    "valid": valid,
                    "score": float(score),
                    "energy_pct": energy_pct,
                    **metrics,
                }
            )
        return rows

    def _ranked_action_candidates(self) -> list[dict[str, float | int | bool]]:
        rows = self._candidate_rows()
        valid_rows = [row for row in rows if row["valid"]]
        if not valid_rows:
            valid_rows = rows
        return sorted(
            valid_rows,
            key=lambda row: (
                float(row["score"]),
                float(row["energy_pct"]),
                float(row["Signal_Strength"]),
                -float(row["Delay"]),
            ),
            reverse=True,
        )

    def _update_lagrangian(self, metrics: dict[str, float], energy: float) -> None:
        violations = {
            "delay": cons.violation_delay(metrics["Delay"]),
            "pdr": cons.violation_pdr(metrics["PDR"]),
            "energy": cons.violation_energy(energy_to_percent(energy)),
            "signal": cons.violation_signal(metrics["Signal_Strength"]),
        }
        for key, violation in violations.items():
            self._lagrangian[key] = min(
                10.0,
                max(0.0, self._lagrangian[key] + self._lambda_lr * violation),
            )

    def _move_vehicle(self):
        sx, sy = self._scenario.vehicle_step_x, self._scenario.vehicle_step_y
        self._vehicle_xy[0] += self._rng.integers(sx[0], sx[1])
        self._vehicle_xy[1] += self._rng.integers(sy[0], sy[1])
        self._vehicle_xy = np.clip(self._vehicle_xy, 0.0, self._area_size)

    def _get_obs(self) -> np.ndarray:
        area = self._area_size
        features: list[float] = [
            self._vehicle_xy[0] / area,
            self._vehicle_xy[1] / area,
        ]

        for i in range(self.NUM_UAVS):
            ux, uy, uz = self._uav_positions[i]
            m = compute_link_metrics(
                self._vehicle_xy[0],
                self._vehicle_xy[1],
                ux,
                uy,
                uz,
                rng=self._rng,
            )
            rel_x = (ux - self._vehicle_xy[0]) / area
            rel_y = (uy - self._vehicle_xy[1]) / area
            features.extend(
                [
                    np.clip(rel_x, -1.0, 1.0) * 0.5 + 0.5,
                    np.clip(rel_y, -1.0, 1.0) * 0.5 + 0.5,
                    m["Signal_Strength"],
                    self._uav_energies[i] / 50.0,
                ]
            )

        return np.array(features, dtype=np.float32)
