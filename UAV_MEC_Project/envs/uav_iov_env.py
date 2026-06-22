from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .communication import CommunicationModel
from .costs import CostCalculator
from .energy import EnergyModel
from .entities import Task
from .mobility import MobilityModel
from .reward import RewardCalculator
from .scenarios import ScenarioBuilder
from .task_model import TaskGenerator


class UAVIoVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        env_cfg = config["env"]

        self.area_size = tuple(env_cfg["area_size"])
        self.num_drones = int(env_cfg["num_drones"])
        self.max_vehicles = int(env_cfg["max_vehicles"])
        self.dt = float(env_cfg["dt"])
        self.max_steps = int(env_cfg["max_steps"])
        self.initial_battery_j = float(env_cfg["initial_battery_j"])
        self.rng = np.random.default_rng(config.get("seed", 1))

        self.mobility = MobilityModel(
            self.dt,
            self.area_size,
            env_cfg["altitude_m"],
        )

        self.energy = EnergyModel(
            env_cfg["idle_power_w"],
            env_cfg["move_coeff"],
            self.dt,
        )

        self.comm = CommunicationModel(env_cfg["comm_range_m"])
        self.reward_calc = RewardCalculator(**config["reward"])

        self.cost_calc = CostCalculator(
            env_cfg["safe_distance_m"],
            env_cfg["critical_battery_j"],
        )

        self.task_gen = TaskGenerator(
            tuple(env_cfg["task_data_size_range_mbits"]),
            tuple(env_cfg["task_compute_range_gcycles"]),
            env_cfg["task_deadline_s"],
            env_cfg["task_generation_probability"],
        )

        self.builder = ScenarioBuilder(config)

        self.drone_state_dim = 6
        self.vehicle_state_dim = 2

        self.state_dim = (
            self.num_drones * self.drone_state_dim
            + self.max_vehicles * self.vehicle_state_dim
        )

        self.obs_dim = (
            self.drone_state_dim
            + (self.num_drones - 1) * 3
            + self.max_vehicles * 2
        )

        self.action_dim = 2

        max_speed = env_cfg["max_drone_speed_mps"]

        self.action_space = spaces.Box(
            -max_speed,
            max_speed,
            shape=(self.num_drones, 2),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "global_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
                "local_obs": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.num_drones, self.obs_dim),
                    dtype=np.float32,
                ),
            }
        )

        self.drones = []
        self.vehicles = []
        self.pending_tasks: List[Task] = []

        self.time_s = 0.0
        self.step_count = 0
        self.next_task_id = 0

        self.metrics: Dict[str, float] = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.drones, self.vehicles = self.builder.build(self.rng)

        self.pending_tasks = []
        self.time_s = 0.0
        self.step_count = 0
        self.next_task_id = 0

        self.metrics = {
            "completed_tasks": 0,
            "failed_tasks": 0,
            "collision_events": 0,
            "low_battery_events": 0,
            "total_energy_j": 0.0,
            "avg_delay_ms": 0.0,
            "avg_pdr_pct": 0.0,
            "avg_signal": 0.0,
            "avg_energy_pct": 0.0,
            "qos_penalty": 0.0,
        }

        return self._obs(), {"metrics": self.metrics.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.num_drones,
            2,
        )

        step_energy = 0.0

        for drone, drone_action in zip(self.drones, action):
            self.mobility.update_drone(drone, drone_action)

            drone.battery_j, used = self.energy.update_battery(
                drone.battery_j,
                drone.velocity,
            )

            drone.total_energy_j += used
            step_energy += used

        for vehicle in self.vehicles:
            self.mobility.update_vehicle(vehicle)

        self._maybe_add_emergency_vehicles()
        self._generate_tasks()

        completed_now, failed_now = self._assign_and_complete_tasks()

        reward, reward_info = self.reward_calc.compute(
            completed_now,
            self.drones,
            self.pending_tasks,
            step_energy,
        )

        qos_reward, qos_info = self._compute_qos_reward()
        reward += qos_reward
        reward_info.update(qos_info)

        costs, cost_info = self.cost_calc.compute(self.drones)

        self.metrics["completed_tasks"] += completed_now
        self.metrics["failed_tasks"] += failed_now
        self.metrics["collision_events"] += int(costs[0])
        self.metrics["low_battery_events"] += int(costs[1])
        self.metrics["total_energy_j"] += step_energy

        self.metrics["avg_delay_ms"] = qos_info["avg_delay_ms"]
        self.metrics["avg_pdr_pct"] = qos_info["avg_pdr_pct"]
        self.metrics["avg_signal"] = qos_info["avg_signal"]
        self.metrics["avg_energy_pct"] = qos_info["avg_energy_pct"]
        self.metrics["qos_penalty"] = qos_info["qos_penalty"]

        self.time_s += self.dt
        self.step_count += 1

        truncated = self.step_count >= self.max_steps
        terminated = False

        info = {
            "reward_breakdown": reward_info,
            "cost_breakdown": cost_info,
            "costs": costs,
            "metrics": self.metrics.copy(),
            "time_s": self.time_s,
        }

        return self._obs(), reward, terminated, truncated, info

    def _generate_tasks(self) -> None:
        multiplier = self.config["scenario"].get(
            "task_probability_multiplier",
            1.0,
        )

        if (
            self.config["scenario"]["name"] == "emergency_response"
            and 120.0 <= self.time_s <= 240.0
        ):
            multiplier = max(
                multiplier,
                self.config["scenario"].get(
                    "emergency_task_multiplier",
                    4.0,
                ),
            )

        for vehicle in self.vehicles:
            tasks = self.task_gen.maybe_generate(
                self.rng,
                vehicle,
                self.time_s,
                self.next_task_id,
                multiplier,
            )

            for task in tasks:
                self.pending_tasks.append(task)
                self.next_task_id += 1

    def _assign_and_complete_tasks(self) -> tuple[int, int]:
        completed = 0
        failed = 0
        still_pending: List[Task] = []

        for task in self.pending_tasks:
            if self.time_s - task.created_at > task.deadline_s:
                task.failed = True
                failed += 1
                continue

            drone_id, _ = self.comm.nearest_drone(task, self.drones)

            if drone_id is None:
                still_pending.append(task)
            else:
                task.assigned_drone_id = drone_id
                task.completed = True
                task.completed_at = self.time_s
                completed += 1

        self.pending_tasks = still_pending

        return completed, failed

    def _maybe_add_emergency_vehicles(self) -> None:
        if self.config["scenario"]["name"] != "emergency_response":
            return

        if abs(self.time_s - 120.0) < self.dt / 2 and len(self.vehicles) < 15:
            self.vehicles.extend(
                self.builder.emergency_vehicles(len(self.vehicles), 10)
            )

    def _compute_qos_reward(self) -> tuple[float, Dict[str, float]]:
        qos_cfg = self.config.get("qos", {})
        scenario_cfg = self.config.get("scenario", {})

        delay_max = float(qos_cfg.get("delay_max_ms", 100.0))
        pdr_min = float(qos_cfg.get("pdr_min_pct", 48.0))
        energy_min = float(qos_cfg.get("energy_min_pct", 8.0))
        signal_min = float(qos_cfg.get("signal_min", 0.05))

        penalty_delay = float(qos_cfg.get("penalty_delay", 0.25))
        penalty_pdr = float(qos_cfg.get("penalty_pdr", 0.15))
        penalty_energy = float(qos_cfg.get("penalty_energy", 0.15))
        penalty_signal = float(qos_cfg.get("penalty_signal", 0.08))

        signal_weight = float(qos_cfg.get("signal_term_weight", 120.0))
        pdr_weight = float(qos_cfg.get("pdr_term_weight", 1.5))
        delay_weight = float(qos_cfg.get("delay_term_weight", 1.2))
        energy_weight = float(qos_cfg.get("energy_term_weight", 0.8))

        delay_multiplier = float(scenario_cfg.get("delay_multiplier", 1.0))
        pdr_offset = float(scenario_cfg.get("pdr_offset", 0.0))

        if not self.drones or not self.vehicles:
            return 0.0, {
                "avg_delay_ms": 0.0,
                "avg_pdr_pct": 0.0,
                "avg_signal": 0.0,
                "avg_energy_pct": 0.0,
                "qos_penalty": 0.0,
            }

        delays = []
        pdrs = []
        signals = []
        energies = []

        comm_range = float(self.config["env"]["comm_range_m"])

        for vehicle in self.vehicles:
            nearest_distance = min(
                float(np.linalg.norm(vehicle.position - drone.position[:2]))
                for drone in self.drones
            )

            normalized_distance = min(nearest_distance / comm_range, 1.0)

            signal = max(0.0, 1.0 - normalized_distance)

            delay_ms = (
                10.0
                + 90.0 * normalized_distance * delay_multiplier
            )

            pdr_pct = max(
                0.0,
                min(
                    100.0,
                    100.0 - 50.0 * normalized_distance + pdr_offset,
                ),
            )

            signals.append(signal)
            delays.append(delay_ms)
            pdrs.append(pdr_pct)

        for drone in self.drones:
            energy_pct = (
                drone.battery_j / self.initial_battery_j
            ) * 100.0

            energies.append(energy_pct)

        avg_delay = float(np.mean(delays))
        avg_pdr = float(np.mean(pdrs))
        avg_signal = float(np.mean(signals))
        avg_energy = float(np.mean(energies))

        qos_penalty = 0.0

        if avg_delay > delay_max:
            qos_penalty += penalty_delay * (avg_delay - delay_max)

        if avg_pdr < pdr_min:
            qos_penalty += penalty_pdr * (pdr_min - avg_pdr)

        if avg_energy < energy_min:
            qos_penalty += penalty_energy * (energy_min - avg_energy)

        if avg_signal < signal_min:
            qos_penalty += penalty_signal * (signal_min - avg_signal)

        qos_reward = (
            signal_weight * avg_signal
            + pdr_weight * avg_pdr
            - delay_weight * avg_delay
            + energy_weight * avg_energy
            - qos_penalty
        )

        return float(qos_reward), {
            "avg_delay_ms": avg_delay,
            "avg_pdr_pct": avg_pdr,
            "avg_signal": avg_signal,
            "avg_energy_pct": avg_energy,
            "qos_penalty": float(qos_penalty),
        }

    def _global_state(self) -> np.ndarray:
        states = [
            d.state_vector(self.area_size, self.initial_battery_j)
            for d in self.drones
        ]

        vehicle_states = [
            v.state_vector(self.area_size)
            for v in self.vehicles[: self.max_vehicles]
        ]

        while len(vehicle_states) < self.max_vehicles:
            vehicle_states.append(np.zeros(2, dtype=np.float32))

        return np.concatenate(states + vehicle_states).astype(np.float32)

    def _local_obs(self) -> np.ndarray:
        obs = []

        for i, drone in enumerate(self.drones):
            parts = [
                drone.state_vector(
                    self.area_size,
                    self.initial_battery_j,
                )
            ]

            for j, other in enumerate(self.drones):
                if i == j:
                    continue

                rel = (other.position - drone.position) / np.array(
                    [
                        self.area_size[0],
                        self.area_size[1],
                        max(drone.position[2], 1.0),
                    ]
                )

                parts.append(rel.astype(np.float32))

            for vehicle in self.vehicles[: self.max_vehicles]:
                rel2 = (
                    vehicle.position - drone.position[:2]
                ) / np.array(self.area_size)

                parts.append(rel2.astype(np.float32))

            while len(parts) < 1 + (self.num_drones - 1) + self.max_vehicles:
                parts.append(np.zeros(2, dtype=np.float32))

            obs.append(np.concatenate(parts).astype(np.float32))

        return np.stack(obs, axis=0)

    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            "global_state": self._global_state(),
            "local_obs": self._local_obs(),
        }