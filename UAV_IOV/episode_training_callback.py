"""
Record per-episode rewards during SB3 training (Monitor-compatible).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EpisodeRecord:
    episode: int
    total_reward: float
    average_reward: float
    scenario_id: str
    scenario_name: str
    training_time_sec: float
    n_steps: int


@dataclass
class EpisodeTrainingLog:
    scenario_id: str
    scenario_name: str
    records: list[EpisodeRecord] = field(default_factory=list)

    def to_dataframe_rows(self) -> list[dict]:
        return [
            {
                "Episode": r.episode,
                "Total Reward": round(r.total_reward, 4),
                "Average Reward": round(r.average_reward, 4),
                "Scenario": r.scenario_name,
                "Training Time": round(r.training_time_sec, 4),
            }
            for r in self.records
        ]

    @property
    def episode_returns(self):
        import numpy as np

        return np.array([r.total_reward for r in self.records], dtype=np.float64)


class EpisodeRewardCallback(BaseCallback):
    """Collect episode return when Monitor reports episode end in infos."""

    def __init__(
        self,
        scenario_id: str,
        scenario_name: str,
        target_episodes: int,
        steps_per_episode: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.target_episodes = target_episodes
        self.steps_per_episode = steps_per_episode
        self.log = EpisodeTrainingLog(scenario_id=scenario_id, scenario_name=scenario_name)
        self._episode_start_time = time.perf_counter()
        self._training_start = time.perf_counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if ep_info is None:
                continue
            total = float(ep_info["r"])
            length = max(int(ep_info["l"]), 1)
            elapsed = time.perf_counter() - self._episode_start_time
            self.log.records.append(
                EpisodeRecord(
                    episode=len(self.log.records) + 1,
                    total_reward=total,
                    average_reward=total / length,
                    scenario_id=self.scenario_id,
                    scenario_name=self.scenario_name,
                    training_time_sec=elapsed,
                    n_steps=length,
                )
            )
            self._episode_start_time = time.perf_counter()
            if self.verbose and len(self.log.records) % 100 == 0:
                print(
                    f"  [{self.scenario_id}] episode {len(self.log.records)}"
                    f"/{self.target_episodes} — return {total:.2f}"
                )
        return len(self.log.records) < self.target_episodes

    @property
    def episode_returns(self) -> np.ndarray:
        return np.array([r.total_reward for r in self.log.records], dtype=np.float64)
