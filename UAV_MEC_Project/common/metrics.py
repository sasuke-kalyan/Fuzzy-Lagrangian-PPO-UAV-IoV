from __future__ import annotations

from typing import Dict, List


class EpisodeLogger:
    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, row: Dict) -> None:
        self.rows.append(row)

    def mean(self, key: str) -> float:
        vals = [float(r[key]) for r in self.rows if key in r]
        return sum(vals) / max(len(vals), 1)
