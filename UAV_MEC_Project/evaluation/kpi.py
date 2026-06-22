from __future__ import annotations

from typing import Dict, List


def aggregate_kpis(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for key in keys:
        if key == "episode":
            continue
        vals = [float(r.get(key, 0.0)) for r in rows]
        out[f"{key}_mean"] = sum(vals) / len(vals)
    completed = out.get("completed_tasks_mean", 0.0)
    energy = out.get("total_energy_j_mean", 0.0)
    out["energy_per_task_j_mean"] = energy / max(completed, 1.0)
    return out
