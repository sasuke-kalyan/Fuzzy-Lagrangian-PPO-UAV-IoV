from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rows: List[Dict], output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r["episode"] for r in rows]
    ys = [r["reward"] for r in rows]

    # Moving Average Smoothing
    window = 1000

    smooth_y = []
    for i in range(len(ys)):
        start = max(0, i - window + 1)
        smooth_y.append(np.mean(ys[start:i + 1]))

    plt.figure(figsize=(10, 6))

    # Smoothed reward curve
    plt.plot(
        xs,
        smooth_y,
        linewidth=3,
        label="Moving Average (1000)"
    )

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title(title, fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()