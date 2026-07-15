"""Smoothing helpers for publication-style metric and reward curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_curve(
    values,
    *,
    window_fraction: float = 0.18,
    min_window: int = 21,
    max_window: int | None = 201,
    passes: int = 3,
) -> np.ndarray:
    """
    Return a strongly smoothed curve while preserving the original length.

    The repeated centered rolling mean removes jagged reward spikes and produces
    paper-style curves without changing the underlying logged data.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n <= 2:
        return arr

    window = max(min_window, int(round(n * window_fraction)))
    if max_window is not None:
        window = min(window, max_window)
    window = min(window, n)
    if window % 2 == 0 and window > 1:
        window -= 1
    window = max(3, window)

    curve = pd.Series(arr)
    for _ in range(max(1, passes)):
        curve = curve.rolling(window=window, min_periods=1, center=True).mean()
        curve = curve.bfill().ffill()
    return curve.to_numpy(dtype=np.float64)
