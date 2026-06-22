from __future__ import annotations

import numpy as np


class GaussianNoise:
    def __init__(self, sigma: float, clip: float | None = None):
        self.sigma = float(sigma)
        self.clip = clip

    def sample(self, shape):
        noise = np.random.normal(0.0, self.sigma, size=shape).astype(np.float32)
        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)
        return noise
