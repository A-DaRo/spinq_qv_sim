"""
Quasi-static stochastic noise helpers.

Maps T2* to detuning sigma and provides samplers for quasi-static detunings
(e.g., sample once per circuit from Normal(0, sigma^2)).
"""

import numpy as np
from typing import Optional


def t2_star_to_sigma(t2_star: float) -> float:
    """
    Convert T2* to detuning standard deviation sigma using formula:
      sigma = sqrt(2) / T2*
    as documented in project Technicalities.

    Args:
        t2_star: T2* in same time units as sigma's inverse (e.g., seconds)
    Returns:
        sigma: standard deviation of detuning (rad/s)
    """
    if t2_star <= 0:
        raise ValueError("t2_star must be positive")

    return np.sqrt(2.0) / t2_star


class QuasiStaticSampler:
    """
    Sampler that produces one detuning per circuit from N(0, sigma^2).
    Uses numpy Generator for reproducibility.
    """
    def __init__(self, sigma: float, seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def sample_detuning(self) -> float:
        return float(self.rng.normal(0.0, self.sigma))

    def sample_many(self, n: int):
        return self.rng.normal(0.0, self.sigma, size=n)
