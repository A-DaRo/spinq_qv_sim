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


class DriftingSigmaSampler:
    """
    Sampler where the quasi-static noise magnitude itself varies over time.
    
    Models slow drift in charge noise magnitude on timescale of minutes/hours.
    Each "run" or "experiment" samples a new sigma value.
    """
    def __init__(self, sigma_mean: float, sigma_std: float, seed: Optional[int] = None):
        """
        Args:
            sigma_mean: Mean detuning standard deviation
            sigma_std: Standard deviation of sigma itself (meta-noise)
            seed: Random seed for reproducibility
        """
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.rng = np.random.default_rng(seed)
    
    def sample_sigma(self) -> float:
        """Sample a new sigma value for this experimental run."""
        sigma = self.rng.normal(self.sigma_mean, self.sigma_std)
        return max(0.0, float(sigma))  # Ensure positive
    
    def sample_detuning_with_drift(self) -> tuple[float, float]:
        """
        Sample both the drifted sigma and the detuning for this circuit.
        
        Returns:
            (sigma, detuning): sigma value for this run and detuning sample
        """
        sigma = self.sample_sigma()
        detuning = float(self.rng.normal(0.0, sigma))
        return sigma, detuning


class CoherentErrorDriftSampler:
    """
    Sampler for drifting coherent errors (e.g., gate over-rotations).
    
    Models calibration drift where systematic errors vary slowly over time.
    """
    def __init__(self, error_mean: float, error_std: float, seed: Optional[int] = None):
        """
        Args:
            error_mean: Mean systematic error angle (rad)
            error_std: Standard deviation of drift (rad)
            seed: Random seed
        """
        self.error_mean = error_mean
        self.error_std = error_std
        self.rng = np.random.default_rng(seed)
    
    def sample_error_angle(self) -> float:
        """Sample a coherent error angle for this run/circuit."""
        return float(self.rng.normal(self.error_mean, self.error_std))
    
    def sample_many_angles(self, n: int) -> np.ndarray:
        """Sample multiple error angles (e.g., one per circuit in a campaign)."""
        return self.rng.normal(self.error_mean, self.error_std, size=n)

