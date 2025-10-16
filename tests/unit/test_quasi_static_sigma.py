"""
Test mapping from T2* to detuning sigma.
"""

import numpy as np
from spinq_qv.noise.stochastic import t2_star_to_sigma


def test_t2_star_sigma():
    T2s = [20e-6, 50e-6, 100e-6]
    for t2 in T2s:
        sigma = t2_star_to_sigma(t2)
        expected = np.sqrt(2.0) / t2
        assert abs(sigma - expected) < 1e-18
