"""
Unit tests verifying Kraus operators preserve trace.
"""

import numpy as np
from spinq_qv.noise import channels


def test_amplitude_damping_trace_preserving():
    for g in [0.0, 1e-6, 0.01, 0.2, 0.8]:
        K = channels.amplitude_damping_kraus(g)
        assert channels.is_trace_preserving(K)


def test_phase_damping_trace_preserving():
    for p in [0.0, 1e-6, 0.01, 0.5, 1.0]:
        K = channels.phase_damping_kraus(p)
        assert channels.is_trace_preserving(K)


def test_depolarizing_trace_preserving():
    for p in [0.0, 1e-6, 0.01, 0.5, 1.0]:
        K = channels.depolarizing_kraus(p)
        assert channels.is_trace_preserving(K)
