"""
Coherent error primitives: small rotations and residual ZZ couplings.

Provides helpers to build single-qubit small-angle rotation unitaries and
two-qubit ZZ-phase unitaries.
"""

import numpy as np
from typing import Tuple


def small_rotation(axis: str, angle: float) -> np.ndarray:
    """
    Return a single-qubit rotation unitary about given axis by angle (radians).

    axis: 'x', 'y', or 'z'
    angle: rotation angle in radians
    """
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)

    if axis.lower() == 'x':
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    elif axis.lower() == 'y':
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    elif axis.lower() == 'z':
        return np.array([[np.exp(-1j * angle / 2), 0.0], [0.0, np.exp(1j * angle / 2)]], dtype=np.complex128)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def zz_phase_unitary(phi: float) -> np.ndarray:
    """
    Two-qubit ZZ-phase unitary: exp(-i * phi * Z ⊗ Z / 2)

    Returns 4x4 unitary matrix.
    """
    # Diagonal in computational basis with phases
    phases = [np.exp(-1j * phi / 2), np.exp(1j * phi / 2), np.exp(1j * phi / 2), np.exp(-1j * phi / 2)]
    return np.diag(phases).astype(np.complex128)
