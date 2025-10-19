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


def zz_crosstalk_unitary(zeta: float, duration: float) -> np.ndarray:
    """
    ZZ crosstalk unitary between neighboring qubits.
    
    Models always-on parasitic interaction: U_ZZ = exp(-i ζ Z⊗Z t)
    
    Args:
        zeta: Crosstalk coupling strength (rad/s)
        duration: Time duration of interaction (s)
    
    Returns:
        4x4 unitary matrix for two qubits
    """
    phi = zeta * duration
    # exp(-i phi Z⊗Z) is diagonal with phases [e^(-iφ), e^(iφ), e^(iφ), e^(-iφ)]
    phases = [np.exp(-1j * phi), np.exp(1j * phi), np.exp(1j * phi), np.exp(-1j * phi)]
    return np.diag(phases).astype(np.complex128)


def control_crosstalk_unitary(target_axis: str, target_angle: float, crosstalk_fraction: float) -> np.ndarray:
    """
    Two-qubit unitary for control pulse crosstalk.
    
    When a control pulse targets qubit i with rotation R(axis, θ), 
    a fraction α of that field leaks to neighboring qubit j.
    
    Result: U = R_i(axis, θ) ⊗ R_j(axis, α*θ)
    
    Args:
        target_axis: Rotation axis ('x', 'y', or 'z')
        target_angle: Target rotation angle (rad)
        crosstalk_fraction: Leakage fraction α (typically 0.01 - 0.1)
    
    Returns:
        4x4 unitary matrix (tensor product of two single-qubit rotations)
    """
    R_target = small_rotation(target_axis, target_angle)
    R_spectator = small_rotation(target_axis, crosstalk_fraction * target_angle)
    
    return np.kron(R_target, R_spectator)


def compute_unitary_infidelity(U_actual: np.ndarray, U_ideal: np.ndarray) -> float:
    """
    Compute infidelity of actual unitary compared to ideal unitary.
    
    Infidelity = 1 - |Tr(U_ideal^† U_actual)|^2 / d^2
    
    Args:
        U_actual: Actual (noisy) unitary
        U_ideal: Ideal (target) unitary
    
    Returns:
        Infidelity in [0, 1]
    """
    dim = U_actual.shape[0]
    trace_overlap = np.trace(U_ideal.conj().T @ U_actual)
    fidelity = abs(trace_overlap)**2 / dim**2
    
    return float(1.0 - fidelity)

