"""
Kraus channel implementations for common noise processes.

Provides amplitude damping, phase damping (dephasing), and depolarizing
Kraus operators with numerically stable formulas.
"""

from typing import List
import numpy as np


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Amplitude damping channel Kraus operators.

    K0 = [[1, 0], [0, sqrt(1 - gamma)]]
    K1 = [[0, sqrt(gamma)], [0, 0]]

    Args:
        gamma: Probability of decay from |1> -> |0> in the time window
    Returns:
        List of 2 Kraus operators (2x2 complex numpy arrays)
    """
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")

    sqrt1mg = np.sqrt(max(0.0, 1.0 - gamma))
    sqrtg = np.sqrt(max(0.0, gamma))

    K0 = np.array([[1.0, 0.0], [0.0, sqrt1mg]], dtype=np.complex128)
    K1 = np.array([[0.0, sqrtg], [0.0, 0.0]], dtype=np.complex128)

    return [K0, K1]


def phase_damping_kraus(p_phi: float) -> List[np.ndarray]:
    """
    Phase damping (pure dephasing) Kraus operators for single qubit.

    Model as probabilistic application of Z with probability p_phi/2, or
    equivalent Kraus form that maps off-diagonal elements to (1-p_phi)*offdiag.

    We use 2 Kraus operators:
      K0 = sqrt(1 - p_phi) * I
      K1 = sqrt(p_phi) * Z
    However, this simple form is not trace-preserving for all p_phi.

    Instead implement canonical Kraus:
      K0 = [[1, 0], [0, sqrt(1 - p_phi)]]
      K1 = [[0, 0], [0, sqrt(p_phi)]]

    This gives decay of coherence by sqrt(1 - p_phi).

    Args:
        p_phi: dephasing probability in time interval
    Returns:
        List of 2 Kraus operators
    """
    if p_phi < 0 or p_phi > 1:
        raise ValueError("p_phi must be in [0, 1]")

    sqrt1mp = np.sqrt(max(0.0, 1.0 - p_phi))
    sqrtp = np.sqrt(max(0.0, p_phi))

    K0 = np.array([[1.0, 0.0], [0.0, sqrt1mp]], dtype=np.complex128)
    K1 = np.array([[0.0, 0.0], [0.0, sqrtp]], dtype=np.complex128)

    return [K0, K1]


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Single-qubit depolarizing channel as Kraus operators.

    For single qubit, depolarizing channel:
      E(ρ) = (1 - p) ρ + p * I/2
    This can be represented by 4 Kraus operators:
      K0 = sqrt(1 - 3p/4) * I
      K1 = sqrt(p/4) * X
      K2 = sqrt(p/4) * Y
      K3 = sqrt(p/4) * Z

    Args:
        p: depolarizing probability in [0, 1]
    Returns:
        List of 4 Kraus operators (2x2 matrices)
    """
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")

    # Avoid negative under sqrt due to floating point
    k0_coeff = np.sqrt(max(0.0, 1.0 - 3.0 * p / 4.0))
    k_other = np.sqrt(max(0.0, p / 4.0))

    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    K0 = k0_coeff * I
    K1 = k_other * X
    K2 = k_other * Y
    K3 = k_other * Z

    return [K0, K1, K2, K3]


def is_trace_preserving(kraus_ops: List[np.ndarray], atol: float = 1e-12) -> bool:
    """
    Check that sum_k K_k^† K_k == I within tolerance.
    """
    if len(kraus_ops) == 0:
        return False

    dim = kraus_ops[0].shape[0]
    accum = np.zeros((dim, dim), dtype=np.complex128)
    for K in kraus_ops:
        accum += K.conj().T @ K

    I = np.eye(dim, dtype=np.complex128)
    return np.allclose(accum, I, atol=atol)
