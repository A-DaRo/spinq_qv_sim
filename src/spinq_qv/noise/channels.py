"""
Kraus channel implementations for common noise processes.

Provides amplitude damping, phase damping (dephasing), and depolarizing
Kraus operators with numerically stable formulas.

Also includes SPAM (State Preparation And Measurement) error models.
"""

from typing import List, Tuple
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


def depolarizing_kraus_2q(p: float) -> List[np.ndarray]:
    """
    Two-qubit depolarizing channel as Kraus operators.
    
    For two qubits, the depolarizing channel is:
      E(ρ) = (1 - p) ρ + p * I/4
    
    This can be represented by 16 Kraus operators (tensor products of Paulis):
      K_ij = sqrt(w_ij) * (P_i ⊗ P_j)
    where P_i ∈ {I, X, Y, Z} and weights are chosen so sum of K^†K = I.
    
    For uniform depolarizing:
      K_00 = sqrt(1 - 15p/16) * (I ⊗ I)
      K_ij = sqrt(p/16) * (P_i ⊗ P_j) for (i,j) ≠ (0,0)
    
    Args:
        p: depolarizing probability in [0, 1]
    Returns:
        List of 16 Kraus operators (4x4 matrices)
    """
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    
    # Pauli matrices
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    
    paulis = [I, X, Y, Z]
    
    # Coefficients
    k0_coeff = np.sqrt(max(0.0, 1.0 - 15.0 * p / 16.0))
    k_other = np.sqrt(max(0.0, p / 16.0))
    
    kraus_ops = []
    for i, P1 in enumerate(paulis):
        for j, P2 in enumerate(paulis):
            # Tensor product P1 ⊗ P2
            tensor_prod = np.kron(P1, P2)
            
            if i == 0 and j == 0:
                # Identity term
                kraus_ops.append(k0_coeff * tensor_prod)
            else:
                # Error terms
                kraus_ops.append(k_other * tensor_prod)
    
    return kraus_ops


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


def compute_channel_fidelity(kraus_ops: List[np.ndarray], dim: int = 2) -> float:
    """
    Compute average gate fidelity of a quantum channel represented by Kraus operators.
    
    Average fidelity is defined as:
        F_avg = (Tr(M^† M) + d) / (d(d+1))
    
    where M is the superoperator representation and d is the Hilbert space dimension.
    
    For computational efficiency, we use:
        F_avg = (1/(d+1)) * [ (1/d) * sum_i Tr(K_i^† K_i) + sum_i |Tr(K_i)|^2 ]
    
    Args:
        kraus_ops: List of Kraus operators for the channel
        dim: Hilbert space dimension (2 for single qubit, 4 for two qubits)
    
    Returns:
        Average gate fidelity in [0, 1]
    """
    if not kraus_ops:
        raise ValueError("kraus_ops cannot be empty")
    
    # Check trace preservation
    trace_sum = sum(np.trace(K.conj().T @ K) for K in kraus_ops)
    if not np.isclose(trace_sum, dim, atol=1e-10):
        raise ValueError(f"Channel is not trace-preserving: Tr sum = {trace_sum}, expected {dim}")
    
    # Compute fidelity using formula from Nielsen & Chuang
    # F_avg = (sum_i |Tr(K_i)|^2 + d) / (d * (d+1))
    trace_squares = sum(abs(np.trace(K))**2 for K in kraus_ops)
    
    F_avg = (trace_squares + dim) / (dim * (dim + 1))
    
    return float(np.real(F_avg))


def compose_kraus_channels(kraus_list1: List[np.ndarray], kraus_list2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose two quantum channels represented by Kraus operators.
    
    For channels E1(ρ) = sum_i K1_i ρ K1_i^† and E2(ρ) = sum_j K2_j ρ K2_j^†,
    the composition E2(E1(ρ)) has Kraus operators {K2_j K1_i}.
    
    Args:
        kraus_list1: Kraus operators for first channel (applied first)
        kraus_list2: Kraus operators for second channel (applied second)
    
    Returns:
        List of composed Kraus operators
    """
    composed = []
    for K2 in kraus_list2:
        for K1 in kraus_list1:
            composed.append(K2 @ K1)
    
    return composed


def state_prep_error_dm(p_excited: float) -> np.ndarray:
    """
    Create imperfect initial state density matrix for single qubit.
    
    Instead of pure |0⟩, initialize to mixed state:
        ρ_0 = (1 - p) |0⟩⟨0| + p |1⟩⟨1|
    
    Args:
        p_excited: Probability of being in |1⟩ state after reset
    
    Returns:
        2x2 density matrix
    """
    if p_excited < 0 or p_excited > 1:
        raise ValueError("p_excited must be in [0, 1]")
    
    rho = np.array([
        [1.0 - p_excited, 0.0],
        [0.0, p_excited]
    ], dtype=np.complex128)
    
    return rho


def measurement_povm_operators(p_1given0: float, p_0given1: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create POVM (Positive Operator-Valued Measure) for asymmetric measurement errors.
    
    Returns measurement operators M_0, M_1 such that:
        P(measure 0 | state ρ) = Tr(M_0^† M_0 ρ)
        P(measure 1 | state ρ) = Tr(M_1^† M_1 ρ)
    
    Args:
        p_1given0: P(measure 1 | true state is |0⟩) - false positive rate
        p_0given1: P(measure 0 | true state is |1⟩) - false negative rate
    
    Returns:
        (M_0, M_1): POVM operators as 2x2 matrices
    """
    if not (0 <= p_1given0 <= 1 and 0 <= p_0given1 <= 1):
        raise ValueError("POVM probabilities must be in [0, 1]")
    
    # M_0 = [[sqrt(1 - p_1given0), 0], [0, sqrt(p_0given1)]]
    # M_1 = [[sqrt(p_1given0), 0], [0, sqrt(1 - p_0given1)]]
    
    M_0 = np.array([
        [np.sqrt(1.0 - p_1given0), 0.0],
        [0.0, np.sqrt(p_0given1)]
    ], dtype=np.complex128)
    
    M_1 = np.array([
        [np.sqrt(p_1given0), 0.0],
        [0.0, np.sqrt(1.0 - p_0given1)]
    ], dtype=np.complex128)
    
    # Verify completeness: M_0^† M_0 + M_1^† M_1 = I
    completeness = M_0.conj().T @ M_0 + M_1.conj().T @ M_1
    if not np.allclose(completeness, np.eye(2), atol=1e-10):
        raise ValueError("POVM operators do not satisfy completeness relation")
    
    return M_0, M_1


def apply_povm_measurement(state: np.ndarray, M_operators: List[np.ndarray]) -> List[float]:
    """
    Apply POVM measurement to a quantum state and return outcome probabilities.
    
    Args:
        state: Density matrix (2x2 for single qubit) or state vector
        M_operators: List of POVM operators
    
    Returns:
        List of probabilities for each measurement outcome
    """
    # Convert state vector to density matrix if needed
    if state.ndim == 1:
        state_dm = np.outer(state, state.conj())
    else:
        state_dm = state
    
    probs = []
    for M in M_operators:
        # P(outcome) = Tr(M^† M ρ)
        prob = np.real(np.trace(M.conj().T @ M @ state_dm))
        probs.append(float(prob))
    
    # Normalize to handle numerical errors
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    
    return probs

