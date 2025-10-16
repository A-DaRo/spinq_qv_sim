"""
Unit tests for stochastic noise application in statevector backend.
"""

import numpy as np
from spinq_qv.sim import StatevectorBackend
from spinq_qv.noise.channels import depolarizing_kraus, amplitude_damping_kraus


def test_apply_pauli_error_no_error():
    """Test that zero error probability leaves state unchanged."""
    backend = StatevectorBackend(n_qubits=1, seed=42)
    
    # Apply Hadamard to get |+⟩
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    backend.apply_unitary(H, [0])
    
    state_before = backend.get_statevector()
    
    # Apply zero error
    backend.apply_pauli_error(p=0.0, targets=[0])
    
    state_after = backend.get_statevector()
    
    assert np.allclose(state_before, state_after)


def test_apply_pauli_error_changes_state():
    """Test that non-zero error probability modifies state."""
    backend = StatevectorBackend(n_qubits=1, seed=42)
    
    # Prepare |+⟩
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    backend.apply_unitary(H, [0])
    
    state_before = backend.get_statevector()
    
    # Apply moderate error (should change state with high probability)
    backend.apply_pauli_error(p=0.8, targets=[0])
    
    state_after = backend.get_statevector()
    
    # State should be different (with very high probability given p=0.8)
    # Note: There's a small chance this fails due to randomness
    assert not np.allclose(state_before, state_after, atol=1e-10)


def test_apply_kraus_stochastic_preserves_norm():
    """Test that Kraus application preserves state normalization."""
    backend = StatevectorBackend(n_qubits=1, seed=42)
    
    # Prepare arbitrary state
    theta = 0.5
    phi = 0.3
    lam = 0.7
    U = backend._u3_matrix(theta, phi, lam)
    backend.apply_unitary(U, [0])
    
    # Apply amplitude damping channel
    gamma = 0.1
    kraus_ops = amplitude_damping_kraus(gamma)
    
    backend.apply_kraus_stochastic(kraus_ops, [0])
    
    # Check normalization
    state = backend.get_statevector()
    norm = np.linalg.norm(state)
    
    assert np.isclose(norm, 1.0, atol=1e-10)


def test_apply_kraus_stochastic_depolarizing():
    """Test stochastic depolarizing channel via Kraus operators."""
    backend = StatevectorBackend(n_qubits=1, seed=123)
    
    # Prepare |1⟩
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    backend.apply_unitary(X, [0])
    
    # Apply depolarizing channel multiple times
    p = 0.1
    kraus_ops = depolarizing_kraus(p)
    
    for _ in range(10):
        backend.apply_kraus_stochastic(kraus_ops, [0])
    
    # State should still be normalized
    state = backend.get_statevector()
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0, atol=1e-10)
    
    # Probabilities should sum to 1
    probs = backend.get_probabilities()
    assert np.isclose(probs.sum(), 1.0, atol=1e-10)


def test_stochastic_methods_with_multiple_qubits():
    """Test stochastic noise on multi-qubit states."""
    backend = StatevectorBackend(n_qubits=2, seed=42)
    
    # Prepare Bell state
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)
    
    backend.apply_unitary(H, [0])
    backend.apply_unitary(CNOT, [0, 1])
    
    # Apply noise to both qubits
    backend.apply_pauli_error(p=0.05, targets=[0, 1])
    
    # Check normalization preserved
    state = backend.get_statevector()
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0, atol=1e-10)


def test_deterministic_seeded_noise():
    """Test that seeded backends produce identical stochastic results."""
    seed = 999
    
    # First run
    backend1 = StatevectorBackend(n_qubits=1, seed=seed)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    backend1.apply_unitary(H, [0])
    backend1.apply_pauli_error(p=0.5, targets=[0])
    state1 = backend1.get_statevector()
    
    # Second run with same seed
    backend2 = StatevectorBackend(n_qubits=1, seed=seed)
    backend2.apply_unitary(H, [0])
    backend2.apply_pauli_error(p=0.5, targets=[0])
    state2 = backend2.get_statevector()
    
    # Should be identical
    assert np.allclose(state1, state2)
