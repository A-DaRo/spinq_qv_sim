"""
Unit tests for density-matrix simulator Kraus channel application.

Verifies that density-matrix propagation via Kraus operators produces
expected results for known quantum channels.
"""

import pytest
import numpy as np
from spinq_qv.sim import DensityMatrixBackend
from spinq_qv.noise.channels import (
    amplitude_damping_kraus,
    phase_damping_kraus,
    depolarizing_kraus,
)


class TestDensityMatrixInitialization:
    """Test density matrix initialization and basic properties."""
    
    def test_init_default_state(self):
        """Test initialization to |0⟩⟨0|."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        rho = backend.get_density_matrix()
        
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(rho, expected)
    
    def test_init_from_statevector(self):
        """Test initialization from statevector |+⟩."""
        plus_state = np.array([1.0, 1.0]) / np.sqrt(2)
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        backend.init_state(plus_state)
        rho = backend.get_density_matrix()
        
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        assert np.allclose(rho, expected)
    
    def test_trace_equals_one(self):
        """Test that density matrix has unit trace."""
        backend = DensityMatrixBackend(n_qubits=2, seed=42)
        rho = backend.get_density_matrix()
        assert np.abs(np.trace(rho) - 1.0) < 1e-10
    
    def test_hermitian(self):
        """Test that density matrix is Hermitian."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        rho = backend.get_density_matrix()
        assert np.allclose(rho, rho.conj().T)


class TestKrausChannelApplication:
    """Test Kraus channel application gives expected results."""
    
    def test_amplitude_damping_complete_decay(self):
        """Test amplitude damping with γ=1 maps |1⟩ to |0⟩."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Initialize to |1⟩
        one_state = np.array([0.0, 1.0], dtype=np.complex128)
        backend.init_state(one_state)
        
        # Apply full amplitude damping
        kraus_ops = amplitude_damping_kraus(gamma=1.0)
        backend.apply_kraus(kraus_ops, [0])
        
        rho = backend.get_density_matrix()
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(rho, expected, atol=1e-10)
    
    def test_phase_damping_preserves_populations(self):
        """Test that phase damping preserves diagonal elements."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Initialize to |+⟩ = (|0⟩ + |1⟩)/√2
        plus_state = np.array([1.0, 1.0]) / np.sqrt(2)
        backend.init_state(plus_state)
        
        rho_before = backend.get_density_matrix()
        diag_before = np.diag(rho_before)
        
        # Apply phase damping
        kraus_ops = phase_damping_kraus(p_phi=0.5)
        backend.apply_kraus(kraus_ops, [0])
        
        rho_after = backend.get_density_matrix()
        diag_after = np.diag(rho_after)
        
        # Diagonal should be unchanged
        assert np.allclose(diag_after, diag_before)
    
    def test_depolarizing_approaches_maximally_mixed(self):
        """Test that full depolarizing maps to I/2."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Start with pure state |0⟩
        backend.init_state()
        
        # Apply depolarizing with p approaching 1 multiple times
        kraus_ops = depolarizing_kraus(p=0.9)
        for _ in range(10):
            backend.apply_kraus(kraus_ops, [0])
        
        rho = backend.get_density_matrix()
        
        # Should approach maximally mixed state I/2
        expected = np.eye(2, dtype=np.complex128) / 2
        assert np.allclose(rho, expected, atol=0.1)
    
    def test_identity_kraus_preserves_state(self):
        """Test that identity Kraus operator preserves state."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Initialize to arbitrary pure state
        psi = np.array([0.6, 0.8], dtype=np.complex128)
        backend.init_state(psi)
        rho_before = backend.get_density_matrix()
        
        # Apply identity channel
        I = np.eye(2, dtype=np.complex128)
        backend.apply_kraus([I], [0])
        
        rho_after = backend.get_density_matrix()
        assert np.allclose(rho_after, rho_before)
    
    def test_kraus_trace_preservation(self):
        """Test that Kraus application preserves trace."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Initialize to |+⟩
        plus_state = np.array([1.0, 1.0]) / np.sqrt(2)
        backend.init_state(plus_state)
        
        # Apply amplitude damping
        kraus_ops = amplitude_damping_kraus(gamma=0.3)
        backend.apply_kraus(kraus_ops, [0])
        
        rho = backend.get_density_matrix()
        trace = np.trace(rho)
        assert np.abs(trace - 1.0) < 1e-10


class TestUnitaryGates:
    """Test unitary gate application on density matrices."""
    
    def test_pauli_x_gate(self):
        """Test Pauli X flips |0⟩ to |1⟩."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        backend.init_state()  # |0⟩
        
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        backend.apply_unitary(X, [0])
        
        rho = backend.get_density_matrix()
        expected = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        assert np.allclose(rho, expected)
    
    def test_hadamard_creates_superposition(self):
        """Test Hadamard creates equal superposition."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        backend.init_state()  # |0⟩
        
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        backend.apply_unitary(H, [0])
        
        rho = backend.get_density_matrix()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        assert np.allclose(rho, expected)


class TestMemorySafety:
    """Test memory safety checks for large qubit counts."""
    
    def test_warning_for_large_n(self):
        """Test that large n_qubits triggers warning."""
        with pytest.warns(ResourceWarning):
            backend = DensityMatrixBackend(n_qubits=13, seed=42)
    
    def test_error_for_excessive_n(self):
        """Test that excessive n_qubits raises error."""
        with pytest.raises(ValueError, match="too much memory"):
            backend = DensityMatrixBackend(n_qubits=17, seed=42)


class TestMeasurement:
    """Test measurement sampling from density matrices."""
    
    def test_measure_pure_state(self):
        """Test measurement of pure |0⟩ state."""
        backend = DensityMatrixBackend(n_qubits=2, seed=42)
        backend.init_state()  # |00⟩
        
        counts = backend.measure(shots=100)
        assert '00' in counts
        assert counts['00'] == 100
    
    def test_measure_mixed_state(self):
        """Test measurement of mixed state gives distribution."""
        backend = DensityMatrixBackend(n_qubits=1, seed=42)
        
        # Create mixed state: 50% |0⟩, 50% |1⟩
        rho_mixed = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
        backend.init_state(rho_mixed)
        
        counts = backend.measure(shots=1000)
        
        # Should get roughly 500 of each
        assert '0' in counts and '1' in counts
        assert 400 < counts.get('0', 0) < 600
        assert 400 < counts.get('1', 0) < 600
