"""
Unit tests for StatevectorBackend basic functionality.

Tests single-qubit gates, measurement sampling, and state manipulation.
"""

import pytest
import numpy as np

from spinq_qv.sim import StatevectorBackend


class TestStatevectorInitialization:
    """Tests for backend initialization and state setup."""
    
    def test_init_default_state(self):
        """Test that default initialization creates |0...0⟩."""
        backend = StatevectorBackend(n_qubits=3, seed=42)
        
        state = backend.get_statevector()
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = 1.0
        
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_init_custom_state(self):
        """Test initialization with custom statevector."""
        n_qubits = 2
        custom_state = np.array([0.6, 0.8, 0, 0], dtype=np.complex128)
        
        backend = StatevectorBackend(n_qubits=n_qubits, seed=42)
        backend.init_state(custom_state)
        
        state = backend.get_statevector()
        np.testing.assert_allclose(state, custom_state, atol=1e-15)
    
    def test_init_normalizes_state(self):
        """Test that unnormalized states are automatically normalized."""
        n_qubits = 2
        unnormalized = np.array([2.0, 2.0, 0, 0], dtype=np.complex128)
        
        backend = StatevectorBackend(n_qubits=n_qubits, seed=42)
        backend.init_state(unnormalized)
        
        state = backend.get_statevector()
        norm = np.linalg.norm(state)
        
        assert abs(norm - 1.0) < 1e-10
        expected = unnormalized / np.linalg.norm(unnormalized)
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_probabilities_sum_to_one(self):
        """Test that probabilities always sum to 1."""
        backend = StatevectorBackend(n_qubits=3, seed=42)
        probs = backend.get_probabilities()
        
        assert abs(probs.sum() - 1.0) < 1e-10
        assert np.all(probs >= 0)


class TestSingleQubitGates:
    """Tests for single-qubit gate application."""
    
    def test_pauli_x_gate(self):
        """Test X gate flips |0⟩ to |1⟩."""
        backend = StatevectorBackend(n_qubits=1, seed=42)
        
        # Pauli X matrix
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        backend.apply_unitary(X, [0])
        
        state = backend.get_statevector()
        expected = np.array([0, 1], dtype=np.complex128)
        
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_hadamard_gate(self):
        """Test Hadamard creates equal superposition."""
        backend = StatevectorBackend(n_qubits=1, seed=42)
        
        # Hadamard matrix
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        backend.apply_unitary(H, [0])
        
        state = backend.get_statevector()
        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_single_qubit_gate_on_specific_qubit(self):
        """Test applying X gate to qubit 1 in 3-qubit system."""
        backend = StatevectorBackend(n_qubits=3, seed=42)
        
        # Apply X to qubit 1 (middle qubit)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        backend.apply_unitary(X, [1])
        
        # State should be |010⟩
        state = backend.get_statevector()
        expected = np.zeros(8, dtype=np.complex128)
        expected[0b010] = 1.0  # Binary 010 = decimal 2
        
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_u3_gate_implementation(self):
        """Test U3 gate matrix construction."""
        backend = StatevectorBackend(n_qubits=1, seed=42)
        
        # U3(π/2, 0, 0) should be Ry(π/2)
        theta, phi, lam = np.pi/2, 0.0, 0.0
        u3_matrix = backend._u3_matrix(theta, phi, lam)
        
        backend.apply_unitary(u3_matrix, [0])
        
        state = backend.get_statevector()
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
        
        np.testing.assert_allclose(state, expected, atol=1e-15)


class TestTwoQubitGates:
    """Tests for two-qubit gate application."""
    
    def test_cnot_gate(self):
        """Test CNOT gate on |10⟩ → |11⟩."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        
        # Prepare |10⟩ state
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        backend.apply_unitary(X, [0])
        
        # Apply CNOT (control=0, target=1)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        backend.apply_unitary(CNOT, [0, 1])
        
        state = backend.get_statevector()
        expected = np.zeros(4, dtype=np.complex128)
        expected[0b11] = 1.0  # |11⟩
        
        np.testing.assert_allclose(state, expected, atol=1e-15)
    
    def test_bell_state_creation(self):
        """Test creating Bell state (|00⟩ + |11⟩)/√2."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        
        # Apply H on qubit 0
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        backend.apply_unitary(H, [0])
        
        # Apply CNOT
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        backend.apply_unitary(CNOT, [0, 1])
        
        state = backend.get_statevector()
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        
        np.testing.assert_allclose(state, expected, atol=1e-15)


class TestMeasurement:
    """Tests for measurement sampling."""
    
    def test_measure_computational_basis(self):
        """Test measuring |0⟩ always gives '0'."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        
        counts = backend.measure(shots=100)
        
        assert len(counts) == 1
        assert '00' in counts
        assert counts['00'] == 100
    
    def test_measure_x_state(self):
        """Test measuring |1⟩ always gives '1'."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        
        # Prepare |11⟩
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        backend.apply_unitary(X, [0])
        backend.apply_unitary(X, [1])
        
        counts = backend.measure(shots=100)
        
        assert len(counts) == 1
        assert '11' in counts
        assert counts['11'] == 100
    
    def test_measure_superposition(self):
        """Test measuring Hadamard state gives approximately 50/50 split."""
        backend = StatevectorBackend(n_qubits=1, seed=42)
        
        # Create |+⟩ = (|0⟩ + |1⟩)/√2
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        backend.apply_unitary(H, [0])
        
        counts = backend.measure(shots=10000)
        
        # Check both outcomes appear
        assert '0' in counts
        assert '1' in counts
        
        # Check approximate 50/50 distribution (within 3σ)
        # Standard deviation for binomial: sqrt(n*p*(1-p)) = sqrt(10000*0.5*0.5) = 50
        # 3σ = 150, so allow ±150 from 5000
        assert 4850 <= counts['0'] <= 5150
        assert 4850 <= counts['1'] <= 5150
    
    def test_measurement_reproducibility(self):
        """Test that same seed produces same measurement outcomes."""
        backend1 = StatevectorBackend(n_qubits=2, seed=12345)
        backend2 = StatevectorBackend(n_qubits=2, seed=12345)
        
        # Create superposition
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        backend1.apply_unitary(H, [0])
        backend2.apply_unitary(H, [0])
        
        counts1 = backend1.measure(shots=100)
        counts2 = backend2.measure(shots=100)
        
        # Should be identical with same seed
        assert counts1 == counts2
    
    def test_measurement_statistics_match_probabilities(self):
        """Test empirical frequencies match Born rule probabilities."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        
        # Create unequal superposition: 0.6|00⟩ + 0.8|11⟩
        state = np.array([0.6, 0, 0, 0.8], dtype=np.complex128)
        backend.init_state(state)
        
        # Get ideal probabilities
        ideal_probs = backend.get_probabilities()
        
        # Sample many times
        counts = backend.measure(shots=100000)
        
        # Compute empirical probabilities
        empirical = {
            '00': counts.get('00', 0) / 100000,
            '11': counts.get('11', 0) / 100000,
        }
        
        # Should match within statistical tolerance
        # For 100k shots: std dev ≈ sqrt(p*(1-p)/n) ≈ 0.0015
        # Use 5σ tolerance
        assert abs(empirical['00'] - 0.36) < 0.01  # 0.6^2 = 0.36
        assert abs(empirical['11'] - 0.64) < 0.01  # 0.8^2 = 0.64


class TestCircuitApplication:
    """Tests for applying full circuits."""
    
    def test_apply_qv_circuit(self):
        """Test applying a QV circuit generates valid final state."""
        from spinq_qv.circuits import generate_qv_circuit
        
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        
        # Check state is normalized
        state = backend.get_statevector()
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
        
        # Check probabilities sum to 1
        probs = backend.get_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-10
    
    def test_circuit_matches_ideal_probabilities(self):
        """Test that backend produces same probabilities as ideal computation."""
        from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
        
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        # Compute ideal probabilities (from generator.py)
        ideal_probs = compute_ideal_probabilities(circuit)
        
        # Compute using backend
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        backend_probs = backend.get_probabilities()
        
        # Should match exactly (both use same math)
        np.testing.assert_allclose(ideal_probs, backend_probs, atol=1e-12)
