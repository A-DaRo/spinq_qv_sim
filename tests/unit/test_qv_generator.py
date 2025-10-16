"""
Unit tests for QV circuit generator.

Tests determinism, probability normalization, and heavy-output set properties.
"""

import pytest
import numpy as np

from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.io.formats import compute_heavy_outputs


class TestQVCircuitGeneration:
    """Tests for QV circuit generation."""
    
    def test_same_seed_produces_same_circuit(self):
        """Test that identical seeds yield identical circuits."""
        m = 3
        seed = 42
        
        circuit1 = generate_qv_circuit(m, seed)
        circuit2 = generate_qv_circuit(m, seed)
        
        # Should have same structure
        assert circuit1.width == circuit2.width
        assert circuit1.depth == circuit2.depth
        assert circuit1.seed == circuit2.seed
        assert len(circuit1.gates) == len(circuit2.gates)
        
        # Gates should be identical
        for g1, g2 in zip(circuit1.gates, circuit2.gates):
            assert g1["type"] == g2["type"]
            assert g1["qubits"] == g2["qubits"]
            
            # Parameters should match (within numerical precision)
            if "params" in g1:
                for key in g1["params"]:
                    if isinstance(g1["params"][key], list):
                        np.testing.assert_allclose(
                            g1["params"][key],
                            g2["params"][key],
                            rtol=1e-15
                        )
                    else:
                        assert abs(g1["params"][key] - g2["params"][key]) < 1e-15
    
    def test_different_seeds_produce_different_circuits(self):
        """Test that different seeds yield different circuits."""
        m = 3
        
        circuit1 = generate_qv_circuit(m, seed=42)
        circuit2 = generate_qv_circuit(m, seed=123)
        
        # Seeds should differ
        assert circuit1.seed != circuit2.seed
        
        # At least some gate parameters should differ
        # (Extremely unlikely to be identical with different seeds)
        params_differ = False
        for g1, g2 in zip(circuit1.gates, circuit2.gates):
            if "params" in g1 and "params" in g2:
                for key in g1["params"]:
                    if isinstance(g1["params"][key], (int, float)):
                        if abs(g1["params"][key] - g2["params"][key]) > 1e-10:
                            params_differ = True
                            break
            if params_differ:
                break
        
        assert params_differ, "Different seeds should produce different parameters"
    
    def test_circuit_dimensions(self):
        """Test that circuit has correct width and depth."""
        for m in [2, 3, 4, 5]:
            circuit = generate_qv_circuit(m, seed=42)
            
            assert circuit.width == m
            assert circuit.depth == m
            assert len(circuit.gates) > 0
    
    def test_circuit_has_single_and_two_qubit_gates(self):
        """Test that QV circuit contains both 1Q and 2Q gates."""
        m = 4
        circuit = generate_qv_circuit(m, seed=42)
        
        single_qubit = circuit.num_single_qubit_gates()
        two_qubit = circuit.num_two_qubit_gates()
        
        assert single_qubit > 0, "Should have single-qubit gates"
        assert two_qubit > 0, "Should have two-qubit gates"
    
    def test_qubit_indices_valid(self):
        """Test that all qubit indices are within valid range."""
        m = 5
        circuit = generate_qv_circuit(m, seed=42)
        
        for gate in circuit.gates:
            for qubit in gate["qubits"]:
                assert 0 <= qubit < m, f"Qubit index {qubit} out of range [0, {m})"
    
    def test_single_qubit_gates_have_u3_params(self):
        """Test that single-qubit gates have required parameters."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        for gate in circuit.gates:
            if gate["type"] == "u3":
                assert "params" in gate
                params = gate["params"]
                assert "theta" in params
                assert "phi" in params
                assert "lambda" in params
    
    def test_two_qubit_gates_have_unitary_params(self):
        """Test that two-qubit gates have unitary parameters."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        for gate in circuit.gates:
            if gate["type"] == "su4":
                assert "params" in gate
                params = gate["params"]
                assert "unitary_real" in params
                assert "unitary_imag" in params
                
                # Should be 4x4 = 16 elements
                assert len(params["unitary_real"]) == 16
                assert len(params["unitary_imag"]) == 16


class TestIdealProbabilities:
    """Tests for ideal probability computation."""
    
    def test_probabilities_sum_to_one(self):
        """Test that probabilities are properly normalized."""
        for m in [2, 3, 4]:
            circuit = generate_qv_circuit(m, seed=42)
            probs = compute_ideal_probabilities(circuit)
            
            assert len(probs) == 2**m
            np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-10)
    
    def test_probabilities_are_nonnegative(self):
        """Test that all probabilities are >= 0."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        assert np.all(probs >= 0), "All probabilities must be non-negative"
    
    def test_probabilities_are_real(self):
        """Test that probabilities are real-valued."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        assert probs.dtype == np.float64 or probs.dtype == np.float32
    
    def test_different_circuits_have_different_probabilities(self):
        """Test that different circuits produce different distributions."""
        m = 3
        
        circuit1 = generate_qv_circuit(m, seed=42)
        probs1 = compute_ideal_probabilities(circuit1)
        
        circuit2 = generate_qv_circuit(m, seed=123)
        probs2 = compute_ideal_probabilities(circuit2)
        
        # Should be different (with very high probability)
        assert not np.allclose(probs1, probs2, rtol=1e-10)
    
    def test_same_circuit_gives_same_probabilities(self):
        """Test that computing probabilities twice gives same result."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        probs1 = compute_ideal_probabilities(circuit)
        probs2 = compute_ideal_probabilities(circuit)
        
        np.testing.assert_allclose(probs1, probs2, rtol=1e-15)
    
    def test_identity_circuit_probabilities(self):
        """Test a trivial case: empty circuit should give |0...0> state."""
        from spinq_qv.io.formats import CircuitSpec
        
        m = 3
        # Create empty circuit (no gates)
        circuit = CircuitSpec(width=m, depth=0, seed=0, gates=[])
        
        probs = compute_ideal_probabilities(circuit)
        
        # Should be all weight on |000> (index 0)
        expected = np.zeros(2**m)
        expected[0] = 1.0
        
        np.testing.assert_allclose(probs, expected, rtol=1e-10)


class TestHeavyOutputs:
    """Tests for heavy-output computation."""
    
    def test_heavy_output_set_size(self):
        """Test that heavy-output set has size 2^(m-1) for random circuits."""
        # For truly random Haar distributions, heavy outputs = 2^(m-1)
        # (half the outputs are above median)
        
        for m in [2, 3, 4]:
            circuit = generate_qv_circuit(m, seed=42)
            probs = compute_ideal_probabilities(circuit)
            
            heavy_indices = compute_heavy_outputs(probs)
            
            # Should be approximately 2^(m-1), allowing for ties at median
            expected_size = 2**(m-1)
            
            # Allow some tolerance for numerical precision and ties
            assert abs(len(heavy_indices) - expected_size) <= 2, \
                f"Heavy output set size {len(heavy_indices)} far from expected {expected_size}"
    
    def test_heavy_outputs_above_median(self):
        """Test that all heavy outputs have prob > median."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        median = np.median(probs)
        heavy_indices = compute_heavy_outputs(probs)
        
        for idx in heavy_indices:
            assert probs[idx] > median, \
                f"Heavy output {idx} has prob {probs[idx]} <= median {median}"
    
    def test_non_heavy_outputs_not_above_median(self):
        """Test that non-heavy outputs are not above median."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        median = np.median(probs)
        heavy_indices = set(compute_heavy_outputs(probs))
        
        for idx in range(2**m):
            if idx not in heavy_indices:
                assert probs[idx] <= median, \
                    f"Non-heavy output {idx} has prob {probs[idx]} > median {median}"
    
    def test_uniform_distribution_heavy_outputs(self):
        """Test heavy outputs for uniform distribution."""
        m = 3
        probs = np.ones(2**m) / (2**m)  # Uniform
        
        heavy_indices = compute_heavy_outputs(probs)
        
        # For uniform distribution, median = mean
        # All outputs have same probability, so technically none are "heavy"
        # But numerically, all might be considered heavy or none
        # We just check that the function doesn't crash
        assert isinstance(heavy_indices, list)


class TestCircuitSerialization:
    """Tests for circuit serialization."""
    
    def test_circuit_to_dict(self):
        """Test circuit serialization to dict."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        circuit_dict = circuit.to_dict()
        
        assert circuit_dict["width"] == m
        assert circuit_dict["seed"] == 42
        assert "gates" in circuit_dict
        assert len(circuit_dict["gates"]) > 0
    
    def test_circuit_to_json(self):
        """Test circuit serialization to JSON."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        
        json_str = circuit.to_json()
        
        assert isinstance(json_str, str)
        assert "width" in json_str
        assert "gates" in json_str
    
    def test_circuit_json_roundtrip(self):
        """Test that circuit can be saved and loaded via JSON."""
        from spinq_qv.io.formats import CircuitSpec
        
        m = 3
        circuit1 = generate_qv_circuit(m, seed=42)
        
        # Serialize to JSON
        json_str = circuit1.to_json()
        
        # Deserialize
        circuit2 = CircuitSpec.from_json(json_str)
        
        # Should match
        assert circuit2.width == circuit1.width
        assert circuit2.depth == circuit1.depth
        assert circuit2.seed == circuit1.seed
        assert len(circuit2.gates) == len(circuit1.gates)
    
    def test_probabilities_after_deserialization(self):
        """Test that deserialized circuit gives same probabilities."""
        from spinq_qv.io.formats import CircuitSpec
        
        m = 3
        circuit1 = generate_qv_circuit(m, seed=42)
        probs1 = compute_ideal_probabilities(circuit1)
        
        # Serialize and deserialize
        circuit2 = CircuitSpec.from_json(circuit1.to_json())
        probs2 = compute_ideal_probabilities(circuit2)
        
        # Probabilities should match
        np.testing.assert_allclose(probs1, probs2, rtol=1e-10)
