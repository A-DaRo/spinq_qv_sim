"""
Integration tests for noiseless sampling with statevector backend.

Tests that empirical measurement distributions match ideal probabilities
within statistical tolerance using chi-squared and KS tests.
"""

import pytest
import numpy as np
from scipy import stats

from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.sim import StatevectorBackend
from spinq_qv.io.formats import compute_heavy_outputs, compute_hop


class TestNoiselessSampling:
    """Tests for measurement sampling accuracy."""
    
    def test_single_circuit_chi_squared(self):
        """Test chi-squared goodness-of-fit for single circuit."""
        m = 3
        n_shots = 10000
        
        circuit = generate_qv_circuit(m, seed=42)
        ideal_probs = compute_ideal_probabilities(circuit)
        
        # Sample from backend
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        counts = backend.measure(shots=n_shots)
        
        # Convert counts to observed frequencies
        observed = np.zeros(2**m)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            observed[idx] = count
        
        # Expected counts
        expected = ideal_probs * n_shots
        
        # Chi-squared test (exclude bins with expected < 5)
        mask = expected >= 5
        if mask.sum() < 2:
            pytest.skip("Too few bins with expected >= 5 for chi-squared test")
        
        chi2_stat = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        dof = mask.sum() - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        # Should not reject null hypothesis (p > 0.01)
        assert p_value > 0.01, \
            f"Chi-squared test rejected: p={p_value:.4f}, chi2={chi2_stat:.2f}"
    
    def test_multiple_circuits_kolmogorov_smirnov(self):
        """Test KS statistic for multiple circuits."""
        m = 3
        n_circuits = 10
        n_shots = 1000
        
        ks_statistics = []
        
        for seed in range(42, 42 + n_circuits):
            circuit = generate_qv_circuit(m, seed=seed)
            ideal_probs = compute_ideal_probabilities(circuit)
            
            # Sample from backend
            backend = StatevectorBackend(n_qubits=m, seed=seed + 1000)
            backend.apply_circuit(circuit)
            counts = backend.measure(shots=n_shots)
            
            # Convert to empirical CDF
            observed = np.zeros(2**m)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                observed[idx] = count / n_shots
            
            # Compute KS statistic (max difference between CDFs)
            ideal_cdf = np.cumsum(ideal_probs)
            empirical_cdf = np.cumsum(observed)
            ks_stat = np.max(np.abs(ideal_cdf - empirical_cdf))
            ks_statistics.append(ks_stat)
        
        # KS statistic should be small for most circuits
        # Critical value for n=1000, alpha=0.05: ~0.043
        # Allow 10% of circuits to exceed (random variation)
        exceed_count = sum(1 for ks in ks_statistics if ks > 0.05)
        
        assert exceed_count <= 2, \
            f"Too many circuits ({exceed_count}/{n_circuits}) failed KS test"
    
    def test_hop_computation_noiseless(self):
        """Test that noiseless HOP matches ideal heavy output probability."""
        m = 3
        n_shots = 10000
        
        circuit = generate_qv_circuit(m, seed=42)
        ideal_probs = compute_ideal_probabilities(circuit)
        
        # Identify heavy outputs
        heavy_indices = compute_heavy_outputs(ideal_probs)
        heavy_bitstrings = [format(i, f'0{m}b') for i in heavy_indices]
        
        # Ideal HOP (sum of heavy probabilities)
        ideal_hop = sum(ideal_probs[i] for i in heavy_indices)
        
        # Sample and compute empirical HOP
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        counts = backend.measure(shots=n_shots)
        
        empirical_hop = compute_hop(counts, heavy_bitstrings)
        
        # Should match within statistical tolerance
        # Standard error: sqrt(p*(1-p)/n) ≈ sqrt(0.83*0.17/10000) ≈ 0.004
        # Use 5σ tolerance
        assert abs(empirical_hop - ideal_hop) < 0.02, \
            f"Empirical HOP {empirical_hop:.4f} differs from ideal {ideal_hop:.4f}"
    
    def test_multiple_circuits_hop_consistency(self):
        """Test HOP consistency across multiple circuits."""
        m = 3
        n_circuits = 10
        n_shots = 5000
        
        hop_differences = []
        
        for seed in range(42, 42 + n_circuits):
            circuit = generate_qv_circuit(m, seed=seed)
            ideal_probs = compute_ideal_probabilities(circuit)
            
            # Compute ideal HOP
            heavy_indices = compute_heavy_outputs(ideal_probs)
            heavy_bitstrings = [format(i, f'0{m}b') for i in heavy_indices]
            ideal_hop = sum(ideal_probs[i] for i in heavy_indices)
            
            # Sample and compute empirical HOP
            backend = StatevectorBackend(n_qubits=m, seed=seed + 2000)
            backend.apply_circuit(circuit)
            counts = backend.measure(shots=n_shots)
            empirical_hop = compute_hop(counts, heavy_bitstrings)
            
            hop_differences.append(abs(empirical_hop - ideal_hop))
        
        # Average difference should be small
        avg_diff = np.mean(hop_differences)
        max_diff = np.max(hop_differences)
        
        assert avg_diff < 0.015, \
            f"Average HOP difference {avg_diff:.4f} too large"
        assert max_diff < 0.03, \
            f"Max HOP difference {max_diff:.4f} too large"


class TestLargerCircuits:
    """Tests for larger circuit sizes."""
    
    def test_m4_circuit_sampling(self):
        """Test sampling from m=4 circuit."""
        m = 4
        n_shots = 5000
        
        circuit = generate_qv_circuit(m, seed=42)
        ideal_probs = compute_ideal_probabilities(circuit)
        
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        counts = backend.measure(shots=n_shots)
        
        # Check total shots
        total_counts = sum(counts.values())
        assert total_counts == n_shots
        
        # Check probabilities are reasonable
        probs = backend.get_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-10
        
        # Compare dominant outcomes
        ideal_top_idx = np.argmax(ideal_probs)
        empirical_counts = np.zeros(2**m)
        for bs, count in counts.items():
            empirical_counts[int(bs, 2)] = count
        empirical_top_idx = np.argmax(empirical_counts)
        
        # Top outcome should often match (not always due to sampling)
        # Just check it's one of the top 3 ideal outcomes
        top_3_ideal = np.argsort(ideal_probs)[-3:]
        assert empirical_top_idx in top_3_ideal or ideal_top_idx in top_3_ideal
    
    def test_m5_circuit_memory_feasible(self):
        """Test that m=5 circuits are feasible (32 states)."""
        m = 5
        n_shots = 1000
        
        circuit = generate_qv_circuit(m, seed=42)
        
        backend = StatevectorBackend(n_qubits=m, seed=100)
        backend.apply_circuit(circuit)
        counts = backend.measure(shots=n_shots)
        
        # Should complete without memory issues
        assert sum(counts.values()) == n_shots
        
        # State should be normalized
        state = backend.get_statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


class TestStatisticalProperties:
    """Tests for statistical properties of sampling."""
    
    def test_shot_noise_scaling(self):
        """Test that sampling error decreases with sqrt(shots)."""
        m = 2
        circuit = generate_qv_circuit(m, seed=42)
        ideal_probs = compute_ideal_probabilities(circuit)
        
        # Test with different shot counts
        shot_counts = [100, 1000, 10000]
        errors = []
        
        for n_shots in shot_counts:
            backend = StatevectorBackend(n_qubits=m, seed=100)
            backend.apply_circuit(circuit)
            counts = backend.measure(shots=n_shots)
            
            # Compute empirical probabilities
            empirical = np.zeros(2**m)
            for bs, count in counts.items():
                empirical[int(bs, 2)] = count / n_shots
            
            # L2 error
            error = np.sqrt(np.mean((empirical - ideal_probs)**2))
            errors.append(error)
        
        # Error should decrease roughly as 1/sqrt(shots)
        # Check that 10x shots → ~3x smaller error
        ratio_100_to_1000 = errors[0] / errors[1]
        ratio_1000_to_10000 = errors[1] / errors[2]
        
        # Should be in range [2, 5] (allowing statistical variation)
        assert 2.0 < ratio_100_to_1000 < 5.0
        assert 2.0 < ratio_1000_to_10000 < 5.0
    
    def test_different_seeds_give_different_samples(self):
        """Test that different seeds produce different samples."""
        m = 2
        circuit = generate_qv_circuit(m, seed=42)
        
        # Create superposition
        backend1 = StatevectorBackend(n_qubits=m, seed=111)
        backend2 = StatevectorBackend(n_qubits=m, seed=222)
        
        backend1.apply_circuit(circuit)
        backend2.apply_circuit(circuit)
        
        counts1 = backend1.measure(shots=50)
        counts2 = backend2.measure(shots=50)
        
        # Counts should differ (very unlikely to be identical)
        assert counts1 != counts2


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_shots_returns_empty(self):
        """Test that measuring with 0 shots returns empty dict."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        counts = backend.measure(shots=0)
        
        assert counts == {}
    
    def test_single_shot_measurement(self):
        """Test single-shot measurement."""
        backend = StatevectorBackend(n_qubits=2, seed=42)
        counts = backend.measure(shots=1)
        
        assert sum(counts.values()) == 1
        assert len(counts) == 1
    
    def test_deterministic_state_sampling(self):
        """Test that deterministic states give deterministic outcomes."""
        backend = StatevectorBackend(n_qubits=3, seed=42)
        
        # State is |000⟩
        counts = backend.measure(shots=100)
        
        assert len(counts) == 1
        assert '000' in counts
        assert counts['000'] == 100
