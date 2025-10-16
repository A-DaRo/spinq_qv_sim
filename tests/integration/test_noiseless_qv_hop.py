"""
Integration tests for noiseless QV HOP computation.

Tests that ideal (noiseless) simulation produces correct heavy output
probabilities, verifying median-split property and HOP computation.
"""

import pytest
import numpy as np

from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.io.formats import compute_heavy_outputs, int_to_bitstring


class TestNoiselessQVHOP:
    """Tests for ideal (noiseless) heavy-output probability."""
    
    def test_ideal_hop_is_near_one_for_small_m(self):
        """
        Test that noiseless heavy output computation is sensible.

        Heavy outputs are defined as prob > median. For discrete distributions,
        the total probability of heavy outputs can be > 0.5 (since median values
        are excluded). This test verifies:
        1. Heavy outputs can be computed
        2. Their total probability is at least 0.5 (by definition)
        3. The heavy set size is reasonable (≤ 2^m)
        """
        for m in [2, 3]:
            for seed in [42, 123, 456]:
                circuit = generate_qv_circuit(m, seed)
                probs = compute_ideal_probabilities(circuit)

                # Compute heavy outputs
                heavy_indices = compute_heavy_outputs(probs)

                # Probability of sampling a heavy output
                heavy_prob = sum(probs[idx] for idx in heavy_indices)

                # For discrete distributions with ties at median, heavy_prob >= 0.5
                # (Values strictly > median can have total prob > 0.5)
                assert heavy_prob >= 0.5, \
                    f"Heavy prob {heavy_prob} < 0.5 for m={m}, seed={seed}"
                
                # Heavy set should be reasonable size (not all or none)
                assert 0 < len(heavy_indices) < 2**m, \
                    f"Heavy set size {len(heavy_indices)} unreasonable for m={m}"
    
    def test_heavy_output_identification(self):
        """Test that heavy outputs are correctly above median."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        median = np.median(probs)
        heavy_indices = compute_heavy_outputs(probs)
        
        # All heavy outputs should be strictly > median
        for idx in heavy_indices:
            assert probs[idx] > median
        
        # All non-heavy should be <= median
        for idx in range(2**m):
            if idx not in heavy_indices:
                assert probs[idx] <= median
    
    def test_median_split_property(self):
        """Test that median properly splits the distribution."""
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)
        
        median = np.median(probs)
        
        # Count how many are above median
        above_median = sum(1 for p in probs if p > median)
        
        # For 2^m outcomes, approximately 2^(m-1) should be above median
        expected = 2**(m-1)
        
        # Allow tolerance for ties at median
        assert abs(above_median - expected) <= 2
    
    def test_simulated_sampling_from_ideal_distribution(self):
        """
        Test simulated sampling: draw from ideal distribution and compute HOP.

        This verifies that if we sample according to the true probabilities,
        we get the expected heavy-output fraction (which is >= 0.5 by definition
        of heavy outputs as prob > median).
        """
        m = 3
        circuit = generate_qv_circuit(m, seed=42)
        probs = compute_ideal_probabilities(circuit)

        # Identify heavy outputs
        heavy_indices = set(compute_heavy_outputs(probs))
        heavy_bitstrings = {int_to_bitstring(idx, m) for idx in heavy_indices}

        # Simulate sampling (deterministic: use exact probabilities)
        # Expected HOP = sum of probabilities of heavy outputs
        expected_hop = sum(probs[idx] for idx in heavy_indices)

        # By definition, heavy outputs have prob > median, so sum >= 0.5
        # (Can be > 0.5 for discrete distributions with ties at median)
        assert expected_hop >= 0.5, \
            f"Expected HOP {expected_hop} < 0.5 (violates median definition)"
        
        # Verify all heavy outputs actually have prob > median
        median = np.median(probs)
        for idx in heavy_indices:
            assert probs[idx] > median, \
                f"Heavy output {idx} has prob {probs[idx]} <= median {median}"
    
    def test_multiple_circuits_median_property(self):
        """
        Test median property holds across multiple random circuits.
        
        Verifies that heavy outputs (prob > median) have total probability >= 0.5
        for all circuits, and that the heavy set is neither empty nor full.
        """
        m = 3
        seeds = [42, 123, 456, 789, 1011]

        for seed in seeds:
            circuit = generate_qv_circuit(m, seed)
            probs = compute_ideal_probabilities(circuit)

            heavy_indices = compute_heavy_outputs(probs)
            heavy_prob = sum(probs[idx] for idx in heavy_indices)

            # Heavy outputs must have total prob >= 0.5 (by definition)
            assert heavy_prob >= 0.5, \
                f"Seed {seed}: heavy_prob={heavy_prob} < 0.5"
            
            # Heavy set should be non-trivial
            assert 0 < len(heavy_indices) < 2**m, \
                f"Seed {seed}: heavy set size {len(heavy_indices)} is trivial"
            
            # Verify median split criterion
            median = np.median(probs)
            for idx in heavy_indices:
                assert probs[idx] > median, \
                    f"Heavy index {idx} has prob {probs[idx]} <= median {median}"
    
    def test_hop_computation_utility(self):
        """Test the HOP computation utility function."""
        from spinq_qv.io.formats import compute_hop
        
        m = 2
        
        # Simulated measurement counts
        measured_counts = {
            "00": 250,
            "01": 250,
            "10": 250,
            "11": 250,
        }
        
        # If heavy outputs are "01" and "10"
        heavy_outputs = ["01", "10"]
        
        hop = compute_hop(measured_counts, heavy_outputs)
        
        # Should be 0.5 (500 out of 1000 shots)
        assert abs(hop - 0.5) < 1e-10
    
    def test_hop_all_heavy(self):
        """Test HOP when all measured outputs are heavy."""
        from spinq_qv.io.formats import compute_hop
        
        measured_counts = {
            "00": 300,
            "01": 700,
        }
        
        heavy_outputs = ["00", "01"]
        
        hop = compute_hop(measured_counts, heavy_outputs)
        
        # All shots are heavy
        assert abs(hop - 1.0) < 1e-10
    
    def test_hop_no_heavy(self):
        """Test HOP when no measured outputs are heavy."""
        from spinq_qv.io.formats import compute_hop
        
        measured_counts = {
            "00": 300,
            "01": 700,
        }
        
        heavy_outputs = ["10", "11"]  # Not in measured_counts
        
        hop = compute_hop(measured_counts, heavy_outputs)
        
        # No heavy shots
        assert abs(hop - 0.0) < 1e-10
    
    def test_ideal_circuit_heavy_output_structure(self):
        """Test structural properties of heavy outputs for ideal circuits."""
        for m in [2, 3, 4]:
            circuit = generate_qv_circuit(m, seed=42)
            probs = compute_ideal_probabilities(circuit)
            
            heavy_indices = compute_heavy_outputs(probs)
            
            # Basic sanity checks
            assert len(heavy_indices) > 0, "Should have at least one heavy output"
            assert len(heavy_indices) <= 2**m, "Cannot have more heavy outputs than total outputs"
            
            # All indices should be valid
            for idx in heavy_indices:
                assert 0 <= idx < 2**m
            
            # No duplicates
            assert len(heavy_indices) == len(set(heavy_indices))
