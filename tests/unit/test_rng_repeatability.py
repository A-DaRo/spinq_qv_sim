"""
Unit tests for RNG repeatability and determinism.

Tests that identical seeds produce identical random number sequences,
and that subsystem RNGs are independent but reproducible.
"""

import pytest
import numpy as np

from spinq_qv.utils.rng import RNGManager, initialize_global_rng, get_rng


class TestRNGDeterminism:
    """Tests for RNG deterministic behavior."""
    
    def test_same_seed_produces_same_sequence(self):
        """Test that identical seeds yield identical random sequences."""
        seed = 42
        
        # First manager
        rng1 = RNGManager(global_seed=seed)
        gen1 = rng1.get_rng("test")
        sequence1 = gen1.random(10)
        
        # Second manager with same seed
        rng2 = RNGManager(global_seed=seed)
        gen2 = rng2.get_rng("test")
        sequence2 = gen2.random(10)
        
        # Should be identical
        np.testing.assert_array_equal(sequence1, sequence2)
    
    def test_different_seeds_produce_different_sequences(self):
        """Test that different seeds yield different sequences."""
        rng1 = RNGManager(global_seed=42)
        gen1 = rng1.get_rng("test")
        sequence1 = gen1.random(10)
        
        rng2 = RNGManager(global_seed=123)
        gen2 = rng2.get_rng("test")
        sequence2 = gen2.random(10)
        
        # Should be different (with overwhelming probability)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(sequence1, sequence2)
    
    def test_multiple_calls_maintain_state(self):
        """Test that RNG state advances correctly across calls."""
        seed = 42
        
        # Generate sequence in one go
        rng1 = RNGManager(global_seed=seed)
        gen1 = rng1.get_rng("test")
        full_sequence = gen1.random(20)
        
        # Generate same sequence in chunks
        rng2 = RNGManager(global_seed=seed)
        gen2 = rng2.get_rng("test")
        chunk1 = gen2.random(10)
        chunk2 = gen2.random(10)
        chunked_sequence = np.concatenate([chunk1, chunk2])
        
        # Should match
        np.testing.assert_array_equal(full_sequence, chunked_sequence)
    
    def test_reset_restores_determinism(self):
        """Test that reset() produces the same sequence again."""
        seed = 42
        
        rng = RNGManager(global_seed=seed)
        gen = rng.get_rng("test")
        sequence1 = gen.random(10)
        
        # Reset with same seed
        rng.reset(global_seed=seed)
        gen = rng.get_rng("test")
        sequence2 = gen.random(10)
        
        # Should be identical
        np.testing.assert_array_equal(sequence1, sequence2)
    
    def test_none_seed_is_non_deterministic(self):
        """Test that None seed produces different sequences."""
        rng1 = RNGManager(global_seed=None)
        gen1 = rng1.get_rng("test")
        sequence1 = gen1.random(10)
        
        rng2 = RNGManager(global_seed=None)
        gen2 = rng2.get_rng("test")
        sequence2 = gen2.random(10)
        
        # Should be different (with high probability)
        # Note: There's a tiny chance this could fail, but it's negligible
        assert not np.allclose(sequence1, sequence2, rtol=1e-10)


class TestSubsystemIndependence:
    """Tests for subsystem RNG independence."""
    
    def test_subsystems_have_different_seeds(self):
        """Test that different subsystems get different seeds."""
        rng = RNGManager(global_seed=42)
        
        state = rng.get_state_summary()
        # Initially empty
        assert len(state) == 0
        
        # Create subsystems
        gen1 = rng.get_rng("circuits")
        gen2 = rng.get_rng("noise")
        
        state = rng.get_state_summary()
        
        # Different seeds
        assert state["circuits"]["seed"] != state["noise"]["seed"]
    
    def test_subsystems_are_independent(self):
        """Test that subsystem RNGs produce independent sequences."""
        seed = 42
        
        rng = RNGManager(global_seed=seed)
        
        circuits_rng = rng.get_rng("circuits")
        noise_rng = rng.get_rng("noise")
        
        circuits_seq = circuits_rng.random(10)
        noise_seq = noise_rng.random(10)
        
        # Sequences should be different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(circuits_seq, noise_seq)
    
    def test_subsystems_are_reproducible(self):
        """Test that subsystems produce same sequence across manager instances."""
        seed = 42
        
        # First manager
        rng1 = RNGManager(global_seed=seed)
        circuits_seq1 = rng1.get_rng("circuits").random(10)
        noise_seq1 = rng1.get_rng("noise").random(10)
        
        # Second manager with same seed
        rng2 = RNGManager(global_seed=seed)
        circuits_seq2 = rng2.get_rng("circuits").random(10)
        noise_seq2 = rng2.get_rng("noise").random(10)
        
        # Same subsystem should produce same sequence
        np.testing.assert_array_equal(circuits_seq1, circuits_seq2)
        np.testing.assert_array_equal(noise_seq1, noise_seq2)
    
    def test_subsystem_order_matters(self):
        """Test that order of subsystem creation affects seeds."""
        seed = 42
        
        # Order 1: circuits then noise
        rng1 = RNGManager(global_seed=seed)
        rng1.get_rng("circuits")
        rng1.get_rng("noise")
        state1 = rng1.get_state_summary()
        
        # Order 2: noise then circuits
        rng2 = RNGManager(global_seed=seed)
        rng2.get_rng("noise")
        rng2.get_rng("circuits")
        state2 = rng2.get_state_summary()
        
        # Seeds depend on creation order
        assert state1["circuits"]["seed"] != state2["circuits"]["seed"]
        assert state1["noise"]["seed"] != state2["noise"]["seed"]
    
    def test_same_subsystem_returns_same_generator(self):
        """Test that repeated get_rng calls return the same generator."""
        rng = RNGManager(global_seed=42)
        
        gen1 = rng.get_rng("test")
        gen2 = rng.get_rng("test")
        
        # Same instance (state is shared)
        assert gen1 is gen2
        
        # State advances
        seq1 = gen1.random(5)
        seq2 = gen2.random(5)  # Should continue from gen1's state
        
        # Not the same values
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(seq1, seq2)


class TestRNGStateTracking:
    """Tests for RNG state tracking and metadata."""
    
    def test_call_count_increments(self):
        """Test that call counter increments correctly."""
        rng = RNGManager(global_seed=42)
        
        gen = rng.get_rng("test")
        state = rng.get_state_summary()
        assert state["test"]["call_count"] == 1
        
        gen = rng.get_rng("test")
        state = rng.get_state_summary()
        assert state["test"]["call_count"] == 2
        
        gen = rng.get_rng("test")
        state = rng.get_state_summary()
        assert state["test"]["call_count"] == 3
    
    def test_state_summary_structure(self):
        """Test that state summary has expected structure."""
        rng = RNGManager(global_seed=42)
        
        rng.get_rng("circuits")
        rng.get_rng("noise")
        
        state = rng.get_state_summary()
        
        assert "circuits" in state
        assert "noise" in state
        
        for subsystem in ["circuits", "noise"]:
            assert "seed" in state[subsystem]
            assert "call_count" in state[subsystem]
            assert isinstance(state[subsystem]["seed"], (int, np.integer))
            assert isinstance(state[subsystem]["call_count"], int)
    
    def test_seed_sequence_generation(self):
        """Test that seed_sequence generates correct number of seeds."""
        rng = RNGManager(global_seed=42)
        
        seeds = rng.seed_sequence("parallel", n=10)
        
        assert len(seeds) == 10
        assert all(isinstance(s, (int, np.integer)) for s in seeds)
        
        # Seeds should be different
        assert len(set(seeds)) == 10
    
    def test_seed_sequence_is_reproducible(self):
        """Test that seed sequences are reproducible."""
        seed = 42
        
        rng1 = RNGManager(global_seed=seed)
        seeds1 = rng1.seed_sequence("parallel", n=5)
        
        rng2 = RNGManager(global_seed=seed)
        seeds2 = rng2.seed_sequence("parallel", n=5)
        
        np.testing.assert_array_equal(seeds1, seeds2)


class TestGlobalRNGManager:
    """Tests for global RNG manager singleton."""
    
    def test_initialize_global_rng(self):
        """Test global RNG initialization."""
        mgr = initialize_global_rng(seed=42)
        
        assert mgr is not None
        assert mgr.global_seed == 42
    
    def test_get_rng_convenience_function(self):
        """Test convenience function for getting RNG."""
        initialize_global_rng(seed=42)
        
        gen1 = get_rng("test")
        seq1 = gen1.random(5)
        
        # Should use global manager
        gen2 = get_rng("test")
        seq2 = gen2.random(5)
        
        # Same generator instance
        assert gen1 is gen2
        
        # Sequences continue (not restarted)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(seq1, seq2)
    
    def test_reinitialize_resets_global_manager(self):
        """Test that reinitializing resets the global manager."""
        initialize_global_rng(seed=42)
        gen1 = get_rng("test")
        seq1 = gen1.random(5)
        
        # Reinitialize with same seed
        initialize_global_rng(seed=42)
        gen2 = get_rng("test")
        seq2 = gen2.random(5)
        
        # Should produce same sequence (fresh start)
        np.testing.assert_array_equal(seq1, seq2)


class TestNumpyIntegration:
    """Tests for integration with numpy random API."""
    
    def test_supports_standard_numpy_distributions(self):
        """Test that RNG supports standard numpy distributions."""
        rng = RNGManager(global_seed=42)
        gen = rng.get_rng("test")
        
        # Standard distributions should work
        uniform = gen.uniform(0, 1, size=10)
        assert uniform.shape == (10,)
        assert all(0 <= x <= 1 for x in uniform)
        
        normal = gen.normal(0, 1, size=10)
        assert normal.shape == (10,)
        
        integers = gen.integers(0, 100, size=10)
        assert integers.shape == (10,)
        assert all(0 <= x < 100 for x in integers)
    
    def test_numpy_dtypes_preserved(self):
        """Test that numpy dtypes are correctly handled."""
        rng = RNGManager(global_seed=42)
        gen = rng.get_rng("test")
        
        float64 = gen.random(5)
        assert float64.dtype == np.float64
        
        int64 = gen.integers(0, 100, size=5)
        assert int64.dtype in [np.int64, np.int32]  # Platform dependent
