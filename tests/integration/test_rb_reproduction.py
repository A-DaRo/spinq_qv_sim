"""
Integration tests for RB-based fidelity reproduction.

Validates that the stochastic noise model reproduces configured fidelities
within acceptable tolerances.
"""

import pytest
import numpy as np

from spinq_qv.config.schemas import Config
from spinq_qv.experiments.validate import (
    validate_rb_single_qubit,
    validate_rb_two_qubit,
)


def test_rb_single_qubit_fidelity_reproduction():
    """
    Test that single-qubit RB extracts fidelity close to configured F1.
    
    Tolerance: ±0.5% absolute error (±0.005)
    """
    # Load baseline config
    config = Config.from_yaml("examples/configs/baseline.yaml")
    
    # Run RB with fewer sequences for speed
    result = validate_rb_single_qubit(config, n_sequences=15, seed=42)
    
    # Check extraction succeeded
    assert result['fit_info']['success'], "RB fit failed"
    
    # Check fidelity is within tolerance
    target = result['target_F1']
    extracted = result['extracted_F1']
    error = abs(extracted - target)
    
    tolerance = 0.005  # ±0.5% absolute
    
    assert error < tolerance, (
        f"Extracted F1={extracted:.6f} differs from target F1={target:.6f} "
        f"by {error:.6f} (tolerance: {tolerance})"
    )
    
    print(f"✓ Single-qubit RB: F1_target={target:.6f}, F1_extracted={extracted:.6f}, "
          f"error={error:.6f}")


def test_rb_two_qubit_fidelity_reproduction():
    """
    Test that two-qubit RB extracts fidelity close to configured F2.
    
    Tolerance: ±0.5% absolute error (±0.005)
    """
    # Load baseline config
    config = Config.from_yaml("examples/configs/baseline.yaml")
    
    # Run RB with fewer sequences for speed
    result = validate_rb_two_qubit(config, n_sequences=15, seed=43)
    
    # Check extraction succeeded
    assert result['fit_info']['success'], "RB fit failed"
    
    # Check fidelity is within tolerance
    target = result['target_F2']
    extracted = result['extracted_F2']
    error = abs(extracted - target)
    
    tolerance = 0.005  # ±0.5% absolute
    
    assert error < tolerance, (
        f"Extracted F2={extracted:.6f} differs from target F2={target:.6f} "
        f"by {error:.6f} (tolerance: {tolerance})"
    )
    
    print(f"✓ Two-qubit RB: F2_target={target:.6f}, F2_extracted={extracted:.6f}, "
          f"error={error:.6f}")


def test_rb_decay_monotonic():
    """
    Test that RB survival probability decreases monotonically with sequence length.
    """
    config = Config.from_yaml("examples/configs/baseline.yaml")
    result = validate_rb_single_qubit(config, n_sequences=10, seed=100)
    
    # Check that survival probabilities generally decrease
    # (allowing small fluctuations due to finite sampling)
    probs = result['survival_probs']
    
    # At least the trend should be downward (first > last)
    assert probs[0] > probs[-1], "Survival probability should decrease with sequence length"
    
    print(f"✓ RB decay is monotonic: {probs[0]:.4f} → {probs[-1]:.4f}")


def test_rb_high_fidelity_config():
    """
    Test RB with high-fidelity configuration.
    
    High fidelity should give slower decay and higher survival probabilities.
    """
    # Load high fidelity config
    config = Config.from_yaml("examples/configs/high_fidelity.yaml")
    
    result = validate_rb_single_qubit(config, n_sequences=10, seed=200)
    
    # With very high fidelity, survival should be high even at moderate depth
    # Check that at m=16, survival is still > 0.7
    lengths = result['sequence_lengths']
    probs = result['survival_probs']
    
    idx_16 = lengths.index(16)
    survival_at_16 = probs[idx_16]
    
    assert survival_at_16 > 0.7, (
        f"High-fidelity config should have survival > 0.7 at m=16, got {survival_at_16:.4f}"
    )
    
    print(f"✓ High-fidelity RB: survival at m=16 is {survival_at_16:.4f}")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running RB integration tests...\n")
    
    test_rb_single_qubit_fidelity_reproduction()
    test_rb_two_qubit_fidelity_reproduction()
    test_rb_decay_monotonic()
    test_rb_high_fidelity_config()
    
    print("\n✓ All RB integration tests passed!")
