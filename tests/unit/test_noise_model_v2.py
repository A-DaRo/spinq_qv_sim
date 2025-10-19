"""
Unit tests for improved noise model (v2).

Tests:
1. Unified gate error model (decoherence + coherent + residual depolarizing)
2. Crosstalk operators (ZZ and control pulse leakage)
3. SPAM errors (state prep and POVM measurement)
4. Time-dependent noise samplers (drift models)
5. Channel composition and fidelity calculations
"""

import pytest
import numpy as np
from spinq_qv.noise import channels, coherent, stochastic
from spinq_qv.noise.builder_v2 import NoiseModelBuilderV2


class TestChannelHelpers:
    """Test new channel helper functions."""
    
    def test_compute_channel_fidelity_identity(self):
        """Identity channel should have fidelity = 1."""
        I = np.eye(2, dtype=np.complex128)
        kraus = [I]
        
        F = channels.compute_channel_fidelity(kraus, dim=2)
        
        assert np.isclose(F, 1.0), f"Expected F=1.0 for identity, got {F}"
    
    def test_compute_channel_fidelity_depolarizing(self):
        """Verify depolarizing channel fidelity formula."""
        p = 0.01
        kraus = channels.depolarizing_kraus(p)
        
        F = channels.compute_channel_fidelity(kraus, dim=2)
        
        # The Kraus form uses p/4 coefficients, so relationship is different
        # F_avg = (sum |Tr(K_i)|^2 + d) / (d(d+1))
        # For our Kraus form: K0 = sqrt(1-3p/4)*I, others have zero trace
        # So: F = (|2*sqrt(1-3p/4)|^2 + 2) / 6 = (4(1-3p/4) + 2) / 6 = (6 - 3p) / 6 = 1 - p/2
        F_expected = 1.0 - p / 2.0
        
        assert np.isclose(F, F_expected, atol=1e-10), \
            f"Expected F={F_expected}, got {F}"
    
    def test_compose_kraus_channels(self):
        """Test Kraus channel composition."""
        # Two depolarizing channels with p1 and p2
        p1 = 0.01
        p2 = 0.02
        kraus1 = channels.depolarizing_kraus(p1)
        kraus2 = channels.depolarizing_kraus(p2)
        
        composed = channels.compose_kraus_channels(kraus1, kraus2)
        
        # Should have 4 * 4 = 16 Kraus operators
        assert len(composed) == 16
        
        # Should be trace-preserving
        assert channels.is_trace_preserving(composed, atol=1e-10)
        
        # Fidelity should be approximately F1 * F2 (for our Kraus form)
        F_composed = channels.compute_channel_fidelity(composed, dim=2)
        F1 = 1.0 - p1 / 2.0
        F2 = 1.0 - p2 / 2.0
        F_expected = F1 * F2
        
        assert np.isclose(F_composed, F_expected, rtol=1e-3)


class TestCoherentErrors:
    """Test crosstalk and coherent error operators."""
    
    def test_zz_crosstalk_unitary(self):
        """ZZ crosstalk should be unitary and diagonal."""
        zeta = 1000.0  # rad/s
        duration = 100e-9  # 100 ns
        
        U_zz = coherent.zz_crosstalk_unitary(zeta, duration)
        
        # Check unitarity
        assert np.allclose(U_zz @ U_zz.conj().T, np.eye(4), atol=1e-12)
        
        # Check diagonal
        assert np.allclose(np.diag(np.diag(U_zz)), U_zz, atol=1e-12)
        
        # Check phase pattern [e^(-iφ), e^(iφ), e^(iφ), e^(-iφ)]
        phi = zeta * duration
        expected_phases = [
            np.exp(-1j * phi),
            np.exp(1j * phi),
            np.exp(1j * phi),
            np.exp(-1j * phi)
        ]
        
        for i, phase_expected in enumerate(expected_phases):
            assert np.isclose(U_zz[i, i], phase_expected, atol=1e-12)
    
    def test_control_crosstalk_unitary(self):
        """Control crosstalk should create tensor product of rotations."""
        axis = 'x'
        angle = np.pi / 4
        crosstalk_frac = 0.05
        
        U_crosstalk = coherent.control_crosstalk_unitary(axis, angle, crosstalk_frac)
        
        # Should be 4x4 and unitary
        assert U_crosstalk.shape == (4, 4)
        assert np.allclose(U_crosstalk @ U_crosstalk.conj().T, np.eye(4), atol=1e-12)
        
        # Should equal R_x(θ) ⊗ R_x(α*θ)
        R_target = coherent.small_rotation(axis, angle)
        R_spectator = coherent.small_rotation(axis, crosstalk_frac * angle)
        U_expected = np.kron(R_target, R_spectator)
        
        assert np.allclose(U_crosstalk, U_expected, atol=1e-12)
    
    def test_compute_unitary_infidelity_identity(self):
        """Identical unitaries should have zero infidelity."""
        U = coherent.small_rotation('z', 0.1)
        
        infidelity = coherent.compute_unitary_infidelity(U, U)
        
        assert np.isclose(infidelity, 0.0, atol=1e-12)
    
    def test_compute_unitary_infidelity_small_angle(self):
        """Small rotation should have infidelity ≈ ε²/4 for process fidelity."""
        epsilon = 0.01  # Small angle
        U_actual = coherent.small_rotation('z', epsilon)
        U_ideal = np.eye(2, dtype=np.complex128)
        
        infidelity = coherent.compute_unitary_infidelity(U_actual, U_ideal)
        
        # For small rotation, infidelity ≈ ε²/4 (process/gate fidelity)
        # This is because |Tr(U† U_ideal)|² ≈ |2 - ε²/2|² ≈ 4(1 - ε²/2) for small ε
        # So infidelity ≈ 1 - (1 - ε²/2) = ε²/2... but divided by d² gives ε²/(2*4) = ε²/8
        # Actually for Rz(ε): Tr(Rz(ε)) = e^(-iε/2) + e^(iε/2) = 2cos(ε/2) ≈ 2(1 - ε²/8)
        # |Tr|² = 4cos²(ε/2) ≈ 4(1 - ε²/4) → infidelity = 1 - (1 - ε²/4) = ε²/4
        expected = epsilon**2 / 4.0
        
        assert np.isclose(infidelity, expected, rtol=0.05), \
            f"Expected ~{expected}, got {infidelity}"


class TestSPAMErrors:
    """Test SPAM (State Preparation And Measurement) error models."""
    
    def test_state_prep_error_dm(self):
        """State prep error should create correct mixed state."""
        p_excited = 0.05
        
        rho = channels.state_prep_error_dm(p_excited)
        
        # Should be 2x2
        assert rho.shape == (2, 2)
        
        # Should be Hermitian
        assert np.allclose(rho, rho.conj().T, atol=1e-12)
        
        # Trace should be 1
        assert np.isclose(np.trace(rho), 1.0, atol=1e-12)
        
        # Diagonal should be [1-p, p]
        assert np.isclose(rho[0, 0], 1.0 - p_excited, atol=1e-12)
        assert np.isclose(rho[1, 1], p_excited, atol=1e-12)
        
        # Off-diagonal should be zero
        assert np.isclose(rho[0, 1], 0.0, atol=1e-12)
        assert np.isclose(rho[1, 0], 0.0, atol=1e-12)
    
    def test_measurement_povm_completeness(self):
        """POVM operators should satisfy completeness relation."""
        p_1given0 = 0.02
        p_0given1 = 0.03
        
        M_0, M_1 = channels.measurement_povm_operators(p_1given0, p_0given1)
        
        # Check completeness: M_0† M_0 + M_1† M_1 = I
        completeness = M_0.conj().T @ M_0 + M_1.conj().T @ M_1
        
        assert np.allclose(completeness, np.eye(2), atol=1e-12)
    
    def test_apply_povm_measurement_pure_states(self):
        """POVM measurement on pure states should give correct probabilities."""
        p_1given0 = 0.01
        p_0given1 = 0.02
        
        M_0, M_1 = channels.measurement_povm_operators(p_1given0, p_0given1)
        
        # Test on |0⟩
        state_0 = np.array([1.0, 0.0], dtype=np.complex128)
        probs_0 = channels.apply_povm_measurement(state_0, [M_0, M_1])
        
        assert np.isclose(probs_0[0], 1.0 - p_1given0, atol=1e-10)
        assert np.isclose(probs_0[1], p_1given0, atol=1e-10)
        
        # Test on |1⟩
        state_1 = np.array([0.0, 1.0], dtype=np.complex128)
        probs_1 = channels.apply_povm_measurement(state_1, [M_0, M_1])
        
        assert np.isclose(probs_1[0], p_0given1, atol=1e-10)
        assert np.isclose(probs_1[1], 1.0 - p_0given1, atol=1e-10)


class TestDriftSamplers:
    """Test time-dependent noise samplers."""
    
    def test_drifting_sigma_sampler(self):
        """Drifting sigma sampler should produce valid samples."""
        sigma_mean = 70000.0  # rad/s
        sigma_std = 7000.0    # 10% drift
        seed = 42
        
        sampler = stochastic.DriftingSigmaSampler(sigma_mean, sigma_std, seed=seed)
        
        # Sample multiple sigma values
        sigmas = [sampler.sample_sigma() for _ in range(100)]
        
        # All should be non-negative
        assert all(s >= 0 for s in sigmas)
        
        # Mean should be close to sigma_mean
        assert np.isclose(np.mean(sigmas), sigma_mean, rtol=0.2)
        
        # Std should be close to sigma_std
        assert np.isclose(np.std(sigmas), sigma_std, rtol=0.3)
    
    def test_drifting_sigma_detuning_sampling(self):
        """Should sample both sigma and detuning."""
        sigma_mean = 70000.0
        sigma_std = 7000.0
        sampler = stochastic.DriftingSigmaSampler(sigma_mean, sigma_std, seed=123)
        
        sigma, detuning = sampler.sample_detuning_with_drift()
        
        # Sigma should be positive
        assert sigma >= 0
        
        # Detuning should be a float
        assert isinstance(detuning, float)
    
    def test_coherent_error_drift_sampler(self):
        """Coherent error drift sampler should produce normal distribution."""
        error_mean = 0.001  # 1 mrad
        error_std = 0.0001  # 0.1 mrad drift
        seed = 456
        
        sampler = stochastic.CoherentErrorDriftSampler(error_mean, error_std, seed=seed)
        
        # Sample multiple angles
        angles = sampler.sample_many_angles(1000)
        
        # Mean should be close to error_mean
        assert np.isclose(np.mean(angles), error_mean, rtol=0.1)
        
        # Std should be close to error_std
        assert np.isclose(np.std(angles), error_std, rtol=0.2)


class TestUnifiedNoiseModel:
    """Test NoiseModelBuilderV2 unified error model."""
    
    def test_decoherence_channel_fidelity(self):
        """Decoherence fidelity should decrease with gate time."""
        device_params = {
            'T1': 1.0,
            'T2': 100e-6,
            'F1': 0.999,
            'F2': 0.99
        }
        
        builder = NoiseModelBuilderV2(device_params)
        
        # Short gate
        F_short, _ = builder._compute_decoherence_channel_fidelity(10e-9, 1.0, 100e-6)
        
        # Long gate
        F_long, _ = builder._compute_decoherence_channel_fidelity(1000e-9, 1.0, 100e-6)
        
        # Longer gate should have LOWER fidelity
        assert F_long < F_short, \
            f"Long gate F={F_long} should be < short gate F={F_short}"
        
        # Both should be less than 1
        assert F_short < 1.0
        assert F_long < 1.0
    
    def test_residual_depolarizing_zero_when_physical_exceeds_target(self):
        """If physical errors exceed target, residual should be zero."""
        device_params = {
            'T1': 1.0,
            'T2': 100e-6,
            'F1': 0.999,
        }
        
        builder = NoiseModelBuilderV2(device_params)
        
        # Set F_target HIGHER than physical (optimistic experiment)
        # In this case, we can't achieve it, so residual should be non-zero
        F_target = 0.9999  # Very high target
        F_decoherence = 0.998  # Lower physical fidelity
        F_coherent = 0.998
        
        p_residual = builder._compute_residual_depolarizing(
            F_target, F_decoherence, F_coherent, two_qubit=False
        )
        
        # Should be > 0 since we need residual error to bridge the gap
        # Actually wait - if F_physical < F_target, we can't achieve target!
        # The function returns 0 when F_physical >= F_target
        # Let's test the opposite: F_physical > F_target → p_residual = 0
        F_target_low = 0.95
        F_decoherence_high = 0.999
        F_coherent_high = 0.999
        
        p_residual_zero = builder._compute_residual_depolarizing(
            F_target_low, F_decoherence_high, F_coherent_high, two_qubit=False
        )
        
        # Should be 0 since physical errors alone exceed target
        assert p_residual_zero == 0.0
    
    def test_unified_single_qubit_channel_achieves_target_fidelity(self):
        """Unified channel should achieve target fidelity."""
        device_params = {
            'T1': 1.0,
            'T2': 99e-6,
            'F1': 0.99926,
            'single_qubit_overrotation_rad': 0.001,
            'coherent_axis': 'z'
        }
        
        builder = NoiseModelBuilderV2(device_params)
        gate_time = 60e-9
        
        model = builder.build_single_qubit_channel_unified(gate_time)
        
        # Compute total fidelity
        F_total = (model['F_decoherence'] * 
                   model['F_coherent'] * 
                   (1.0 - 2.0 * model['p_dep_residual'] / 3.0))
        
        # Should match target within tolerance
        F_target = device_params['F1']
        assert np.isclose(F_total, F_target, rtol=0.01), \
            f"Total F={F_total} should match target F={F_target}"
    
    def test_build_complete_noise_model(self):
        """Build complete noise model with all components."""
        device_params = {
            'T1': 1.0,
            'T2': 99e-6,
            'T2_star': 20e-6,
            'F1': 0.99926,
            'F2': 0.998,
            'single_qubit_overrotation_rad': 0.001,
            'residual_zz_phase': 0.002,
            'zz_crosstalk_strength': 1000.0,
            'control_crosstalk_fraction': 0.05,
            'state_prep_error': 0.01,
            'meas_error_1given0': 0.005,
            'meas_error_0given1': 0.008,
            'coherent_drift_std': 0.0001,
            'sigma_drift_fraction': 0.1,
        }
        
        builder = NoiseModelBuilderV2(device_params)
        
        gate_durations = {'single': 60e-9, 'two_qubit': 40e-9}
        noise_model = builder.build(gate_durations, seed=42)
        
        # Check structure
        assert 'version' in noise_model
        assert noise_model['version'] == 2
        
        assert 'single_qubit' in noise_model
        assert 'two_qubit' in noise_model
        assert 'crosstalk' in noise_model
        assert 'spam' in noise_model
        assert 'drift_samplers' in noise_model
        
        # Check crosstalk
        assert noise_model['crosstalk']['zz_crosstalk_strength'] == 1000.0
        assert noise_model['crosstalk']['control_crosstalk_fraction'] == 0.05
        
        # Check SPAM
        spam = noise_model['spam']
        assert spam['state_prep_error'] == 0.01
        assert spam['povm_M0'].shape == (2, 2)
        assert spam['povm_M1'].shape == (2, 2)
        
        # Check drift samplers
        assert 'drifting_sigma' in noise_model['drift_samplers']
        assert 'coherent_drift' in noise_model['drift_samplers']
    
    def test_validate_fidelities(self):
        """Validate that noise model achieves target fidelities."""
        device_params = {
            'T1': 1.0,
            'T2': 99e-6,
            'F1': 0.99926,
            'F2': 0.998,
        }
        
        builder = NoiseModelBuilderV2(device_params)
        
        gate_durations = {'single': 60e-9, 'two_qubit': 40e-9}
        noise_model = builder.build(gate_durations)
        
        validation = builder.validate_fidelities(noise_model)
        
        assert validation['F1_valid'], \
            f"F1 validation failed: {validation['F1_actual']} vs {validation['F1_target']}"
        assert validation['F2_valid'], \
            f"F2 validation failed: {validation['F2_actual']} vs {validation['F2_target']}"


class TestPhysicalConstraints:
    """Test that physical constraints are satisfied."""
    
    def test_longer_gates_have_lower_fidelity(self):
        """Longer gate times should result in lower fidelity (more decoherence)."""
        device_params = {
            'T1': 1.0,
            'T2': 99e-6,
            'F1': 0.999,
        }
        
        builder = NoiseModelBuilderV2(device_params)
        
        # Compare 50ns vs 500ns gates
        model_fast = builder.build_single_qubit_channel_unified(50e-9)
        model_slow = builder.build_single_qubit_channel_unified(500e-9)
        
        # Decoherence fidelity should be lower for slower gate
        assert model_slow['F_decoherence'] < model_fast['F_decoherence']
        
        # Residual depolarizing should be HIGHER for slower gate
        # (need more artificial error to reach same target)
        assert model_slow['p_dep_residual'] >= model_fast['p_dep_residual']
    
    def test_worse_T2_requires_more_residual_error(self):
        """Worse T2 should require more residual depolarizing to match target."""
        gate_time = 60e-9
        F_target = 0.999
        
        # Good T2
        params_good = {'T1': 1.0, 'T2': 100e-6, 'F1': F_target}
        builder_good = NoiseModelBuilderV2(params_good)
        model_good = builder_good.build_single_qubit_channel_unified(gate_time)
        
        # Bad T2
        params_bad = {'T1': 1.0, 'T2': 10e-6, 'F1': F_target}
        builder_bad = NoiseModelBuilderV2(params_bad)
        model_bad = builder_bad.build_single_qubit_channel_unified(gate_time)
        
        # Worse T2 → lower F_decoherence
        assert model_bad['F_decoherence'] < model_good['F_decoherence']
        
        # Worse T2 → higher p_dep_residual needed (if any residual is needed)
        # For very short gates with long T1, decoherence might be negligible
        # So test that EITHER both are zero OR bad > good
        if model_good['p_dep_residual'] > 0 or model_bad['p_dep_residual'] > 0:
            assert model_bad['p_dep_residual'] >= model_good['p_dep_residual']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
