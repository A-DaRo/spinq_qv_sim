"""
Unit tests for configuration validation.

Tests that invalid configuration values are properly rejected and
that validation constraints are correctly enforced.
"""

import pytest
from pydantic import ValidationError

from spinq_qv.config import DeviceConfig, SimulationConfig, Config


class TestDeviceConfigValidation:
    """Tests for DeviceConfig validation."""
    
    def test_fidelity_must_be_in_range(self):
        """Test that fidelities must be in [0, 1]."""
        # Valid fidelities
        config = DeviceConfig(F1=0.9, F2=0.95)
        assert config.F1 == 0.9
        assert config.F2 == 0.95
        
        # F1 > 1
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            DeviceConfig(F1=1.5)
        
        # F1 < 0
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DeviceConfig(F1=-0.1)
        
        # F2 > 1
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            DeviceConfig(F2=1.2)
        
        # F2 < 0
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DeviceConfig(F2=-0.05)
    
    def test_t2_cannot_exceed_twice_t1(self):
        """Test T2 <= 2*T1 constraint."""
        # Valid: T2 < 2*T1
        config = DeviceConfig(T1=1.0, T2=1.5)
        assert config.T2 == 1.5
        
        # Valid: T2 = 2*T1 (boundary)
        config = DeviceConfig(T1=1.0, T2=2.0)
        assert config.T2 == 2.0
        
        # Invalid: T2 > 2*T1
        with pytest.raises(ValidationError, match="T2.*cannot exceed"):
            DeviceConfig(T1=1.0, T2=2.5)
    
    def test_t2_star_cannot_exceed_t2(self):
        """Test T2* <= T2 constraint."""
        # Valid: T2* < T2
        config = DeviceConfig(T2=100e-6, T2_star=20e-6)
        assert config.T2_star == 20e-6
        
        # Valid: T2* = T2 (boundary)
        config = DeviceConfig(T2=100e-6, T2_star=100e-6)
        assert config.T2_star == 100e-6
        
        # Invalid: T2* > T2
        with pytest.raises(ValidationError, match="T2\\*.*cannot exceed"):
            DeviceConfig(T2=100e-6, T2_star=200e-6)
    
    def test_coherence_times_must_be_positive(self):
        """Test that all time parameters must be positive."""
        # T1 must be positive
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(T1=0)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(T1=-1.0)
        
        # T2 must be positive
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(T2=0)
        
        # T2* must be positive
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(T2_star=0)
    
    def test_gate_times_must_be_positive(self):
        """Test that gate duration parameters must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(t_single_gate=0)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(t_two_gate=-1e-9)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            DeviceConfig(t_readout=0)
    
    def test_spam_errors_in_range(self):
        """Test SPAM error parameters must be in [0, 1]."""
        # Valid
        config = DeviceConfig(
            state_prep_error=0.01,
            meas_error_1given0=0.005,
            meas_error_0given1=0.008
        )
        assert config.state_prep_error == 0.01
        assert config.meas_error_1given0 == 0.005
        assert config.meas_error_0given1 == 0.008
        
        # Invalid state_prep_error
        with pytest.raises(ValidationError):
            DeviceConfig(state_prep_error=1.5)
        
        # Invalid meas_error_1given0
        with pytest.raises(ValidationError):
            DeviceConfig(meas_error_1given0=-0.1)
    
    def test_default_values_are_valid(self):
        """Test that default configuration is physically valid."""
        config = DeviceConfig()
        
        # Check fidelities
        assert 0 <= config.F1 <= 1
        assert 0 <= config.F2 <= 1
        
        # Check SPAM errors
        assert 0 <= config.state_prep_error <= 1
        assert 0 <= config.meas_error_1given0 <= 1
        assert 0 <= config.meas_error_0given1 <= 1
        
        # Check coherence constraints
        assert config.T2 <= 2 * config.T1
        assert config.T2_star <= config.T2
        
        # Check positivity
        assert config.T1 > 0
        assert config.T2 > 0
        assert config.T2_star > 0
        assert config.t_single_gate > 0
        assert config.t_two_gate > 0
        assert config.t_readout > 0


class TestSimulationConfigValidation:
    """Tests for SimulationConfig validation."""
    
    def test_backend_must_be_valid(self):
        """Test that backend must be one of the supported types."""
        valid_backends = ["statevector", "density_matrix", "mcwf", "tensornet"]
        
        # Valid backends
        for backend in valid_backends:
            config = SimulationConfig(backend=backend)
            assert config.backend == backend
        
        # Invalid backend
        with pytest.raises(ValidationError, match="Backend must be one of"):
            SimulationConfig(backend="invalid_simulator")
        
        with pytest.raises(ValidationError, match="Backend must be one of"):
            SimulationConfig(backend="qiskit")
    
    def test_n_circuits_must_be_positive(self):
        """Test that number of circuits must be >= 1."""
        # Valid
        config = SimulationConfig(n_circuits=50)
        assert config.n_circuits == 50
        
        config = SimulationConfig(n_circuits=1)
        assert config.n_circuits == 1
        
        # Invalid
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SimulationConfig(n_circuits=0)
        
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SimulationConfig(n_circuits=-10)
    
    def test_n_shots_must_be_positive(self):
        """Test that number of shots must be >= 1."""
        # Valid
        config = SimulationConfig(n_shots=1000)
        assert config.n_shots == 1000
        
        config = SimulationConfig(n_shots=1)
        assert config.n_shots == 1
        
        # Invalid
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SimulationConfig(n_shots=0)
        
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SimulationConfig(n_shots=-5)
    
    def test_widths_must_be_positive(self):
        """Test that all widths must be positive integers."""
        # Valid
        config = SimulationConfig(widths=[1, 2, 3, 4, 5])
        assert config.widths == [1, 2, 3, 4, 5]
        
        # Invalid: contains zero
        with pytest.raises(ValidationError, match="must be positive"):
            SimulationConfig(widths=[1, 2, 0, 4])
        
        # Invalid: contains negative
        with pytest.raises(ValidationError, match="must be positive"):
            SimulationConfig(widths=[2, 3, -1])
    
    def test_widths_are_sorted_on_validation(self):
        """Test that widths are automatically sorted."""
        config = SimulationConfig(widths=[5, 2, 8, 3, 1])
        assert config.widths == [1, 2, 3, 5, 8]
    
    def test_random_seed_can_be_none(self):
        """Test that random seed can be None (for non-deterministic runs)."""
        config = SimulationConfig(random_seed=None)
        assert config.random_seed is None
        
        config = SimulationConfig(random_seed=42)
        assert config.random_seed == 42


class TestFullConfigValidation:
    """Tests for complete Config validation."""
    
    def test_nested_validation_works(self):
        """Test that validation cascades through nested configs."""
        # Invalid device config should fail
        with pytest.raises(ValidationError):
            Config(
                device={"F1": 1.5},  # Invalid fidelity
                simulation={"backend": "statevector"}
            )
        
        # Invalid simulation config should fail
        with pytest.raises(ValidationError):
            Config(
                device={"F1": 0.999},
                simulation={"backend": "invalid_backend"}
            )
    
    def test_metadata_is_flexible(self):
        """Test that metadata accepts arbitrary key-value pairs."""
        config = Config(metadata={
            "experiment_name": "test_run",
            "notes": "Testing metadata",
            "custom_field": 123,
            "nested": {"key": "value"}
        })
        
        assert config.metadata["experiment_name"] == "test_run"
        assert config.metadata["custom_field"] == 123
        assert config.metadata["nested"]["key"] == "value"
    
    def test_default_config_is_complete(self):
        """Test that default config has all required fields."""
        config = Config()
        
        # Device config exists
        assert config.device is not None
        assert hasattr(config.device, "F1")
        assert hasattr(config.device, "T1")
        
        # Simulation config exists
        assert config.simulation is not None
        assert hasattr(config.simulation, "backend")
        assert hasattr(config.simulation, "n_circuits")
        
        # Metadata exists (even if empty)
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)
    
    def test_partial_config_uses_defaults(self):
        """Test that partial configs fill in defaults."""
        config = Config(
            device={"F1": 0.995},  # Only override F1
            simulation={"n_circuits": 100}  # Only override n_circuits
        )
        
        # Overridden values
        assert config.device.F1 == 0.995
        assert config.simulation.n_circuits == 100
        
        # Default values still present
        assert config.device.F2 == 0.998  # default
        assert config.simulation.backend == "statevector"  # default


class TestConfigSerialization:
    """Tests for config YAML serialization."""
    
    def test_yaml_roundtrip_preserves_values(self, tmp_path):
        """Test that save -> load preserves all values."""
        original = Config(
            device={"F1": 0.995, "T1": 2.0},
            simulation={"n_circuits": 100, "widths": [3, 5, 7]},
            metadata={"test": "value"}
        )
        
        yaml_path = tmp_path / "test_config.yaml"
        original.to_yaml(yaml_path)
        
        loaded = Config.from_yaml(yaml_path)
        
        assert loaded.device.F1 == original.device.F1
        assert loaded.device.T1 == original.device.T1
        assert loaded.simulation.n_circuits == original.simulation.n_circuits
        assert loaded.simulation.widths == original.simulation.widths
        assert loaded.metadata["test"] == original.metadata["test"]
    
    def test_invalid_yaml_raises_validation_error(self, tmp_path):
        """Test that loading invalid YAML raises ValidationError."""
        yaml_path = tmp_path / "invalid.yaml"
        
        # Write invalid config
        yaml_path.write_text("""
device:
  F1: 1.5  # Invalid: > 1
simulation:
  backend: statevector
""")
        
        with pytest.raises(ValidationError):
            Config.from_yaml(yaml_path)
