"""
Smoke tests for repository structure and basic functionality.

Tests that package imports work, configuration loads correctly, and basic
infrastructure is in place.
"""

import pytest
from pathlib import Path


def test_package_import():
    """Test that spinq_qv package can be imported."""
    import spinq_qv
    assert spinq_qv.__version__ == "0.1.0"
    assert hasattr(spinq_qv, "config")
    assert hasattr(spinq_qv, "utils")


def test_config_subpackage_import():
    """Test that config subpackage imports work."""
    from spinq_qv.config import DeviceConfig, SimulationConfig, Config
    
    # Check that classes are importable
    assert DeviceConfig is not None
    assert SimulationConfig is not None
    assert Config is not None


def test_utils_subpackage_import():
    """Test that utils subpackage imports work."""
    from spinq_qv.utils import setup_logger, get_logger
    
    # Check that functions are importable
    assert setup_logger is not None
    assert get_logger is not None


def test_default_device_config():
    """Test that DeviceConfig can be instantiated with defaults."""
    from spinq_qv.config import DeviceConfig
    
    config = DeviceConfig()
    
    # Check default values match experimental parameters
    assert config.F1 == 0.99926
    assert config.F2 == 0.998
    assert config.T1 == 1.0
    assert config.T2 == 99e-6
    assert config.T2_star == 20e-6
    assert config.t_single_gate == 60e-9
    assert config.t_two_gate == 40e-9
    # Legacy SPAM fields removed; check new SPAM error parameters instead
    assert pytest.approx(config.state_prep_error, rel=1e-12) == 0.006
    assert pytest.approx(config.meas_error_1given0, rel=1e-12) == 0.00015
    assert pytest.approx(config.meas_error_0given1, rel=1e-12) == 0.00015


def test_default_simulation_config():
    """Test that SimulationConfig can be instantiated with defaults."""
    from spinq_qv.config import SimulationConfig
    
    config = SimulationConfig()
    
    # Check default values
    assert config.backend == "statevector"
    assert config.n_circuits == 50
    assert config.n_shots == 1000
    assert config.widths == [2, 3, 4, 5, 6]
    assert config.random_seed == 42


def test_full_config_creation():
    """Test that full Config can be instantiated."""
    from spinq_qv.config import Config
    
    config = Config()
    
    # Check that device and simulation configs exist
    assert config.device is not None
    assert config.simulation is not None
    assert isinstance(config.metadata, dict)


def test_config_validation_fidelity_bounds():
    """Test that fidelity values are validated to be in [0, 1]."""
    from spinq_qv.config import DeviceConfig
    import pydantic
    
    # Valid fidelity
    config = DeviceConfig(F1=0.999)
    assert config.F1 == 0.999
    
    # Invalid fidelity (> 1)
    with pytest.raises(pydantic.ValidationError):
        DeviceConfig(F1=1.5)
    
    # Invalid fidelity (< 0)
    with pytest.raises(pydantic.ValidationError):
        DeviceConfig(F1=-0.1)


def test_config_validation_t2_constraint():
    """Test that T2 <= 2*T1 constraint is enforced."""
    from spinq_qv.config import DeviceConfig
    import pydantic
    
    # Valid: T2 < 2*T1
    config = DeviceConfig(T1=1.0, T2=1.5)
    assert config.T2 == 1.5
    
    # Invalid: T2 > 2*T1
    with pytest.raises(pydantic.ValidationError, match="T2.*cannot exceed"):
        DeviceConfig(T1=1.0, T2=3.0)


def test_config_validation_t2_star_constraint():
    """Test that T2* <= T2 constraint is enforced."""
    from spinq_qv.config import DeviceConfig
    import pydantic
    
    # Valid: T2* < T2
    config = DeviceConfig(T2=99e-6, T2_star=20e-6)
    assert config.T2_star == 20e-6
    
    # Invalid: T2* > T2
    with pytest.raises(pydantic.ValidationError, match="T2\\*.*cannot exceed"):
        DeviceConfig(T2=99e-6, T2_star=200e-6)


def test_config_validation_backend():
    """Test that backend validation works."""
    from spinq_qv.config import SimulationConfig
    import pydantic
    
    # Valid backends
    for backend in ["statevector", "density_matrix", "mcwf", "tensornet"]:
        config = SimulationConfig(backend=backend)
        assert config.backend == backend
    
    # Invalid backend
    with pytest.raises(pydantic.ValidationError, match="Backend must be one of"):
        SimulationConfig(backend="invalid_backend")


def test_config_validation_positive_widths():
    """Test that widths must be positive integers."""
    from spinq_qv.config import SimulationConfig
    import pydantic
    
    # Valid widths
    config = SimulationConfig(widths=[1, 2, 3, 5, 8])
    assert config.widths == [1, 2, 3, 5, 8]
    
    # Invalid widths (non-positive)
    with pytest.raises(pydantic.ValidationError, match="must be positive"):
        SimulationConfig(widths=[1, 2, 0, 3])


def test_defaults_yaml_exists():
    """Test that defaults.yaml exists in the config directory."""
    import spinq_qv.config
    config_dir = Path(spinq_qv.config.__file__).parent
    defaults_path = config_dir / "defaults.yaml"
    
    assert defaults_path.exists(), "defaults.yaml not found in config directory"


def test_load_defaults_yaml():
    """Test that defaults.yaml can be loaded and validated."""
    from spinq_qv.config import Config
    import spinq_qv.config
    
    config_dir = Path(spinq_qv.config.__file__).parent
    defaults_path = config_dir / "defaults.yaml"
    
    # Load and validate
    config = Config.from_yaml(defaults_path)
    
    # Check that loaded values match expected defaults
    assert config.device.F1 == 0.99926
    assert config.device.F2 == 0.998
    assert config.simulation.backend == "statevector"
    assert config.simulation.n_circuits == 50
    # Verify SPAM defaults loaded correctly
    assert pytest.approx(config.device.state_prep_error, rel=1e-12) == 0.006
    assert pytest.approx(config.device.meas_error_1given0, rel=1e-12) == 0.00015
    assert pytest.approx(config.device.meas_error_0given1, rel=1e-12) == 0.00015


def test_logger_creation():
    """Test that logger can be created and configured."""
    from spinq_qv.utils import setup_logger, get_logger
    import logging
    
    # Create logger
    logger = setup_logger("test_logger", level=logging.INFO)
    assert logger is not None
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    
    # Get existing logger
    logger2 = get_logger("test_logger")
    assert logger2.name == "test_logger"
    assert logger is logger2  # Should be the same instance


def test_config_to_yaml_roundtrip(tmp_path):
    """Test that config can be saved and loaded via YAML."""
    from spinq_qv.config import Config
    
    # Create config
    config1 = Config()
    config1.metadata["test_key"] = "test_value"
    
    # Save to YAML
    yaml_path = tmp_path / "test_config.yaml"
    config1.to_yaml(yaml_path)
    assert yaml_path.exists()
    
    # Load from YAML
    config2 = Config.from_yaml(yaml_path)
    
    # Check that values match
    assert config2.device.F1 == config1.device.F1
    assert config2.device.F2 == config1.device.F2
    assert config2.simulation.backend == config1.simulation.backend
    assert config2.metadata["test_key"] == "test_value"
