"""
Campaign Configuration Generator

Generates systematic parameter sweeps for QV campaigns.
Supports multiple sweep types with intelligent parameter ranges.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import yaml
import copy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from spinq_qv.config.schemas import Config


def generate_parameter_sweep_configs(
    base_config_path: Path,
    sweep_type: str = "comprehensive",
    n_points: int = 5,
    output_dir: Path = Path("campaigns"),
) -> Tuple[Dict[str, Config], Dict[str, List[float]]]:
    """
    Generate configurations for parameter sweep campaign.
    
    Args:
        base_config_path: Path to base configuration file
        sweep_type: Type of sweep (comprehensive, fidelity_focus, coherence_focus, timing_focus, custom)
        n_points: Number of points per parameter
        output_dir: Directory to save individual config files
    
    Returns:
        Tuple of (campaign_configs dict, sweep_params dict)
    """
    # Load base config
    base_config = Config.from_yaml(base_config_path)
    
    # Define sweep ranges based on type
    if sweep_type == "comprehensive":
        sweep_params = _get_comprehensive_sweep(base_config, n_points)
    elif sweep_type == "fidelity_focus":
        sweep_params = _get_fidelity_sweep(base_config, n_points)
    elif sweep_type == "coherence_focus":
        sweep_params = _get_coherence_sweep(base_config, n_points)
    elif sweep_type == "timing_focus":
        sweep_params = _get_timing_sweep(base_config, n_points)
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    # Generate configurations
    campaign_configs = {}
    
    # Add baseline config
    config_name = "baseline"
    campaign_configs[config_name] = copy.deepcopy(base_config)
    campaign_configs[config_name].metadata["config_name"] = config_name
    
    # Generate sweep configs
    for param_name, param_values in sweep_params.items():
        for value in param_values:
            # Determine baseline depending on parameter
            if param_name == "meas_error":
                baseline_value = 0.5 * (
                    base_config.device.meas_error_1given0 + base_config.device.meas_error_0given1
                )
            elif param_name == "state_prep_error":
                baseline_value = base_config.device.state_prep_error
            else:
                baseline_value = getattr(base_config.device, param_name)

            # Skip if same as baseline
            if np.isclose(value, baseline_value):
                continue

            # Create config name (format depends on parameter type)
            config_name = f"{param_name}_{_format_value(value, param_name)}"

            # Clone base config
            new_config = copy.deepcopy(base_config)

            # Modify parameter (convert to Python float to avoid numpy serialization issues)
            if param_name == "meas_error":
                # Apply symmetric measurement errors to both POVM error rates
                new_config.device.meas_error_1given0 = float(value)
                new_config.device.meas_error_0given1 = float(value)
            elif param_name == "state_prep_error":
                new_config.device.state_prep_error = float(value)
            else:
                setattr(new_config.device, param_name, float(value))

            # Update metadata
            new_config.metadata["config_name"] = config_name
            new_config.metadata["sweep_param"] = param_name
            new_config.metadata["sweep_value"] = float(value)

            campaign_configs[config_name] = new_config
    
    # Save individual config files
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in campaign_configs.items():
        config_file = configs_dir / f"{name}.yaml"
        config.to_yaml(config_file)
    
    print(f"[✓] Saved {len(campaign_configs)} config files to {configs_dir}/")
    
    return campaign_configs, sweep_params


def _get_comprehensive_sweep(base_config: Config, n_points: int) -> Dict[str, List[float]]:
    """Generate comprehensive sweep covering all major parameters."""
    
    sweep_params = {}
    
    # Fidelity parameters (±0.5% around baseline)
    F1_baseline = base_config.device.F1
    sweep_params["F1"] = np.linspace(
        max(0.990, F1_baseline - 0.005),
        min(0.99999, F1_baseline + 0.005),
        n_points
    )
    
    F2_baseline = base_config.device.F2
    sweep_params["F2"] = np.linspace(
        max(0.985, F2_baseline - 0.005),
        min(0.9999, F2_baseline + 0.005),
        n_points
    )
    
    # Coherence times (log scale, ±50% around baseline)
    T1_baseline = base_config.device.T1
    sweep_params["T1"] = np.logspace(
        np.log10(T1_baseline * 0.5),
        np.log10(T1_baseline * 2.0),
        n_points
    )
    
    T2_baseline = base_config.device.T2
    sweep_params["T2"] = np.logspace(
        np.log10(max(10e-6, T2_baseline * 0.5)),
        np.log10(min(T1_baseline * 2, T2_baseline * 2.0)),  # Respect T2 <= 2*T1
        n_points
    )
    
    # Gate times (±50% around baseline)
    t_single_baseline = base_config.device.t_single_gate
    sweep_params["t_single_gate"] = np.linspace(
        t_single_baseline * 0.5,
        t_single_baseline * 1.5,
        n_points
    )
    
    t_two_baseline = base_config.device.t_two_gate
    sweep_params["t_two_gate"] = np.linspace(
        t_two_baseline * 0.5,
        t_two_baseline * 1.5,
        n_points
    )
    
    return sweep_params


def _get_fidelity_sweep(base_config: Config, n_points: int) -> Dict[str, List[float]]:
    """Generate sweep focusing on gate fidelities and SPAM."""
    
    sweep_params = {}
    
    # Fine-grained fidelity sweep
    F1_baseline = base_config.device.F1
    sweep_params["F1"] = np.linspace(
        max(0.990, F1_baseline - 0.001),
        min(0.99999, F1_baseline + 0.001),
        n_points * 2  # More points for fidelity
    )
    
    F2_baseline = base_config.device.F2
    sweep_params["F2"] = np.linspace(
        max(0.985, F2_baseline - 0.002),
        min(0.9999, F2_baseline + 0.002),
        n_points * 2
    )
    
    # SPAM parameters (mapped to new parameterization)
    # Readout fidelity (convert to symmetric measurement error rates)
    readout_fidelity_baseline = 1.0 - 0.5 * (
        base_config.device.meas_error_1given0 + base_config.device.meas_error_0given1
    )
    F_readout_range = np.linspace(
        max(0.990, readout_fidelity_baseline - 0.005),
        min(0.9999, readout_fidelity_baseline + 0.001),
        n_points
    )
    # Convert fidelities → symmetric measurement error probabilities (meas_error)
    sweep_params["meas_error"] = list((1.0 - F_readout_range) / 2.0)

    # Preparation fidelity (map to state_prep_error)
    F_init_baseline = 1.0 - base_config.device.state_prep_error
    F_init_range = np.linspace(
        max(0.980, F_init_baseline - 0.01),
        min(0.999, F_init_baseline + 0.005),
        n_points
    )
    sweep_params["state_prep_error"] = list(1.0 - F_init_range)
    
    return sweep_params


def _get_coherence_sweep(base_config: Config, n_points: int) -> Dict[str, List[float]]:
    """Generate sweep focusing on coherence times."""
    
    sweep_params = {}
    
    # Wide-range T1 sweep (log scale)
    T1_baseline = base_config.device.T1
    sweep_params["T1"] = np.logspace(
        np.log10(T1_baseline * 0.2),
        np.log10(T1_baseline * 5.0),
        n_points * 2
    )
    
    # Wide-range T2 sweep (log scale)
    T2_baseline = base_config.device.T2
    sweep_params["T2"] = np.logspace(
        np.log10(max(5e-6, T2_baseline * 0.2)),
        np.log10(min(T1_baseline * 2, T2_baseline * 5.0)),
        n_points * 2
    )
    
    # T2* sweep
    T2_star_baseline = base_config.device.T2_star
    sweep_params["T2_star"] = np.logspace(
        np.log10(max(1e-6, T2_star_baseline * 0.2)),
        np.log10(min(T2_baseline, T2_star_baseline * 5.0)),  # Respect T2* <= T2
        n_points
    )
    
    return sweep_params


def _get_timing_sweep(base_config: Config, n_points: int) -> Dict[str, List[float]]:
    """Generate sweep focusing on gate and readout timing."""
    
    sweep_params = {}
    
    # Single-qubit gate time
    t_single_baseline = base_config.device.t_single_gate
    sweep_params["t_single_gate"] = np.linspace(
        t_single_baseline * 0.3,  # Faster gates
        t_single_baseline * 2.0,   # Slower gates
        n_points * 2
    )
    
    # Two-qubit gate time
    t_two_baseline = base_config.device.t_two_gate
    sweep_params["t_two_gate"] = np.linspace(
        t_two_baseline * 0.3,
        t_two_baseline * 2.0,
        n_points * 2
    )
    
    # Readout time
    t_readout_baseline = base_config.device.t_readout
    sweep_params["t_readout"] = np.linspace(
        t_readout_baseline * 0.5,
        t_readout_baseline * 2.0,
        n_points
    )
    
    return sweep_params


def _format_value(value: float, param_name: str | None = None) -> str:
    """Format parameter value for config name.

    If param_name is provided, use it to distinguish times from probability-like
    parameters so names remain readable and unambiguous.
    """
    # Time-like parameters: param names prefixed with 't_'
    if param_name is not None and param_name.startswith("t_"):
        if value >= 1.0:
            return f"{value:.3f}s".replace(".", "p")
        if value >= 1e-3:
            return f"{value*1e3:.2f}ms".replace(".", "p")
        if value >= 1e-6:
            return f"{value*1e6:.1f}us".replace(".", "p")
        return f"{value*1e9:.0f}ns"

    # Probability-like parameters (errors): format as small decimal
    if param_name is not None and ("error" in param_name or "meas_" in param_name):
        return f"{value:.6f}".replace(".", "p")

    # Fallback heuristics
    if value >= 0.9:
        return f"{value:.5f}".replace(".", "p")
    if value >= 1e-3:
        return f"{value*1e3:.2f}ms".replace(".", "p")
    if value >= 1e-6:
        return f"{value*1e6:.1f}us".replace(".", "p")
    return f"{value:.6f}".replace(".", "p")


if __name__ == "__main__":
    # Test generation
    from pathlib import Path
    
    base_config = Path("examples/configs/production.yaml")
    output_dir = Path("test_campaign_configs")
    
    print("Testing configuration generation...")
    
    for sweep_type in ["comprehensive", "fidelity_focus", "coherence_focus", "timing_focus"]:
        print(f"\n[{sweep_type}]")
        configs, params = generate_parameter_sweep_configs(
            base_config,
            sweep_type=sweep_type,
            n_points=3,
            output_dir=output_dir / sweep_type,
        )
        print(f"  Generated {len(configs)} configurations")
        print(f"  Parameters: {list(params.keys())}")
