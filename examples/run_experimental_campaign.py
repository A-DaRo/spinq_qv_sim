"""
Experimental Parameter Campaign Runner

This script runs a comprehensive campaign using Latin Hypercube Sampling (LHS)
across all parameter ranges reported in experimental literature (experiments.yaml).

Usage:
    python run_experimental_campaign.py --n-samples 50 --output campaigns/experimental_sweep
"""

import argparse
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinq_qv.config.schemas import Config, DeviceConfig
from campaign_executor import CampaignExecutor
from campaign_plotter import create_all_campaign_plots


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1]^n_dims.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions (parameters)
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    
    # Generate LHS samples
    samples = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        # Divide [0, 1] into n_samples bins
        bins = np.linspace(0, 1, n_samples + 1)
        
        # Sample uniformly within each bin
        samples[:, dim] = rng.uniform(bins[:-1], bins[1:])
        
        # Randomly permute to break correlations
        rng.shuffle(samples[:, dim])
    
    return samples


def load_experimental_ranges(yaml_path: Path) -> Dict[str, Optional[List[float]]]:
    """
    Load parameter ranges from experiments.yaml.
    
    Returns:
        Dictionary mapping parameter names to [min, max] ranges or None
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    device_params = data.get('device', {})
    
    # Extract ranges (filter out None/null values)
    ranges = {}
    for param, value in device_params.items():
        if value is None:
            ranges[param] = None
        elif isinstance(value, list) and len(value) == 2:
            ranges[param] = [float(value[0]), float(value[1])]
        else:
            # Should not happen with reformatted experiments.yaml
            ranges[param] = None
    
    return ranges


def generate_lhs_configs(
    base_config_path: Path,
    experimental_ranges: Dict[str, Optional[List[float]]],
    n_samples: int,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[str, Config], Dict[str, List[float]]]:
    """
    Generate campaign configurations using Latin Hypercube Sampling.
    
    Args:
        base_config_path: Path to baseline configuration
        experimental_ranges: Parameter ranges from experiments.yaml
        n_samples: Number of LHS samples to generate
        output_dir: Directory to save generated configs
        seed: Random seed for LHS
        
    Returns:
        Tuple of (campaign_configs, sweep_params)
    """
    # Load base config
    base_config = Config.from_yaml(base_config_path)
    
    # Separate parameters to sweep vs fixed
    sweep_params = {k: v for k, v in experimental_ranges.items() if v is not None and v[0] != v[1]}
    fixed_params = {k: v for k, v in experimental_ranges.items() if v is not None and v[0] == v[1]}
    null_params = [k for k, v in experimental_ranges.items() if v is None]
    
    print(f"Parameters to sweep: {len(sweep_params)}")
    print(f"Fixed parameters: {len(fixed_params)}")
    print(f"Null parameters (using defaults): {len(null_params)}")
    print()
    
    # Get parameter names and bounds
    param_names = list(sweep_params.keys())
    param_bounds = np.array([sweep_params[k] for k in param_names])
    
    n_dims = len(param_names)
    
    if n_dims == 0:
        raise ValueError("No parameters to sweep! All parameters are fixed or null.")
    
    # Generate LHS samples
    print(f"Generating {n_samples} LHS samples in {n_dims}-dimensional space...")
    lhs_samples = latin_hypercube_sample(n_samples, n_dims, seed=seed)
    
    # Scale samples to parameter bounds
    scaled_samples = param_bounds[:, 0] + lhs_samples * (param_bounds[:, 1] - param_bounds[:, 0])
    
    # Create configs
    campaign_configs = {}
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(scaled_samples):
        config = base_config.model_copy(deep=True)
        config_name = f"lhs_sample_{i:04d}"
        
        # Set swept parameters
        for param_name, param_value in zip(param_names, sample):
            setattr(config.device, param_name, float(param_value))
        
        # Set fixed parameters
        for param_name, param_range in fixed_params.items():
            setattr(config.device, param_name, float(param_range[0]))
        
        # Null parameters retain defaults from base_config
        
        # Save config
        config_path = configs_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        campaign_configs[config_name] = config
    
    print(f"[✓] Generated {len(campaign_configs)} LHS configurations")
    
    return campaign_configs, sweep_params


def create_campaign_manifest(
    campaign_configs: Dict[str, Config],
    output_dir: Path,
    sweep_params: Dict[str, List[float]],
    experimental_ranges: Dict[str, Optional[List[float]]],
    base_config_path: Path,
    n_samples: int,
    seed: int,
) -> Path:
    """Create a manifest file documenting the experimental campaign."""
    
    # Separate parameter types
    fixed_params = {k: v for k, v in experimental_ranges.items() 
                    if v is not None and v[0] == v[1]}
    null_params = [k for k, v in experimental_ranges.items() if v is None]
    
    manifest = {
        "campaign_name": output_dir.name,
        "campaign_type": "experimental_lhs",
        "timestamp": datetime.now().isoformat(),
        "base_config": str(base_config_path),
        "experimental_ranges_source": "examples/configs/experiments.yaml",
        "sampling_method": "Latin Hypercube Sampling (LHS)",
        "n_samples": n_samples,
        "random_seed": seed,
        "n_configurations": len(campaign_configs),
        "swept_parameters": {
            param: {
                "min": float(values[0]),
                "max": float(values[1]),
                "range": float(values[1] - values[0]),
            }
            for param, values in sweep_params.items()
        },
        "fixed_parameters": {
            param: float(values[0])
            for param, values in fixed_params.items()
        },
        "null_parameters": null_params,
        "parameter_sources": {
            "F1": "Papers [4], [11], [1] - range: 0.994 to 0.9996",
            "F2": "Papers [1], [9], [3] - range: 0.92 to 0.9981",
            "T1": "Papers [5], [6] - range: 10μs to 1s",
            "T2": "Papers [1], [4], [7] - range: 245ns to 3.1ms",
            "T2_star": "Papers [4], [1], [2] - range: 360ns to 20μs",
            "t_single_gate": "Papers [1], [4] - range: 60-70ns",
            "t_two_gate": "Paper [1] - fixed: 40ns",
            "zz_crosstalk_strength": "Paper [11] - range: 0 to 40kHz",
            "control_crosstalk_fraction": "Paper [3] - range: 0.44% to 1.85%",
            "state_prep_error": "Paper [1] - range: 0.6% to 2.5%",
            "meas_error_0given1": "Paper [1] - range: 0.2% to 1.9%",
        }
    }
    
    manifest_path = output_dir / "campaign_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[✓] Campaign manifest saved to: {manifest_path}")
    return manifest_path


def summarize_campaign_setup(
    campaign_configs: Dict[str, Config],
    sweep_params: Dict[str, List[float]],
    experimental_ranges: Dict[str, Optional[List[float]]],
    n_samples: int,
) -> None:
    """Print summary of campaign configuration."""
    print_banner("EXPERIMENTAL CAMPAIGN SETUP")
    
    print(f"Sampling Method: Latin Hypercube Sampling (LHS)")
    print(f"Total Configurations: {n_samples}\n")
    
    # Swept parameters
    print("Swept Parameters (LHS):")
    print("-" * 80)
    for param, values in sweep_params.items():
        range_val = values[1] - values[0]
        print(f"  {param:30s}: [{values[0]:.6g}, {values[1]:.6g}]  "
              f"(range: {range_val:.6g})")
    
    # Fixed parameters
    fixed_params = {k: v for k, v in experimental_ranges.items() 
                    if v is not None and v[0] == v[1]}
    if fixed_params:
        print("\nFixed Parameters:")
        print("-" * 80)
        for param, values in fixed_params.items():
            print(f"  {param:30s}: {values[0]:.6g}")
    
    # Null parameters
    null_params = [k for k, v in experimental_ranges.items() if v is None]
    if null_params:
        print("\nNull Parameters (using baseline defaults):")
        print("-" * 80)
        for param in null_params:
            print(f"  {param}")
    
    print("\n" + "-" * 80)
    
    # Estimate runtime
    base_config = list(campaign_configs.values())[0]
    n_widths = len(base_config.simulation.widths)
    n_circuits = base_config.simulation.n_circuits
    n_shots = base_config.simulation.n_shots
    
    est_time_per_config = 0.5 * n_widths * n_circuits  # Rough estimate
    total_est_time = est_time_per_config * len(campaign_configs)
    
    print(f"\nSimulation Settings (per configuration):")
    print(f"  Widths: {n_widths} ({min(base_config.simulation.widths)}-{max(base_config.simulation.widths)})")
    print(f"  Circuits per width: {n_circuits}")
    print(f"  Shots per circuit: {n_shots}")
    print(f"\nEstimated Runtime:")
    print(f"  Per configuration: ~{est_time_per_config/60:.1f} minutes")
    print(f"  Total campaign: ~{total_est_time/60:.1f} minutes ({total_est_time/3600:.1f} hours)")
    print("\n" + "=" * 80)


def print_progress(current: int, total: int, config_name: str, elapsed: float) -> None:
    """Print progress bar with timing info."""
    percent = 100 * current / total
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    avg_time = elapsed / current if current > 0 else 0
    eta = avg_time * (total - current)
    
    print(f"\r[{bar}] {percent:5.1f}% | {current}/{total} | "
          f"Current: {config_name[:30]:30s} | "
          f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m", 
          end="", flush=True)


def main():
    """Run experimental parameter campaign with LHS."""
    parser = argparse.ArgumentParser(
        description="Run experimental parameter campaign using Latin Hypercube Sampling"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("examples/configs/production.yaml"),
        help="Base configuration file (default: production.yaml)"
    )
    parser.add_argument(
        "--experimental-ranges",
        type=Path,
        default=Path("examples/configs/experiments.yaml"),
        help="Experimental parameter ranges (default: experiments.yaml)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("campaigns") / f"experimental_lhs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for campaign results"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of LHS samples (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS (default: 42)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution (experimental)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and manifest without running simulations"
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner("EXPERIMENTAL PARAMETER CAMPAIGN (LHS)", "═")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Config: {args.base_config}")
    print(f"Experimental Ranges: {args.experimental_ranges}")
    print(f"Output Dir: {args.output}")
    print(f"LHS Samples: {args.n_samples}")
    print(f"Random Seed: {args.seed}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load experimental ranges
    print_banner("LOADING EXPERIMENTAL PARAMETER RANGES", "-")
    experimental_ranges = load_experimental_ranges(args.experimental_ranges)
    
    print(f"[✓] Loaded {len(experimental_ranges)} parameters from {args.experimental_ranges}")
    
    # Generate LHS configurations
    print_banner("GENERATING LHS CONFIGURATIONS", "-")
    
    campaign_configs, sweep_params = generate_lhs_configs(
        base_config_path=args.base_config,
        experimental_ranges=experimental_ranges,
        n_samples=args.n_samples,
        output_dir=args.output,
        seed=args.seed,
    )
    
    # Create manifest
    manifest_path = create_campaign_manifest(
        campaign_configs,
        args.output,
        sweep_params,
        experimental_ranges,
        args.base_config,
        args.n_samples,
        args.seed,
    )
    
    # Summarize campaign
    summarize_campaign_setup(campaign_configs, sweep_params, experimental_ranges, args.n_samples)
    
    # Dry run check
    if args.dry_run:
        print_banner("DRY RUN COMPLETE", "=")
        print(f"Configurations saved to: {args.output}/configs/")
        print(f"Manifest: {manifest_path}")
        print("\nRun without --dry-run to execute campaign.")
        return
    
    # Confirmation
    print("\n" + "=" * 80)
    response = input("Proceed with campaign execution? [y/N]: ")
    if response.lower() != 'y':
        print("Campaign cancelled.")
        return
    
    # Execute campaign
    print_banner("EXECUTING CAMPAIGN", "=")
    
    executor = CampaignExecutor(
        campaign_configs=campaign_configs,
        output_dir=args.output,
        parallel=args.parallel,
    )
    
    start_time = time.time()
    
    try:
        campaign_results = executor.run_campaign(
            progress_callback=print_progress
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n")  # New line after progress bar
        print_banner("CAMPAIGN EXECUTION COMPLETE", "=")
        print(f"Total Time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
        print(f"Configurations Run: {len(campaign_results)}")
        print(f"Results Directory: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n[!] Campaign interrupted by user")
        print(f"Partial results saved to: {args.output}")
        return
    except Exception as e:
        print(f"\n\n[ERROR] Campaign failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate plots
    print_banner("GENERATING PLOTS", "=")
    
    try:
        create_all_campaign_plots(
            campaign_results=campaign_results,
            output_dir=args.output,
            threshold=2.0 / 3.0,
        )
    except Exception as e:
        print(f"[WARNING] Failed to generate some plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate campaign analysis
    print_banner("GENERATING CAMPAIGN ANALYSIS", "=")
    
    from campaign_analyzer import CampaignAnalyzer
    
    analyzer = CampaignAnalyzer(
        campaign_results=campaign_results,
        campaign_configs=campaign_configs,
        sweep_params=sweep_params,
        output_dir=args.output,
    )
    
    try:
        analyzer.generate_all_analyses()
    except Exception as e:
        print(f"[WARNING] Failed to generate some analyses: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print_banner("CAMPAIGN COMPLETE", "═")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {elapsed_time/60:.1f} minutes")
    print(f"\nResults Location: {args.output}/")
    print(f"  • Raw Results: {args.output}/results/")
    print(f"  • Configurations: {args.output}/configs/")
    print(f"  • Plots: {args.output}/plots/")
    print(f"  • Analysis Report: {args.output}/campaign_report.html")
    print(f"  • Manifest: {manifest_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
