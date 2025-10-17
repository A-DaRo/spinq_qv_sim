"""
Parameter Campaign Runner for Quantum Volume Analysis

This script runs a comprehensive campaign by systematically varying device parameters
to understand their impact on Quantum Volume performance.

Usage:
    python run_parameter_campaign.py --base-config configs/production.yaml --output campaigns/param_sweep
"""

import argparse
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinq_qv.config.schemas import Config
from campaign_executor import CampaignExecutor
from campaign_config_generator import generate_parameter_sweep_configs
from campaign_plotter import create_all_campaign_plots


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


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


def create_campaign_manifest(
    campaign_configs: Dict[str, Config],
    output_dir: Path,
    sweep_params: Dict[str, List[float]],
    base_config_path: Path,
) -> Path:
    """Create a manifest file documenting the campaign."""
    manifest = {
        "campaign_name": output_dir.name,
        "timestamp": datetime.now().isoformat(),
        "base_config": str(base_config_path),
        "n_configurations": len(campaign_configs),
        "swept_parameters": {
            param: {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_values": len(values),
                "values": [float(v) for v in values]
            }
            for param, values in sweep_params.items()
        },
        "configurations": {
            name: {
                "device_params": config.device.model_dump(),
                "simulation": config.simulation.model_dump(),
            }
            for name, config in campaign_configs.items()
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
) -> None:
    """Print summary of campaign configuration."""
    print_banner("CAMPAIGN SETUP SUMMARY")
    
    print(f"Total Configurations: {len(campaign_configs)}\n")
    
    print("Parameter Sweeps:")
    print("-" * 80)
    for param, values in sweep_params.items():
        print(f"  {param:20s}: {len(values):2d} values | "
              f"Range: [{np.min(values):.6g}, {np.max(values):.6g}]")
    
    print("\n" + "-" * 80)
    
    # Estimate runtime
    base_config = list(campaign_configs.values())[0]
    n_widths = len(base_config.simulation.widths)
    n_circuits = base_config.simulation.n_circuits
    n_shots = base_config.simulation.n_shots
    
    est_time_per_config = 0.5 * n_widths * n_circuits  # Rough estimate: 0.5 sec per circuit
    total_est_time = est_time_per_config * len(campaign_configs)
    
    print(f"\nSimulation Settings (per configuration):")
    print(f"  Widths: {n_widths} ({min(base_config.simulation.widths)}-{max(base_config.simulation.widths)})")
    print(f"  Circuits per width: {n_circuits}")
    print(f"  Shots per circuit: {n_shots}")
    print(f"\nEstimated Runtime:")
    print(f"  Per configuration: ~{est_time_per_config/60:.1f} minutes")
    print(f"  Total campaign: ~{total_est_time/60:.1f} minutes ({total_est_time/3600:.1f} hours)")
    print("\n" + "=" * 80)


def main():
    """Run parameter sweep campaign."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive parameter sweep campaign for QV analysis"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("examples/configs/production.yaml"),
        help="Base configuration file (default: production.yaml)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("campaigns") / f"param_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for campaign results"
    )
    parser.add_argument(
        "--sweep-type",
        type=str,
        choices=["comprehensive", "fidelity_focus", "coherence_focus", "timing_focus", "custom"],
        default="comprehensive",
        help="Type of parameter sweep to perform"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=5,
        help="Number of points per parameter (default: 5)"
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
    print_banner("SPINQ QV PARAMETER CAMPAIGN", "═")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Config: {args.base_config}")
    print(f"Output Dir: {args.output}")
    print(f"Sweep Type: {args.sweep_type}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generate campaign configurations
    print_banner("GENERATING CAMPAIGN CONFIGURATIONS", "-")
    
    campaign_configs, sweep_params = generate_parameter_sweep_configs(
        base_config_path=args.base_config,
        sweep_type=args.sweep_type,
        n_points=args.n_points,
        output_dir=args.output,
    )
    
    print(f"[✓] Generated {len(campaign_configs)} configurations")
    
    # Create manifest
    manifest_path = create_campaign_manifest(
        campaign_configs,
        args.output,
        sweep_params,
        args.base_config,
    )
    
    # Summarize campaign
    summarize_campaign_setup(campaign_configs, sweep_params)
    
    # Dry run check
    if args.dry_run:
        print_banner("DRY RUN COMPLETE", "=")
        print(f"Configurations saved to: {args.output}")
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
    
    # Generate width-grouped plots
    print_banner("GENERATING WIDTH-GROUPED PLOTS", "=")
    
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
    print(f"  • Width-Grouped Plots: {args.output}/plots/by_width/")
    print(f"  • Global Comparison Plots: {args.output}/plots/global/")
    print(f"  • Campaign Analysis Plots: {args.output}/plots/")
    print(f"  • Analysis Report: {args.output}/campaign_report.html")
    print(f"  • Manifest: {manifest_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
