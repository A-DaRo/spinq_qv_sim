#!/usr/bin/env python
"""
Quantum Volume experiment runner (CLI entry point).

Loads configuration, initializes RNG, and orchestrates QV benchmark execution.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging
import numpy as np

from spinq_qv.config import Config
from spinq_qv.utils import setup_logger, initialize_global_rng, get_global_rng_manager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Quantum Volume benchmarks with noise simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path (in addition to console)",
    )
    
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON-structured logging format",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print summary without running simulation",
    )
    
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override random seed from config file",
    )
    
    return parser.parse_args()


def print_config_summary(config: Config, output_dir: Path, seed: int) -> None:
    """
    Print JSON summary of configuration and runtime parameters.
    
    Args:
        config: Validated configuration object
        output_dir: Output directory path
        seed: Active random seed
    """
    summary = {
        "experiment": {
            "name": config.metadata.get("experiment_name", "unnamed"),
            "description": config.metadata.get("description", ""),
        },
        "device_parameters": {
            "F1": config.device.F1,
            "F2": config.device.F2,
            "T1_seconds": config.device.T1,
            "T2_seconds": config.device.T2,
            "T2_star_seconds": config.device.T2_star,
            "single_gate_time_ns": config.device.t_single_gate * 1e9,
            "two_gate_time_ns": config.device.t_two_gate * 1e9,
            "F_readout": config.device.F_readout,
            "F_init": config.device.F_init,
        },
        "simulation_parameters": {
            "backend": config.simulation.backend,
            "n_circuits_per_width": config.simulation.n_circuits,
            "n_shots_per_circuit": config.simulation.n_shots,
            "widths": config.simulation.widths,
            "random_seed": seed,
        },
        "output": {
            "directory": str(output_dir.resolve()),
        },
        "noise_model_derived": {
            "p1_depolarizing": 2 * (1 - config.device.F1),
            "p2_depolarizing": (4/3) * (1 - config.device.F2),
            "p_amp_single_gate": 1 - np.exp(-config.device.t_single_gate / config.device.T1),
            "p_phi_single_gate": 1 - np.exp(-config.device.t_single_gate / _compute_tphi(config.device)),
            "p_amp_two_gate": 1 - np.exp(-config.device.t_two_gate / config.device.T1),
            "p_phi_two_gate": 1 - np.exp(-config.device.t_two_gate / _compute_tphi(config.device)),
        },
    }
    
    print(json.dumps(summary, indent=2))


def _compute_tphi(device_config) -> float:
    """Compute pure dephasing time T_phi = 1 / (1/T2 - 1/(2*T1))."""
    return 1 / (1 / device_config.T2 - 1 / (2 * device_config.T1))


def run_experiment(config: Config, output_dir: Path, seed: int) -> None:
    """
    Execute QV experiment pipeline (placeholder for now).
    
    Args:
        config: Validated configuration
        output_dir: Output directory
        seed: Random seed
    """
    logger.info("=" * 70)
    logger.info("Starting Quantum Volume experiment")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")
    
    # Initialize RNG
    rng_manager = initialize_global_rng(seed)
    logger.info(f"RNG initialized with seed={seed}")
    
    # Placeholder: actual experiment implementation will go here
    logger.info(f"Backend: {config.simulation.backend}")
    logger.info(f"Widths to test: {config.simulation.widths}")
    logger.info(f"Circuits per width: {config.simulation.n_circuits}")
    logger.info(f"Shots per circuit: {config.simulation.n_shots}")
    
    logger.warning("Experiment pipeline not yet implemented (Iteration 1 skeleton)")
    logger.info("RNG state summary:")
    logger.info(json.dumps(rng_manager.get_state_summary(), indent=2))
    
    logger.info("=" * 70)
    logger.info("Experiment skeleton completed successfully")
    logger.info("=" * 70)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logger(
        name="spinq_qv",
        level=log_level,
        log_file=args.log_file,
        json_format=args.json_logs,
    )
    
    logger.info(f"Loading configuration from {args.config}")
    
    try:
        # Load and validate configuration
        config = Config.from_yaml(args.config)
        logger.info("Configuration loaded and validated successfully")
        
        # Determine active seed
        seed = (
            args.seed_override
            if args.seed_override is not None
            else config.simulation.random_seed
        )
        
        if seed is None:
            # Generate random seed from system entropy
            seed = int(np.random.SeedSequence().entropy)
            logger.warning(f"No seed specified, using system entropy: {seed}")
        
        # Print configuration summary
        print_config_summary(config, args.output, seed)
        
        if args.dry_run:
            logger.info("Dry run mode: skipping experiment execution")
            return 0
        
        # Run experiment
        run_experiment(config, args.output, seed)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
