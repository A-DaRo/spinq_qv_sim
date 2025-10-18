#!/usr/bin/env python
"""
Quantum Volume experiment runner (CLI entry point).

Loads configuration, initializes RNG, and orchestrates QV benchmark execution.
Full pipeline: generate circuits, transpile, schedule, simulate, measure, analyze.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

from spinq_qv.config import Config
from spinq_qv.utils import setup_logger, initialize_global_rng, get_global_rng_manager
from spinq_qv.utils.perf import PerformanceProfiler, PerformanceLogger, estimate_memory_requirements
from spinq_qv.circuits.generator import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.circuits.transpiler import Transpiler, create_linear_topology, create_all_to_all_topology
from spinq_qv.circuits.scheduling import create_default_scheduler
from spinq_qv.sim.statevector import StatevectorBackend
from spinq_qv.sim.density_matrix import DensityMatrixBackend
from spinq_qv.sim.mcwf import MCWFBackend
from spinq_qv.noise.builder import NoiseModelBuilder
from spinq_qv.analysis.hop import compute_hop_from_result
from spinq_qv.analysis.stats import aggregate_results_by_width, qv_decision_rule
from spinq_qv.analysis.plots import save_all_plots, save_all_sensitivity_plots
from spinq_qv.io.storage import QVResultsWriter
from spinq_qv.io.formats import int_to_bitstring
from spinq_qv.experiments.sensitivity import SensitivityRunner, create_default_param_ranges
from spinq_qv.experiments.campaign import ProductionCampaignRunner

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
        "--mode",
        choices=["qv", "campaign", "ablation", "sensitivity-1d", "sensitivity-2d"],
        default="qv",
        help="Experiment mode: qv (standard QV), campaign (production multi-width with resume), "
             "ablation (error budget), sensitivity-1d (1D parameter sweep), sensitivity-2d (2D grid)",
    )
    
    # Sensitivity-specific options
    parser.add_argument(
        "--param",
        type=str,
        default="device.F2",
        help="Parameter to sweep for sensitivity mode (dot-separated path, e.g., 'device.F2')",
    )
    
    parser.add_argument(
        "--param2",
        type=str,
        default=None,
        help="Second parameter for 2D sensitivity grid (e.g., 'device.F1')",
    )
    
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated values for parameter sweep (e.g., '0.99,0.995,0.998,1.0')",
    )
    
    parser.add_argument(
        "--values2",
        type=str,
        default=None,
        help="Comma-separated values for second parameter in 2D grid",
    )
    
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save checkpoint every N runs in sensitivity mode",
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
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel circuit simulation using multiprocessing",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling and save metrics",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous campaign (campaign mode only)",
    )
    
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Campaign ID for resume or to specify custom ID",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts for failed widths in campaign mode",
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


def simulate_single_circuit_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel circuit simulation.
    
    This function is called by each worker process in parallel mode.
    Must be picklable (top-level function, no closures).
    
    Args:
        args: Tuple of (config_dict, m, circuit_idx, circuit_seed, noise_model_dict, topology_dict)
    
    Returns:
        Dictionary with circuit results
    """
    from spinq_qv.config import Config
    from spinq_qv.circuits.generator import generate_qv_circuit, compute_ideal_probabilities
    from spinq_qv.circuits.transpiler import Transpiler, DeviceTopology
    from spinq_qv.circuits.scheduling import create_default_scheduler
    from spinq_qv.analysis.hop import compute_hop_from_result
    from spinq_qv.utils.perf import PerformanceProfiler
    
    # Unpack arguments
    config_dict, m, circuit_idx, circuit_seed, noise_model_dict, topology_dict = args
    
    # Reconstruct objects
    config = Config(**config_dict)
    
    # Reconstruct topology manually since we can't pass topology_type
    topology = DeviceTopology.__new__(DeviceTopology)
    topology.n_qubits = topology_dict['n_qubits']
    topology.edges = set(tuple(edge) for edge in topology_dict['connectivity'])
    topology.topology_type = "custom"
    
    transpiler = Transpiler(topology)
    
    # Start profiling
    with PerformanceProfiler(f"circuit_m{m}_idx{circuit_idx}") as prof:
        # Generate QV circuit
        circuit = generate_qv_circuit(m, circuit_seed)
        
        # Transpile
        transpiled_circuit = transpiler.transpile(circuit)
        
        # Schedule
        scheduler = create_default_scheduler(config.device.model_dump())
        scheduled_circuit = scheduler.schedule(transpiled_circuit)
        
        # Compute ideal probabilities
        ideal_probs = compute_ideal_probabilities(circuit)
        
        # Simulate
        backend = create_backend(
            config.simulation.backend,
            m,
            circuit_seed + 1000,
        )
        
        measured_counts = simulate_circuit(
            backend,
            transpiled_circuit,
            config.simulation.n_shots,
            noise_model_dict,
            m,
        )
        
        # Compute HOP
        hop, heavy_count, total_shots = compute_hop_from_result(
            measured_counts,
            ideal_probs,
            threshold_type='median',
        )
    
    return {
        'circuit_idx': circuit_idx,
        'circuit_seed': circuit_seed,
        'circuit_spec': circuit,
        'ideal_probs': ideal_probs,
        'measured_counts': measured_counts,
        'hop': hop,
        'heavy_count': heavy_count,
        'total_shots': total_shots,
        'perf_metrics': prof.metrics.to_dict() if prof.metrics else None,
    }


def run_experiment(
    config: Config,
    output_dir: Path,
    seed: int,
    return_aggregated: bool = False,
    parallel: bool = False,
    n_workers: int = 4,
    enable_profiling: bool = False,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Execute full QV experiment pipeline.
    
    Pipeline:
    1. Generate QV circuits for each width
    2. Transpile for device topology
    3. Schedule gates in time
    4. Simulate with noise model
    5. Measure and compute HOPs
    6. Aggregate statistics
    7. Save to HDF5
    8. Generate plots
    
    Args:
        config: Validated configuration
        output_dir: Output directory
        seed: Random seed
        return_aggregated: If True, return aggregated results dict instead of None
        parallel: Enable parallel circuit simulation
        n_workers: Number of parallel workers
        enable_profiling: Enable performance profiling
    
    Returns:
        None (default) or aggregated results dict if return_aggregated=True
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
    
    # Build noise model
    logger.info("Building noise model from device parameters...")
    noise_builder = NoiseModelBuilder(config.device.model_dump())
    gate_durations = {
        'single': config.device.t_single_gate,
        'two_qubit': config.device.t_two_gate,
    }
    noise_model = noise_builder.build(gate_durations)
    
    logger.info(f"Noise model built:")
    logger.info(f"  Single-qubit p_dep={noise_model['single_qubit']['p_dep']:.6f}")
    logger.info(f"  Two-qubit p_dep={noise_model['two_qubit']['p_dep']:.6f}")
    
    # Prepare transpiler and scheduler
    logger.info("Initializing transpiler and scheduler...")
    # For now, use all-to-all topology (no SWAP overhead)
    # Can switch to linear topology for realistic Si/SiGe arrays
    max_width = max(config.simulation.widths)
    topology = create_all_to_all_topology(max_width)
    transpiler = Transpiler(topology)
    scheduler = create_default_scheduler(config.device.model_dump())
    
    # Prepare HDF5 writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"qv_run_{timestamp}.h5"
    
    logger.info(f"Results will be saved to: {output_file}")
    
    with QVResultsWriter(output_file) as writer:
        # Write metadata
        writer.write_metadata(
            config=config.model_dump(),
            additional_metadata={
                'seed': seed,
                'noise_model_summary': {
                    'p1_dep': noise_model['single_qubit']['p_dep'],
                    'p2_dep': noise_model['two_qubit']['p_dep'],
                }
            }
        )
        
        # Storage for all results
        all_results = []
        
        # Initialize performance logger if needed
        perf_logger = PerformanceLogger(output_dir / "performance.json") if enable_profiling else None
        
        # Prepare topology dict for workers
        topology_dict = {
            'n_qubits': topology.n_qubits,
            'connectivity': list(topology.edges),  # Convert set to list for JSON serialization
        }
        
        # Loop over circuit widths
        for m in config.simulation.widths:
            logger.info("=" * 70)
            logger.info(f"Processing width m={m}")
            logger.info("=" * 70)
            
            # Estimate memory requirements
            if enable_profiling:
                mem_est = estimate_memory_requirements(m, config.simulation.backend)
                logger.info(f"  Estimated memory: {mem_est['estimated_total_mb']:.1f} MB per circuit")
            
            width_results = []
            
            # Generate seeds for all circuits in this width
            circuit_seeds = []
            for circuit_idx in range(config.simulation.n_circuits):
                circuit_rng = rng_manager.get_rng(f"circuit_m{m}_idx{circuit_idx}")
                circuit_seed = int(circuit_rng.integers(0, 2**31 - 1))
                circuit_seeds.append((circuit_idx, circuit_seed))
            
            if parallel:
                # Parallel execution
                logger.info(f"  Running {config.simulation.n_circuits} circuits in parallel with {n_workers} workers...")
                
                # Prepare worker arguments
                worker_args = [
                    (
                        config.model_dump(),
                        m,
                        circuit_idx,
                        circuit_seed,
                        noise_model,
                        topology_dict,
                    )
                    for circuit_idx, circuit_seed in circuit_seeds
                ]
                
                # Execute in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all jobs
                    futures = {
                        executor.submit(simulate_single_circuit_worker, args): args[2]
                        for args in worker_args
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        circuit_idx = futures[future]
                        try:
                            result = future.result()
                            
                            logger.info(f"    Circuit {result['circuit_idx'] + 1}/{config.simulation.n_circuits} "
                                      f"complete: HOP = {result['hop']:.4f}")
                            
                            # Save circuit result
                            circuit_id = f"circuit_{result['circuit_idx']:04d}"
                            writer.write_circuit_result(
                                width=m,
                                circuit_id=circuit_id,
                                circuit_spec=result['circuit_spec'],
                                ideal_probs=result['ideal_probs'],
                                measured_counts=result['measured_counts'],
                                hop=result['hop'],
                                metadata={
                                    'circuit_seed': result['circuit_seed'],
                                }
                            )
                            
                            # Store for aggregation
                            width_results.append({
                                'width': m,
                                'hop': result['hop'],
                                'circuit_id': circuit_id,
                            })
                            all_results.append({
                                'width': m,
                                'hop': result['hop'],
                            })
                            
                            # Log performance metrics
                            if enable_profiling and result['perf_metrics']:
                                from spinq_qv.utils.perf import PerformanceMetrics
                                metrics = PerformanceMetrics(**result['perf_metrics'])
                                perf_logger.add_metrics(metrics)
                        
                        except Exception as e:
                            logger.error(f"    Circuit {circuit_idx} failed: {e}")
                            raise
            
            else:
                # Serial execution
                logger.info(f"  Running {config.simulation.n_circuits} circuits serially...")
                
                for circuit_idx, circuit_seed in circuit_seeds:
                    logger.info(f"  Circuit {circuit_idx + 1}/{config.simulation.n_circuits} "
                               f"(seed={circuit_seed})")
                    
                    # Profile if enabled
                    if enable_profiling:
                        prof = PerformanceProfiler(f"circuit_m{m}_idx{circuit_idx}")
                        prof.__enter__()
                    
                    # Generate QV circuit
                    circuit = generate_qv_circuit(m, circuit_seed)
                    
                    # Transpile
                    transpiled_circuit = transpiler.transpile(circuit)
                    
                    # Schedule
                    scheduled_circuit = scheduler.schedule(transpiled_circuit)
                    
                    # Compute ideal probabilities
                    ideal_probs = compute_ideal_probabilities(circuit)
                    
                    # Simulate with noise
                    backend = create_backend(
                        config.simulation.backend,
                        m,
                        circuit_seed + 1000,
                    )
                    
                    # Apply circuit gates
                    measured_counts = simulate_circuit(
                        backend,
                        transpiled_circuit,
                        config.simulation.n_shots,
                        noise_model,
                        m,
                    )
                    
                    # Compute HOP
                    hop, heavy_count, total_shots = compute_hop_from_result(
                        measured_counts,
                        ideal_probs,
                        threshold_type='median',
                    )
                    
                    if enable_profiling:
                        prof.__exit__(None, None, None)
                        perf_logger.add_metrics(prof.metrics)
                    
                    logger.info(f"    HOP = {hop:.4f} ({heavy_count}/{total_shots} in heavy outputs)")
                    
                    # Save circuit result
                    circuit_id = f"circuit_{circuit_idx:04d}"
                    writer.write_circuit_result(
                        width=m,
                        circuit_id=circuit_id,
                        circuit_spec=circuit,
                        ideal_probs=ideal_probs,
                        measured_counts=measured_counts,
                        hop=hop,
                        metadata={
                            'circuit_seed': circuit_seed,
                            'num_gates': circuit.num_gates(),
                            'num_swaps': transpiled_circuit.metadata.get('num_swaps', 0),
                        }
                    )
                    
                    # Store for aggregation
                    width_results.append({
                        'width': m,
                        'hop': hop,
                        'circuit_id': circuit_id,
                    })
                    all_results.append({
                        'width': m,
                        'hop': hop,
                    })
            
            # Aggregate results for this width
            logger.info(f"  Aggregating results for m={m}...")
            hops_array = np.array([r['hop'] for r in width_results])
            
            qv_result = qv_decision_rule(hops_array, random_seed=seed)
            
            logger.info(f"  Mean HOP: {qv_result['mean_hop']:.4f}")
            logger.info(f"  95% CI: [{qv_result['ci_lower']:.4f}, {qv_result['ci_upper']:.4f}]")
            logger.info(f"  QV passed: {qv_result['pass_qv']}")
            
            # Write aggregated results
            writer.write_aggregated_results(m, {
                **qv_result,
                'hops': hops_array,
            })
        
        logger.info("=" * 70)
        logger.info("All widths completed")
        logger.info("=" * 70)
        
        # Save performance metrics
        if enable_profiling and perf_logger:
            perf_logger.save()
            perf_logger.print_summary()
    
    # Generate plots
    logger.info("Generating plots...")
    plots_dir = output_dir / f"plots_{timestamp}"
    
    aggregated = aggregate_results_by_width(all_results)
    save_all_plots(aggregated, plots_dir, formats=['png', 'svg'])
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("QV EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    
    for width in sorted(aggregated.keys()):
        result = aggregated[width]
        status = "[PASS]" if result['pass_qv'] else "[FAIL]"
        logger.info(f"Width m={width}: {status}")
        logger.info(f"  Mean HOP: {result['mean_hop']:.4f}")
        logger.info(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Plots saved to: {plots_dir}/")
    logger.info("=" * 70)
    
    # Return aggregated results if requested
    if return_aggregated:
        return aggregated
    return None


def create_backend(backend_type: str, n_qubits: int, seed: int):
    """
    Create simulation backend.
    
    Args:
        backend_type: Backend type ('statevector', 'density_matrix', 'mcwf')
        n_qubits: Number of qubits
        seed: Random seed
    
    Returns:
        Simulator backend instance
    """
    if backend_type == "statevector":
        return StatevectorBackend(n_qubits, seed=seed)
    elif backend_type == "density_matrix":
        return DensityMatrixBackend(n_qubits, seed=seed)
    elif backend_type == "mcwf":
        return MCWFBackend(n_qubits, seed=seed, n_trajectories=100)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def simulate_circuit(
    backend,
    circuit,
    n_shots: int,
    noise_model: Dict[str, Any],
    m: int,
) -> Dict[str, int]:
    """
    Simulate a circuit with noise and measure.
    
    Args:
        backend: Simulator backend
        circuit: Circuit specification
        n_shots: Number of measurement shots
        noise_model: Noise model dictionary
        m: Number of qubits
    
    Returns:
        Dictionary of measurement counts
    """
    # Initialize state to |0...0⟩
    backend.init_state()
    
    # Extract noise model components
    single_qubit_noise = noise_model.get('single_qubit', {})
    two_qubit_noise = noise_model.get('two_qubit', {})
    
    # Apply each gate with noise
    for gate in circuit.gates:
        gate_type = gate["type"]
        qubits = gate["qubits"]
        params = gate.get("params", {})
        
        if gate_type == "u3":
            # Single-qubit U3 gate
            theta = params["theta"]
            phi = params["phi"]
            lam = params["lambda"]
            
            # Build U3 matrix
            cos_half = np.cos(theta / 2)
            sin_half = np.sin(theta / 2)
            
            u3_matrix = np.array([
                [cos_half, -np.exp(1j * lam) * sin_half],
                [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lam)) * cos_half]
            ], dtype=np.complex128)
            
            # Apply ideal gate
            backend.apply_unitary(u3_matrix, [qubits[0]])
            
            # Apply noise channels in order: amplitude damping, phase damping, depolarizing
            if 'amp_kraus' in single_qubit_noise:
                backend.apply_kraus(single_qubit_noise['amp_kraus'], [qubits[0]])
            
            if 'phase_kraus' in single_qubit_noise:
                backend.apply_kraus(single_qubit_noise['phase_kraus'], [qubits[0]])
            
            if 'dep_kraus' in single_qubit_noise:
                backend.apply_kraus(single_qubit_noise['dep_kraus'], [qubits[0]])
        
        elif gate_type == "su4":
            # Two-qubit SU(4) gate
            real_part = np.array(params["unitary_real"]).reshape(4, 4)
            imag_part = np.array(params["unitary_imag"]).reshape(4, 4)
            unitary = (real_part + 1j * imag_part).astype(np.complex128)
            
            # Apply ideal gate
            backend.apply_unitary(unitary, qubits)
            
            # Apply per-qubit decoherence (amplitude + phase damping)
            if 'amp_kraus_per_qubit' in two_qubit_noise:
                for q in qubits:
                    backend.apply_kraus(two_qubit_noise['amp_kraus_per_qubit'], [q])
            
            if 'phase_kraus_per_qubit' in two_qubit_noise:
                for q in qubits:
                    backend.apply_kraus(two_qubit_noise['phase_kraus_per_qubit'], [q])
            
            # Apply two-qubit depolarizing
            if 'dep_kraus' in two_qubit_noise:
                backend.apply_kraus(two_qubit_noise['dep_kraus'], qubits)
        
        elif gate_type == "swap":
            # SWAP gate
            swap_matrix = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.complex128)
            
            # Apply ideal gate
            backend.apply_unitary(swap_matrix, qubits)
            
            # Apply same noise as two-qubit gate
            if 'amp_kraus_per_qubit' in two_qubit_noise:
                for q in qubits:
                    backend.apply_kraus(two_qubit_noise['amp_kraus_per_qubit'], [q])
            
            if 'phase_kraus_per_qubit' in two_qubit_noise:
                for q in qubits:
                    backend.apply_kraus(two_qubit_noise['phase_kraus_per_qubit'], [q])
            
            if 'dep_kraus' in two_qubit_noise:
                backend.apply_kraus(two_qubit_noise['dep_kraus'], qubits)
    
    # Measure
    counts = backend.measure(shots=n_shots, readout_noise=None)
    
    return counts


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
        
        # Route to appropriate mode
        if args.mode == "qv":
            # Standard QV experiment
            run_experiment(
                config, args.output, seed,
                parallel=args.parallel,
                n_workers=args.workers,
                enable_profiling=args.profile,
            )
        
        elif args.mode == "campaign":
            # Production campaign with resume capability
            run_production_campaign(
                config, args.output, seed,
                resume=args.resume,
                campaign_id=args.campaign_id,
                parallel=args.parallel,
                n_workers=args.workers,
                enable_profiling=args.profile,
                max_retries=args.max_retries,
            )
        
        elif args.mode == "ablation":
            # Ablation study for error budget
            run_ablation_study(config, args.output, seed, args.checkpoint_freq)
        
        elif args.mode == "sensitivity-1d":
            # 1D parameter sweep
            run_sensitivity_1d(config, args.output, seed, args.param, args.values, args.checkpoint_freq)
        
        elif args.mode == "sensitivity-2d":
            # 2D parameter grid
            run_sensitivity_2d(
                config, args.output, seed,
                args.param, args.values,
                args.param2, args.values2,
                args.checkpoint_freq
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


def run_ablation_study(config: Config, output_dir: Path, seed: int, checkpoint_freq: int) -> None:
    """
    Run ablation study to compute error budget.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        seed: Random seed
        checkpoint_freq: Checkpoint frequency
    """
    logger.info("Running ablation study...")
    
    from spinq_qv.analysis.ablation import generate_ablation_sweep, compute_error_budget
    
    runner = SensitivityRunner(config, output_dir / "ablation", checkpoint_freq)
    results = runner.run_ablation_sweep(base_seed=seed)
    
    # Compute and export error budget
    runner.compute_and_save_error_budget()
    
    # Generate plots
    logger.info("Generating ablation plots...")
    plots_dir = output_dir / "ablation" / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Import and use plot functions as needed
    logger.info(f"Ablation study complete. Results in {output_dir / 'ablation'}")


def run_production_campaign(
    config: Config,
    output_dir: Path,
    seed: int,
    resume: bool = False,
    campaign_id: Optional[str] = None,
    parallel: bool = False,
    n_workers: int = 4,
    enable_profiling: bool = False,
    max_retries: int = 2,
) -> None:
    """
    Run production QV campaign with resume capability.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        seed: Random seed
        resume: Resume previous campaign
        campaign_id: Campaign identifier
        parallel: Enable parallel execution
        n_workers: Number of workers
        enable_profiling: Enable profiling
        max_retries: Maximum retry attempts for failed widths
    """
    logger.info("Starting production campaign...")
    
    # Create campaign runner
    runner = ProductionCampaignRunner(
        config=config,
        output_dir=output_dir,
        campaign_id=campaign_id,
        resume=resume,
    )
    
    # Run campaign
    summary = runner.run(
        parallel=parallel,
        n_workers=n_workers,
        enable_profiling=enable_profiling,
        max_retries=max_retries,
    )
    
    logger.info(f"Campaign complete: {summary['completed_widths']}/{summary['total_widths']} widths successful")


def run_ablation_study(config: Config, output_dir: Path, seed: int, checkpoint_freq: int) -> None:
    """
    Run ablation study to compute error budget.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        seed: Random seed
        checkpoint_freq: Checkpoint frequency
    """
    logger.info("Running ablation study...")
    
    # Create wrapper function for run_experiment
    def qv_runner_func(cfg: Config, **kwargs):
        # Run QV and return aggregated results
        initialize_global_rng(seed)
        aggregated_results = run_experiment(cfg, output_dir / "temp", seed, return_aggregated=True)
        return aggregated_results
    
    # Create sensitivity runner
    runner = SensitivityRunner(config, output_dir / "ablation", checkpoint_freq)
    
    # Run ablation sweep
    df = runner.run_ablation_sweep(qv_runner_func)
    
    logger.info(f"Ablation sweep complete. Results saved to {output_dir / 'ablation' / 'ablation_results.csv'}")
    
    # Compute and save error budget
    budget = runner.compute_and_save_error_budget()
    
    # Generate plots
    save_all_sensitivity_plots(
        {},  # No aggregated results for ablation
        budget=budget,
        output_dir=output_dir / "ablation" / "plots"
    )
    
    logger.info("[PASS] Ablation study complete!")


def run_sensitivity_1d(
    config: Config,
    output_dir: Path,
    seed: int,
    param: str,
    values: Optional[str],
    checkpoint_freq: int
) -> None:
    """
    Run 1D parameter sensitivity sweep.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        seed: Random seed
        param: Parameter path (e.g., 'device.F2')
        values: Comma-separated values or None for defaults
        checkpoint_freq: Checkpoint frequency
    """
    logger.info(f"Running 1D sensitivity sweep for {param}...")
    
    # Parse parameter path
    param_path = param.split('.')
    
    # Parse values
    if values:
        param_values = [float(v.strip()) for v in values.split(',')]
    else:
        # Use default ranges
        param_ranges = create_default_param_ranges(config)
        param_name = param_path[-1]
        if param_name not in param_ranges:
            raise ValueError(f"No default range for parameter '{param_name}'. Please specify --values.")
        param_values = param_ranges[param_name]
    
    logger.info(f"Sweeping {len(param_values)} values: {param_values}")
    
    # Create wrapper function
    def qv_runner_func(cfg: Config, **kwargs):
        initialize_global_rng(seed)
        aggregated_results = run_experiment(cfg, output_dir / "temp", seed, return_aggregated=True)
        return aggregated_results
    
    # Create sensitivity runner
    runner = SensitivityRunner(config, output_dir / "sensitivity_1d", checkpoint_freq)
    
    # Run sweep
    df = runner.run_parameter_sweep(param_path, param_values, qv_runner_func)
    
    logger.info(f"Sweep complete. Results saved to {output_dir / 'sensitivity_1d' / f'{param_path[-1]}_sweep.csv'}")
    
    logger.info("[PASS] 1D sensitivity sweep complete!")


def run_sensitivity_2d(
    config: Config,
    output_dir: Path,
    seed: int,
    param1: str,
    values1: Optional[str],
    param2: Optional[str],
    values2: Optional[str],
    checkpoint_freq: int
) -> None:
    """
    Run 2D parameter sensitivity grid.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        seed: Random seed
        param1: First parameter path
        values1: Values for first parameter
        param2: Second parameter path
        values2: Values for second parameter
        checkpoint_freq: Checkpoint frequency
    """
    if param2 is None:
        raise ValueError("Must specify --param2 for 2D sensitivity mode")
    
    logger.info(f"Running 2D sensitivity grid: {param1} x {param2}...")
    
    # Parse parameter paths
    param1_path = param1.split('.')
    param2_path = param2.split('.')
    
    # Parse values
    param_ranges = create_default_param_ranges(config)
    
    if values1:
        param1_values = [float(v.strip()) for v in values1.split(',')]
    else:
        param1_name = param1_path[-1]
        if param1_name not in param_ranges:
            raise ValueError(f"No default range for parameter '{param1_name}'. Please specify --values.")
        param1_values = param_ranges[param1_name]
    
    if values2:
        param2_values = [float(v.strip()) for v in values2.split(',')]
    else:
        param2_name = param2_path[-1]
        if param2_name not in param_ranges:
            raise ValueError(f"No default range for parameter '{param2_name}'. Please specify --values2.")
        param2_values = param_ranges[param2_name]
    
    logger.info(f"Grid size: {len(param1_values)} x {len(param2_values)} = {len(param1_values) * len(param2_values)} points")
    
    # Create wrapper function
    def qv_runner_func(cfg: Config, **kwargs):
        initialize_global_rng(seed)
        aggregated_results = run_experiment(cfg, output_dir / "temp", seed, return_aggregated=True)
        return aggregated_results
    
    # Create sensitivity runner
    runner = SensitivityRunner(config, output_dir / "sensitivity_2d", checkpoint_freq)
    
    # Run grid
    df, grid = runner.run_2d_grid(
        param1_path, param1_values,
        param2_path, param2_values,
        qv_runner_func
    )
    
    param1_name = param1_path[-1]
    param2_name = param2_path[-1]
    
    logger.info(f"Grid complete. Results saved to {output_dir / 'sensitivity_2d' / f'{param1_name}_vs_{param2_name}_grid.csv'}")
    
    # Generate heatmap
    grid_data = (grid, param1_values, param2_values, param1_name, param2_name)
    save_all_sensitivity_plots(
        {},
        budget=None,
        grid_data=grid_data,
        output_dir=output_dir / "sensitivity_2d" / "plots"
    )
    
    logger.info("[PASS] 2D sensitivity grid complete!")


if __name__ == "__main__":
    sys.exit(main())
