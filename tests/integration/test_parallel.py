"""
Integration test for parallel circuit simulation.

Tests that parallel execution produces consistent results with serial execution.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from spinq_qv.config import Config
from spinq_qv.experiments.run_qv import run_experiment
from spinq_qv.io.storage import QVResultsReader


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path(__file__).parent.parent.parent / "examples" / "configs" / "test_small.yaml"


@pytest.fixture
def test_config(test_config_path):
    """Load test configuration."""
    config = Config.from_yaml(test_config_path)
    # Reduce to minimal size for fast testing
    config.simulation.n_circuits = 5
    config.simulation.widths = [2, 3]
    return config


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_parallel_vs_serial_consistency(test_config, temp_output_dir):
    """
    Test that parallel execution produces same HOPs as serial execution.
    
    Verifies:
    - Same random seed produces same circuits
    - Parallel and serial produce same HOP values
    - Results are deterministic and reproducible
    """
    seed = 99999
    
    # Run serial
    serial_dir = temp_output_dir / "serial"
    serial_results = run_experiment(
        test_config,
        serial_dir,
        seed,
        return_aggregated=True,
        parallel=False,
    )
    
    # Run parallel
    parallel_dir = temp_output_dir / "parallel"
    parallel_results = run_experiment(
        test_config,
        parallel_dir,
        seed,
        return_aggregated=True,
        parallel=True,
        n_workers=2,
    )
    
    # Compare results
    assert set(serial_results.keys()) == set(parallel_results.keys())
    
    for width in serial_results.keys():
        serial_data = serial_results[width]
        parallel_data = parallel_results[width]
        
        # Compare mean HOPs (should be identical with same seed)
        assert 'mean_hop' in serial_data
        assert 'mean_hop' in parallel_data
        
        # Allow small numerical tolerance
        np.testing.assert_allclose(
            serial_data['mean_hop'],
            parallel_data['mean_hop'],
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"Mean HOP mismatch for width {width}"
        )
        
        # Compare individual HOPs
        serial_hops = sorted(serial_data['hops'])
        parallel_hops = sorted(parallel_data['hops'])
        
        assert len(serial_hops) == len(parallel_hops)
        
        np.testing.assert_allclose(
            serial_hops,
            parallel_hops,
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"Individual HOP mismatch for width {width}"
        )


def test_profiling_output(test_config, temp_output_dir):
    """
    Test that profiling generates performance metrics.
    
    Verifies:
    - Performance JSON file is created
    - Metrics contain expected fields
    - Summary statistics are computed
    """
    seed = 54321
    
    # Run with profiling enabled
    run_experiment(
        test_config,
        temp_output_dir,
        seed,
        parallel=False,
        enable_profiling=True,
    )
    
    # Check performance metrics file exists
    perf_file = temp_output_dir / "performance.json"
    assert perf_file.exists(), "Performance metrics file should be created"
    
    # Load and verify structure
    import json
    with open(perf_file, 'r') as f:
        perf_data = json.load(f)
    
    assert 'metrics' in perf_data
    assert 'summary' in perf_data
    
    # Check summary has expected keys
    summary = perf_data['summary']
    assert 'count' in summary
    assert 'total_wall_time' in summary
    assert 'mean_wall_time' in summary
    assert 'peak_memory_mb' in summary
    
    # Check we have metrics for each circuit
    expected_count = sum(1 for _ in test_config.simulation.widths) * test_config.simulation.n_circuits
    assert summary['count'] == expected_count


def test_parallel_speedup(test_config, temp_output_dir):
    """
    Test that parallel execution is faster than serial (with 2+ workers).
    
    Note: This is a smoke test - actual speedup depends on hardware.
    We only verify that parallel mode completes successfully.
    """
    seed = 11111
    
    import time
    
    # Serial timing
    serial_dir = temp_output_dir / "serial"
    start_serial = time.perf_counter()
    run_experiment(
        test_config,
        serial_dir,
        seed,
        parallel=False,
    )
    serial_time = time.perf_counter() - start_serial
    
    # Parallel timing
    parallel_dir = temp_output_dir / "parallel"
    start_parallel = time.perf_counter()
    run_experiment(
        test_config,
        parallel_dir,
        seed,
        parallel=True,
        n_workers=2,
    )
    parallel_time = time.perf_counter() - start_parallel
    
    # Just verify both completed
    assert serial_time > 0
    assert parallel_time > 0
    
    # Log times for information (don't assert speedup as it's hardware-dependent)
    print(f"\nSerial time: {serial_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    if parallel_time < serial_time:
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Note: No speedup observed (expected for small workloads)")
