"""
Integration test for full QV pipeline.

Tests the complete pipeline with small parameters (m∈{2,3,4}, 10 circuits, 500 shots)
to verify HDF5 output structure and HOP values are in plausible range.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from spinq_qv.config import Config
from spinq_qv.experiments.run_qv import run_experiment
from spinq_qv.io.storage import QVResultsReader, load_run


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path(__file__).parent.parent.parent / "examples" / "configs" / "test_small.yaml"


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_qv_pipeline_small(test_config_path, temp_output_dir):
    """
    Test full QV pipeline with small parameters.
    
    Verifies:
    - Pipeline completes without errors
    - HDF5 file is created
    - Expected metadata is present
    - Circuit results are stored for all widths
    - Aggregated results exist for all widths
    - HOP values are in plausible range [0.0, 1.0]
    - QV decision rule is computed
    """
    # Load test config
    config = Config.from_yaml(test_config_path)
    
    assert config.simulation.widths == [2, 3, 4]
    assert config.simulation.n_circuits == 10
    assert config.simulation.n_shots == 500
    
    # Run experiment
    seed = 12345
    run_experiment(config, temp_output_dir, seed)
    
    # Find HDF5 file
    h5_files = list(temp_output_dir.glob("qv_run_*.h5"))
    assert len(h5_files) == 1, "Should create exactly one HDF5 file"
    
    h5_path = h5_files[0]
    assert h5_path.exists()
    
    # Read and verify results
    with QVResultsReader(h5_path) as reader:
        # Check metadata
        metadata = reader.read_metadata()
        
        assert 'config' in metadata
        assert 'git_hash' in metadata
        assert 'timestamp' in metadata
        assert metadata['config']['simulation']['widths'] == [2, 3, 4]
        
        # Check all widths are present
        widths = reader.list_widths()
        assert set(widths) == {2, 3, 4}
        
        # Check each width has correct number of circuits
        for width in widths:
            circuit_ids = reader.list_circuits(width)
            assert len(circuit_ids) == 10, f"Width {width} should have 10 circuits"
            
            # Verify first circuit structure
            circuit_result = reader.read_circuit_result(width, circuit_ids[0])
            
            assert 'circuit_id' in circuit_result
            assert 'width' in circuit_result
            assert circuit_result['width'] == width
            assert 'ideal_probs' in circuit_result
            assert 'measured_counts' in circuit_result
            assert 'hop' in circuit_result
            
            # Verify ideal probabilities
            ideal_probs = circuit_result['ideal_probs']
            assert len(ideal_probs) == 2**width
            assert np.isclose(np.sum(ideal_probs), 1.0), "Probabilities should sum to 1"
            assert np.all(ideal_probs >= 0), "Probabilities should be non-negative"
            assert np.all(ideal_probs <= 1), "Probabilities should be <= 1"
            
            # Verify HOP is in valid range
            hop = circuit_result['hop']
            assert 0.0 <= hop <= 1.0, f"HOP should be in [0,1], got {hop}"
            
            # Verify measured counts
            counts = circuit_result['measured_counts']
            assert isinstance(counts, dict)
            total_shots = sum(counts.values())
            assert total_shots == 500, "Should have 500 total shots"
        
        # Check aggregated results
        for width in widths:
            agg_result = reader.read_aggregated_results(width)
            
            assert 'mean_hop' in agg_result
            assert 'ci_lower' in agg_result
            assert 'ci_upper' in agg_result
            assert 'pass_mean' in agg_result
            assert 'pass_ci' in agg_result
            assert 'pass_qv' in agg_result
            assert 'n_circuits' in agg_result
            
            assert agg_result['n_circuits'] == 10
            
            # Verify HOP statistics are sensible
            mean_hop = agg_result['mean_hop']
            ci_lower = agg_result['ci_lower']
            ci_upper = agg_result['ci_upper']
            
            assert 0.0 <= mean_hop <= 1.0
            assert 0.0 <= ci_lower <= 1.0
            assert 0.0 <= ci_upper <= 1.0
            assert ci_lower <= mean_hop <= ci_upper, "Mean should be in CI"
            
            # For noiseless statevector, expect high HOP
            # (close to 0.5 in theory, but with sampling noise can vary)
            # Just check it's in plausible range
            assert 0.33 <= mean_hop <= 1.0, (
                f"Mean HOP for width {width} should be > 0.33 for noiseless, got {mean_hop}"
            )


def test_load_run_convenience(test_config_path, temp_output_dir):
    """Test the convenience load_run function."""
    # Run small experiment
    config = Config.from_yaml(test_config_path)
    run_experiment(config, temp_output_dir, seed=12345)
    
    # Find HDF5 file
    h5_files = list(temp_output_dir.glob("qv_run_*.h5"))
    h5_path = h5_files[0]
    
    # Load using convenience function
    data = load_run(h5_path)
    
    assert 'metadata' in data
    assert 'widths' in data
    assert 'aggregated' in data
    
    assert set(data['widths']) == {2, 3, 4}
    
    for width in data['widths']:
        assert width in data['aggregated']
        assert 'mean_hop' in data['aggregated'][width]


def test_plots_generated(test_config_path, temp_output_dir):
    """Test that plots are generated correctly."""
    config = Config.from_yaml(test_config_path)
    run_experiment(config, temp_output_dir, seed=12345)
    
    # Check plots directory
    plot_dirs = list(temp_output_dir.glob("plots_*"))
    assert len(plot_dirs) > 0, "Should create plots directory"
    
    plots_dir = plot_dirs[0]
    
    # Check for expected plot files
    expected_files = [
        "hop_vs_width.png",
        "hop_vs_width.svg",
        "qv_summary.png",
        "qv_summary.svg",
    ]
    
    for expected_file in expected_files:
        assert (plots_dir / expected_file).exists(), f"Missing plot: {expected_file}"
    
    # Check per-width distribution plots
    for width in [2, 3, 4]:
        assert (plots_dir / f"hop_dist_m{width}.png").exists()
        assert (plots_dir / f"hop_dist_m{width}.svg").exists()
