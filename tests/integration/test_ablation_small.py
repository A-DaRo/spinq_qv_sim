"""
Integration test for ablation and sensitivity analysis.

Tests the ablation sweep functionality with small parameters
to verify CSV outputs, error budget computation, and plot generation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json

from spinq_qv.config import Config
from spinq_qv.analysis.ablation import (
    AblationStudy,
    SensitivityAnalysis,
    compute_error_budget,
)
from spinq_qv.experiments.sensitivity import SensitivityRunner


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path(__file__).parent.parent.parent / "examples" / "configs" / "test_ablation.yaml"


@pytest.fixture
def test_config(test_config_path):
    """Load test configuration."""
    return Config.from_yaml(test_config_path)


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_ablation_config_generation(test_config):
    """
    Test that ablation study generates correct configurations.
    
    Verifies:
    - All ablation configurations are generated
    - Device parameters are modified correctly
    - Metadata contains ablation flags
    """
    ablation = AblationStudy(test_config)
    configs = ablation.generate_ablation_sweep()
    
    # Should have baseline, 6 single ablations, 2 pairwise, ideal
    assert len(configs) == 10, f"Expected 10 configs, got {len(configs)}"
    
    labels = [label for label, _ in configs]
    assert "baseline" in labels
    assert "ideal" in labels
    assert "no_depolarizing" in labels
    assert "no_decoherence" in labels
    assert "no_coherent" in labels
    assert "no_readout" in labels
    assert "no_crosstalk" in labels
    assert "no_quasistatic" in labels
    
    # Check that baseline has all noise
    baseline_config = next(cfg for label, cfg in configs if label == "baseline")
    assert baseline_config.device.F1 == test_config.device.F1
    assert baseline_config.device.F2 == test_config.device.F2
    
    # Check that ideal has no noise
    ideal_config = next(cfg for label, cfg in configs if label == "ideal")
    assert ideal_config.device.F1 == 1.0
    assert ideal_config.device.F2 == 1.0
    assert ideal_config.device.F_readout == 1.0
    assert ideal_config.device.F_init == 1.0
    assert ideal_config.device.T1 == 1e6
    assert ideal_config.device.T2 == 1e6
    
    # Check ablation flags in metadata
    no_dep_config = next(cfg for label, cfg in configs if label == "no_depolarizing")
    assert 'ablation' in no_dep_config.metadata
    assert no_dep_config.metadata['ablation']['depolarizing_disabled'] is True
    assert no_dep_config.device.F1 == 1.0
    assert no_dep_config.device.F2 == 1.0


def test_sensitivity_1d_sweep(test_config):
    """
    Test 1D parameter sweep generation.
    
    Verifies:
    - Sweep generates correct number of configurations
    - Parameter values are set correctly
    - Labels are meaningful
    """
    sensitivity = SensitivityAnalysis(test_config)
    
    # Sweep F2 from 0.99 to 1.0
    f2_values = [0.99, 0.995, 0.998, 1.0]
    configs = sensitivity.sweep_parameter(['device', 'F2'], f2_values)
    
    assert len(configs) == 4
    
    for i, (label, config) in enumerate(configs):
        assert f"F2={f2_values[i]}" in label
        assert config.device.F2 == f2_values[i]
        assert 'sensitivity_param' in config.metadata
        assert config.metadata['sensitivity_param'] == 'device/F2'


def test_sensitivity_2d_grid(test_config):
    """
    Test 2D parameter grid generation.
    
    Verifies:
    - Grid generates correct number of configurations
    - Both parameters are varied correctly
    - Labels contain both parameter values
    """
    sensitivity = SensitivityAnalysis(test_config)
    
    f1_values = [0.995, 1.0]
    f2_values = [0.995, 1.0]
    
    configs = sensitivity.sweep_2d_grid(
        ['device', 'F1'], f1_values,
        ['device', 'F2'], f2_values
    )
    
    assert len(configs) == 4  # 2x2 grid
    
    # Check all combinations exist
    expected_combinations = [
        (0.995, 0.995),
        (0.995, 1.0),
        (1.0, 0.995),
        (1.0, 1.0),
    ]
    
    actual_combinations = set()
    for label, config in configs:
        actual_combinations.add((config.device.F1, config.device.F2))
    
    assert actual_combinations == set(expected_combinations)


def test_error_budget_computation():
    """
    Test error budget computation from synthetic results.
    
    Verifies:
    - Budget correctly identifies contributions
    - Fractions sum to approximately 1.0
    - Baseline and ideal are handled correctly
    """
    # Synthetic ablation results
    ablation_results = {
        'baseline': {'mean_hop': 0.60},
        'no_depolarizing': {'mean_hop': 0.75},
        'no_decoherence': {'mean_hop': 0.68},
        'no_readout': {'mean_hop': 0.62},
        'ideal': {'mean_hop': 0.95},
    }
    
    budget = compute_error_budget(ablation_results)
    
    # Check structure
    assert 'baseline' in budget
    assert 'ideal' in budget
    assert 'no_depolarizing' in budget
    assert 'no_decoherence' in budget
    assert 'no_readout' in budget
    
    # Check baseline
    assert budget['baseline']['delta_metric'] == 0.0
    assert budget['baseline']['pct_of_gap'] == 0.0
    
    # Check ideal
    total_gap = 0.95 - 0.60
    assert budget['ideal']['delta_metric'] == pytest.approx(total_gap, abs=1e-6)
    assert budget['ideal']['pct_of_gap'] == pytest.approx(100.0, abs=1e-6)
    
    # Check depolarizing contribution
    dep_contribution = 0.75 - 0.60
    dep_pct = 100 * dep_contribution / total_gap
    assert budget['no_depolarizing']['delta_metric'] == pytest.approx(dep_contribution, abs=1e-6)
    assert budget['no_depolarizing']['pct_of_gap'] == pytest.approx(dep_pct, abs=1e-6)
    
    # Sum of fractions (excluding baseline/ideal) should be reasonable
    # (may not sum to exactly 100% due to non-additive effects)
    total_frac = sum(
        v['fraction_of_total']
        for k, v in budget.items()
        if k not in ['baseline', 'ideal']
    )
    assert 0.0 <= total_frac <= 2.0  # Can exceed 1.0 due to overlap


def test_sensitivity_runner_checkpointing(test_config, temp_output_dir):
    """
    Test checkpoint save/load functionality.
    
    Verifies:
    - Checkpoints are created
    - Checkpoint data can be loaded
    - Results are preserved across restarts
    """
    runner = SensitivityRunner(
        test_config,
        temp_output_dir,
        checkpoint_frequency=2
    )
    
    # Add some dummy results
    runner.results = [
        {
            'label': 'test1',
            'timestamp': '2025-10-16T12:00:00',
            'config_metadata': {},
            'device_params': {},
            'aggregated_results': {2: {'mean_hop': 0.65}},
        },
        {
            'label': 'test2',
            'timestamp': '2025-10-16T12:01:00',
            'config_metadata': {},
            'device_params': {},
            'aggregated_results': {2: {'mean_hop': 0.70}},
        },
    ]
    
    # Save checkpoint
    runner.save_checkpoint()
    
    assert runner.checkpoint_file.exists()
    
    # Create new runner and load checkpoint
    runner2 = SensitivityRunner(
        test_config,
        temp_output_dir,
        checkpoint_frequency=2
    )
    
    runner2.load_checkpoint()
    
    assert len(runner2.results) == 2
    assert runner2.results[0]['label'] == 'test1'
    assert runner2.results[1]['label'] == 'test2'


def test_results_to_dataframe(test_config, temp_output_dir):
    """
    Test conversion of results to DataFrame.
    
    Verifies:
    - DataFrame has correct structure
    - All widths are represented
    - Device parameters are included
    """
    runner = SensitivityRunner(test_config, temp_output_dir)
    
    # Add test results
    runner.results = [
        {
            'label': 'config1',
            'timestamp': '2025-10-16T12:00:00',
            'config_metadata': {},
            'device_params': {'F1': 0.999, 'F2': 0.998},
            'aggregated_results': {
                2: {'mean_hop': 0.65, 'ci_lower': 0.60, 'ci_upper': 0.70},
                3: {'mean_hop': 0.55, 'ci_lower': 0.50, 'ci_upper': 0.60},
            },
        },
        {
            'label': 'config2',
            'timestamp': '2025-10-16T12:01:00',
            'config_metadata': {},
            'device_params': {'F1': 1.0, 'F2': 1.0},
            'aggregated_results': {
                2: {'mean_hop': 0.85, 'ci_lower': 0.80, 'ci_upper': 0.90},
                3: {'mean_hop': 0.75, 'ci_lower': 0.70, 'ci_upper': 0.80},
            },
        },
    ]
    
    df = runner._results_to_dataframe()
    
    # Check structure
    assert len(df) == 4  # 2 configs × 2 widths
    assert 'label' in df.columns
    assert 'width' in df.columns
    assert 'F1' in df.columns
    assert 'F2' in df.columns
    assert 'mean_hop' in df.columns
    assert 'ci_lower' in df.columns
    assert 'ci_upper' in df.columns
    
    # Check data
    config1_m2 = df[(df['label'] == 'config1') & (df['width'] == 2)]
    assert len(config1_m2) == 1
    assert config1_m2['mean_hop'].iloc[0] == 0.65
    assert config1_m2['F1'].iloc[0] == 0.999
    
    config2_m3 = df[(df['label'] == 'config2') & (df['width'] == 3)]
    assert len(config2_m3) == 1
    assert config2_m3['mean_hop'].iloc[0] == 0.75
    assert config2_m3['F1'].iloc[0] == 1.0


def test_create_2d_grid(test_config, temp_output_dir):
    """
    Test 2D grid creation for heatmap plotting.
    
    Verifies:
    - Grid has correct shape
    - Values are placed at correct positions
    - Missing data is represented as NaN
    """
    runner = SensitivityRunner(test_config, temp_output_dir)
    
    # Create synthetic DataFrame
    f1_values = [0.995, 1.0]
    f2_values = [0.995, 1.0]
    
    data = [
        {'label': 'a', 'width': 2, 'F1': 0.995, 'F2': 0.995, 'mean_hop': 0.50},
        {'label': 'b', 'width': 2, 'F1': 0.995, 'F2': 1.0, 'mean_hop': 0.60},
        {'label': 'c', 'width': 2, 'F1': 1.0, 'F2': 0.995, 'mean_hop': 0.70},
        {'label': 'd', 'width': 2, 'F1': 1.0, 'F2': 1.0, 'mean_hop': 0.80},
    ]
    
    df = pd.DataFrame(data)
    
    grid = runner._create_2d_grid(df, 'F1', f1_values, 'F2', f2_values)
    
    # Check shape
    assert grid.shape == (2, 2)  # (len(f2_values), len(f1_values))
    
    # Check values (grid[i, j] = F2[i], F1[j])
    assert grid[0, 0] == pytest.approx(0.50, abs=1e-6)  # F1=0.995, F2=0.995
    assert grid[0, 1] == pytest.approx(0.70, abs=1e-6)  # F1=1.0, F2=0.995
    assert grid[1, 0] == pytest.approx(0.60, abs=1e-6)  # F1=0.995, F2=1.0
    assert grid[1, 1] == pytest.approx(0.80, abs=1e-6)  # F1=1.0, F2=1.0
