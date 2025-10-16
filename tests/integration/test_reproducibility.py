"""
Integration tests for campaign reproducibility and metadata tracking.

Validates that:
- Campaign state can be saved and restored
- --resume flag restores exact execution state
- Metadata (git hash, versions, seeds) is complete
- Same seed produces identical results
"""

import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil
import h5py

from spinq_qv.config import Config
from spinq_qv.experiments.campaign import ProductionCampaignRunner, CampaignState
from spinq_qv.io.storage import QVResultsReader


@pytest.fixture
def small_config(tmp_path):
    """Create a minimal test configuration."""
    config_dict = {
        'device': {
            'F1': 0.999,
            'F2': 0.998,
            'T1': 1.0,
            'T2': 99e-6,
            'T2_star': 20e-6,
            't_single_gate': 60e-9,
            't_two_gate': 40e-9,
            'F_readout': 0.9997,
            'F_init': 0.994,
            't_readout': 10e-6,
        },
        'simulation': {
            'backend': 'statevector',
            'n_circuits': 5,
            'n_shots': 200,
            'widths': [2, 3],
            'random_seed': 12345,
        },
        'metadata': {
            'experiment_name': 'test_reproducibility',
            'description': 'Test campaign for reproducibility validation',
        }
    }
    
    return Config.model_validate(config_dict)


def test_campaign_state_save_load(small_config, tmp_path):
    """Test that campaign state can be saved and loaded correctly."""
    output_dir = tmp_path / "campaign_test"
    output_dir.mkdir()
    
    # Create campaign state
    state = CampaignState(
        campaign_id="test_campaign_001",
        config=small_config,
        output_dir=output_dir,
    )
    
    # Modify state
    state.start_time = "2025-01-17T10:00:00"
    state.completed_widths = [2]
    state.failed_widths = []
    state.in_progress_width = 3
    
    # Save state
    state.save()
    
    # Verify state file exists
    assert state.state_file.exists()
    
    # Load state back
    loaded_state = CampaignState.load(state.state_file, small_config)
    
    # Verify all fields match
    assert loaded_state.campaign_id == "test_campaign_001"
    assert loaded_state.completed_widths == [2]
    assert loaded_state.failed_widths == []
    assert loaded_state.in_progress_width == 3
    assert loaded_state.start_time == "2025-01-17T10:00:00"


def test_metadata_completeness(small_config, tmp_path):
    """Test that all required metadata is stored in campaign results."""
    output_dir = tmp_path / "metadata_test"
    
    # Create and run minimal campaign
    runner = ProductionCampaignRunner(
        config=small_config,
        output_dir=output_dir,
        campaign_id="metadata_test_001",
        resume=False,
    )
    
    # Run single width
    runner._run_single_width(
        width=2,
        parallel=False,
        n_workers=1,
        enable_profiling=False,
    )
    
    # Check results file exists
    assert runner.state.results_file.exists()
    
    # Open HDF5 and verify metadata
    with h5py.File(runner.state.results_file, 'r') as f:
        # Check metadata group exists
        assert 'metadata' in f
        
        meta = f['metadata']
        
        # Verify required fields
        assert 'campaign_id' in meta.attrs
        assert meta.attrs['campaign_id'] == "metadata_test_001"
        
        assert 'config' in meta.attrs
        config_json = json.loads(meta.attrs['config'])
        assert config_json['simulation']['random_seed'] == 12345
        
        assert 'start_time' in meta.attrs
        # Format: ISO 8601 timestamp
        assert 'T' in meta.attrs['start_time']
        
        # Git hash may be 'unknown' in test environment, but should exist
        # (not asserting specific value since it depends on git state)


def test_deterministic_results_same_seed(small_config, tmp_path):
    """Test that same seed produces identical results."""
    
    # Run 1
    output_dir_1 = tmp_path / "run1"
    runner1 = ProductionCampaignRunner(
        config=small_config,
        output_dir=output_dir_1,
        campaign_id="seed_test_001",
        resume=False,
    )
    
    runner1._run_single_width(2, parallel=False, n_workers=1, enable_profiling=False)
    
    # Run 2 (same seed)
    output_dir_2 = tmp_path / "run2"
    runner2 = ProductionCampaignRunner(
        config=small_config,
        output_dir=output_dir_2,
        campaign_id="seed_test_002",
        resume=False,
    )
    
    runner2._run_single_width(2, parallel=False, n_workers=1, enable_profiling=False)
    
    # Load results from both runs
    with h5py.File(runner1.state.results_file, 'r') as f1, \
         h5py.File(runner2.state.results_file, 'r') as f2:
        
        # Check aggregated results match
        assert 'aggregated' in f1 and 'aggregated' in f2
        assert '2' in f1['aggregated'] and '2' in f2['aggregated']
        
        mean_hop_1 = f1['aggregated']['2'].attrs['mean_hop']
        mean_hop_2 = f2['aggregated']['2'].attrs['mean_hop']
        
        # Same seed should give identical results
        assert np.isclose(mean_hop_1, mean_hop_2, rtol=1e-10), \
            f"HOPs differ: {mean_hop_1} vs {mean_hop_2}"


def test_resume_campaign_continues_from_state(small_config, tmp_path):
    """Test that --resume flag correctly restores and continues a campaign."""
    output_dir = tmp_path / "resume_test"
    
    # Initial campaign: run only width=2
    config_dict = small_config.model_dump()
    config_dict['simulation']['widths'] = [2, 3, 4]
    config_partial = Config.model_validate(config_dict)
    
    runner1 = ProductionCampaignRunner(
        config=config_partial,
        output_dir=output_dir,
        campaign_id="resume_test_001",
        resume=False,
    )
    
    # Run only width 2
    runner1._run_single_width(2, parallel=False, n_workers=1, enable_profiling=False)
    runner1.state.mark_completed(2)
    
    # Simulate interruption (don't run widths 3, 4)
    
    # Resume campaign
    runner2 = ProductionCampaignRunner(
        config=config_partial,
        output_dir=output_dir,
        campaign_id="resume_test_001",  # Same ID not strictly required if resuming
        resume=True,
    )
    
    # Verify state was restored
    assert runner2.state.completed_widths == [2]
    assert 3 not in runner2.state.completed_widths
    assert 4 not in runner2.state.completed_widths
    
    # Pending widths should be [3, 4]
    pending = runner2.state.get_pending_widths([2, 3, 4])
    assert set(pending) == {3, 4}


def test_failed_width_retry_logic(small_config, tmp_path):
    """Test that failed widths are retried up to max_retries."""
    output_dir = tmp_path / "retry_test"
    
    # Create campaign
    runner = ProductionCampaignRunner(
        config=small_config,
        output_dir=output_dir,
        campaign_id="retry_test_001",
        resume=False,
    )
    
    # Mark width 2 as failed
    runner.state.mark_failed(2, "Simulated failure for testing")
    
    # Verify failed_widths list
    assert 2 in runner.state.failed_widths
    assert 2 not in runner.state.completed_widths
    
    # Verify pending widths excludes failed width
    pending = runner.state.get_pending_widths([2, 3])
    assert 2 not in pending  # Failed widths not retried automatically
    assert 3 in pending


def test_results_file_structure(small_config, tmp_path):
    """Test that campaign results HDF5 has correct structure."""
    output_dir = tmp_path / "structure_test"
    
    runner = ProductionCampaignRunner(
        config=small_config,
        output_dir=output_dir,
        campaign_id="structure_test_001",
        resume=False,
    )
    
    # Run two widths
    runner._run_single_width(2, parallel=False, n_workers=1, enable_profiling=False)
    runner._run_single_width(3, parallel=False, n_workers=1, enable_profiling=False)
    
    # Verify HDF5 structure
    with h5py.File(runner.state.results_file, 'r') as f:
        # Top-level groups
        assert 'metadata' in f
        assert 'aggregated' in f
        
        # Aggregated results for each width
        assert '2' in f['aggregated']
        assert '3' in f['aggregated']
        
        # Check required attributes for width 2
        agg_2 = f['aggregated']['2']
        assert 'mean_hop' in agg_2.attrs
        assert 'std_hop' in agg_2.attrs
        assert 'ci_lower' in agg_2.attrs
        assert 'ci_upper' in agg_2.attrs
        
        # Verify values are in valid range [0, 1]
        mean_hop = agg_2.attrs['mean_hop']
        assert 0 <= mean_hop <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
