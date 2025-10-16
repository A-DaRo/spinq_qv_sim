"""
Production-quality QV campaign runner with resume capability.

Provides infrastructure for running full QV campaigns across multiple widths
with automatic state persistence, error recovery, and reproducibility guarantees.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import logging
import numpy as np

from spinq_qv.config.schemas import Config
from spinq_qv.io.storage import QVResultsWriter, QVResultsReader


logger = logging.getLogger(__name__)


class CampaignState:
    """
    State tracker for QV campaign progress.
    
    Stores which widths have been completed, failed, or are in progress
    to enable resumable campaigns.
    """
    
    def __init__(
        self,
        campaign_id: str,
        config: Config,
        output_dir: Path,
    ):
        """
        Initialize campaign state.
        
        Args:
            campaign_id: Unique identifier for this campaign
            config: Experiment configuration
            output_dir: Output directory for results
        """
        self.campaign_id = campaign_id
        self.config = config
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / f"campaign_{campaign_id}_state.json"
        
        # State tracking
        self.completed_widths: List[int] = []
        self.failed_widths: List[int] = []
        self.in_progress_width: Optional[int] = None
        self.start_time: Optional[str] = None
        self.last_update: Optional[str] = None
        
        # Results file path
        self.results_file = self.output_dir / f"campaign_{campaign_id}_results.h5"
    
    def save(self) -> None:
        """Save current state to JSON file."""
        self.last_update = datetime.now().isoformat()
        
        state_data = {
            'campaign_id': self.campaign_id,
            'completed_widths': self.completed_widths,
            'failed_widths': self.failed_widths,
            'in_progress_width': self.in_progress_width,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'results_file': str(self.results_file),
        }
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Campaign state saved to {self.state_file}")
    
    @classmethod
    def load(cls, state_file: Path, config: Config) -> 'CampaignState':
        """
        Load campaign state from JSON file.
        
        Args:
            state_file: Path to state JSON file
            config: Experiment configuration
        
        Returns:
            Loaded CampaignState instance
        """
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Create instance
        campaign_id = state_data['campaign_id']
        output_dir = state_file.parent
        
        state = cls(campaign_id, config, output_dir)
        
        # Restore state
        state.completed_widths = state_data.get('completed_widths', [])
        state.failed_widths = state_data.get('failed_widths', [])
        state.in_progress_width = state_data.get('in_progress_width')
        state.start_time = state_data.get('start_time')
        state.last_update = state_data.get('last_update')
        
        if 'results_file' in state_data:
            state.results_file = Path(state_data['results_file'])
        
        logger.info(f"Campaign state loaded from {state_file}")
        logger.info(f"  Completed widths: {state.completed_widths}")
        logger.info(f"  Failed widths: {state.failed_widths}")
        logger.info(f"  In progress: {state.in_progress_width}")
        
        return state
    
    def mark_started(self, width: int) -> None:
        """Mark a width as started."""
        self.in_progress_width = width
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
        self.save()
    
    def mark_completed(self, width: int) -> None:
        """Mark a width as completed."""
        if width not in self.completed_widths:
            self.completed_widths.append(width)
        self.in_progress_width = None
        self.save()
    
    def mark_failed(self, width: int, error: str) -> None:
        """Mark a width as failed."""
        if width not in self.failed_widths:
            self.failed_widths.append(width)
        self.in_progress_width = None
        logger.error(f"Width {width} failed: {error}")
        self.save()
    
    def get_pending_widths(self, all_widths: List[int]) -> List[int]:
        """
        Get list of widths that still need to be run.
        
        Args:
            all_widths: Complete list of widths to run
        
        Returns:
            List of pending widths (not completed or failed)
        """
        return [
            w for w in all_widths
            if w not in self.completed_widths and w not in self.failed_widths
        ]


class ProductionCampaignRunner:
    """
    High-level runner for production QV campaigns.
    
    Features:
    - Automatic state persistence for resumability
    - Per-width error recovery
    - Comprehensive logging and metadata tracking
    - HDF5 output with compression
    """
    
    def __init__(
        self,
        config: Config,
        output_dir: Path,
        campaign_id: Optional[str] = None,
        resume: bool = False,
    ):
        """
        Initialize campaign runner.
        
        Args:
            config: Experiment configuration
            output_dir: Output directory for results
            campaign_id: Unique campaign identifier (auto-generated if None)
            resume: If True, attempt to resume previous campaign
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or load campaign ID
        if resume:
            # Find most recent state file
            state_files = sorted(self.output_dir.glob("campaign_*_state.json"))
            if not state_files:
                raise FileNotFoundError(
                    f"No campaign state files found in {self.output_dir}"
                )
            
            latest_state = state_files[-1]
            self.state = CampaignState.load(latest_state, config)
            self.campaign_id = self.state.campaign_id
            
            logger.info(f"Resuming campaign {self.campaign_id}")
        
        else:
            # Create new campaign
            if campaign_id is None:
                campaign_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.campaign_id = campaign_id
            self.state = CampaignState(campaign_id, config, output_dir)
            
            logger.info(f"Starting new campaign {self.campaign_id}")
    
    def run(
        self,
        parallel: bool = False,
        n_workers: int = 4,
        enable_profiling: bool = False,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Run full QV campaign across all configured widths.
        
        Args:
            parallel: Enable parallel circuit execution
            n_workers: Number of parallel workers
            enable_profiling: Enable performance profiling
            max_retries: Maximum retry attempts for failed widths
        
        Returns:
            Dictionary with campaign summary
        """
        widths = self.config.simulation.widths
        pending_widths = self.state.get_pending_widths(widths)
        
        if not pending_widths:
            logger.info("All widths already completed!")
            return self._generate_summary()
        
        logger.info(f"Running campaign for widths: {pending_widths}")
        logger.info(f"  Total widths: {len(widths)}")
        logger.info(f"  Completed: {len(self.state.completed_widths)}")
        logger.info(f"  Failed: {len(self.state.failed_widths)}")
        logger.info(f"  Pending: {len(pending_widths)}")
        
        # Run each width
        for width in pending_widths:
            retry_count = 0
            success = False
            
            while retry_count <= max_retries and not success:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Running width m={width} (attempt {retry_count + 1}/{max_retries + 1})")
                    logger.info(f"{'='*60}\n")
                    
                    self.state.mark_started(width)
                    
                    # Run experiment for this width
                    self._run_single_width(
                        width,
                        parallel=parallel,
                        n_workers=n_workers,
                        enable_profiling=enable_profiling,
                    )
                    
                    # Mark as completed
                    self.state.mark_completed(width)
                    success = True
                    
                    logger.info(f"✓ Width m={width} completed successfully")
                
                except Exception as e:
                    retry_count += 1
                    logger.error(f"✗ Width m={width} failed (attempt {retry_count}): {e}")
                    
                    if retry_count > max_retries:
                        self.state.mark_failed(width, str(e))
                        logger.error(f"Max retries exceeded for width {width}, moving to next")
        
        # Generate final summary
        return self._generate_summary()
    
    def _run_single_width(
        self,
        width: int,
        parallel: bool,
        n_workers: int,
        enable_profiling: bool,
    ) -> None:
        """
        Run experiment for a single width.
        
        Args:
            width: Circuit width to run
            parallel: Enable parallel execution
            n_workers: Number of workers
            enable_profiling: Enable profiling
        """
        # Import here to avoid circular dependency
        from spinq_qv.experiments.run_qv import run_experiment
        
        # Create temporary config with single width
        config_dict = self.config.model_dump()
        config_dict['simulation']['widths'] = [width]
        temp_config = Config.model_validate(config_dict)
        
        # Create temporary output directory for this width
        width_output = self.output_dir / f"width_{width}"
        width_output.mkdir(exist_ok=True)
        
        # Run experiment
        aggregated_results = run_experiment(
            temp_config,
            width_output,
            seed=self.config.simulation.random_seed,
            parallel=parallel,
            n_workers=n_workers,
            enable_profiling=enable_profiling,
            return_aggregated=True,
        )
        
        # Append to main campaign results file
        self._append_to_campaign_results(width, aggregated_results)
    
    def _append_to_campaign_results(
        self,
        width: int,
        results: Dict[str, Any],
    ) -> None:
        """
        Append width results to campaign HDF5 file.
        
        Args:
            width: Circuit width
            results: Aggregated results dictionary
        """
        import h5py
        
        # Open campaign results file (create if doesn't exist)
        mode = 'a' if self.state.results_file.exists() else 'w'
        
        with h5py.File(self.state.results_file, mode) as f:
            # Create metadata group if first write
            if 'metadata' not in f:
                meta_group = f.create_group('metadata')
                meta_group.attrs['campaign_id'] = self.campaign_id
                meta_group.attrs['config'] = json.dumps(self.config.model_dump(), indent=2)
                # Only write start_time if it's set
                if self.state.start_time is not None:
                    meta_group.attrs['start_time'] = self.state.start_time
            
            # Create aggregated group if doesn't exist
            if 'aggregated' not in f:
                f.create_group('aggregated')
            
            # Write width results
            width_key = f"{width}"
            if width_key in f['aggregated']:
                # Remove existing data for this width (in case of retry)
                del f['aggregated'][width_key]
            
            agg_group = f['aggregated'].create_group(width_key)
            
            # Store results
            for key, value in results.items():
                if isinstance(value, (int, float, bool, str)):
                    agg_group.attrs[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    agg_group.attrs[key] = value.item()
                elif isinstance(value, np.ndarray):
                    agg_group.create_dataset(key, data=value, compression='gzip', compression_opts=4)
                elif isinstance(value, list):
                    agg_group.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
        
        logger.info(f"Results for width {width} appended to {self.state.results_file}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate campaign summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'campaign_id': self.campaign_id,
            'output_dir': str(self.output_dir),
            'results_file': str(self.state.results_file),
            'state_file': str(self.state.state_file),
            'start_time': self.state.start_time,
            'end_time': datetime.now().isoformat(),
            'completed_widths': self.state.completed_widths,
            'failed_widths': self.state.failed_widths,
            'total_widths': len(self.config.simulation.widths),
            'success_rate': len(self.state.completed_widths) / len(self.config.simulation.widths),
        }
        
        # Save summary to JSON
        summary_file = self.output_dir / f"campaign_{self.campaign_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("CAMPAIGN SUMMARY")
        logger.info("="*60)
        logger.info(f"Campaign ID: {self.campaign_id}")
        logger.info(f"Completed widths: {summary['completed_widths']}")
        logger.info(f"Failed widths: {summary['failed_widths']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Results file: {self.state.results_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info("="*60 + "\n")
        
        return summary
