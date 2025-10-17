"""
Campaign Executor

Executes QV simulations for all configurations in a campaign.
Handles parallel execution, progress tracking, and error recovery.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinq_qv.config.schemas import Config


class CampaignExecutor:
    """Execute QV campaign across multiple configurations."""
    
    def __init__(
        self,
        campaign_configs: Dict[str, Config],
        output_dir: Path,
        parallel: bool = False,
    ):
        """
        Initialize campaign executor.
        
        Args:
            campaign_configs: Dictionary mapping config_name -> Config
            output_dir: Base directory for campaign output
            parallel: Enable parallel execution (experimental)
        """
        self.campaign_configs = campaign_configs
        self.output_dir = Path(output_dir)
        self.parallel = parallel
        
        # Create results directory
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result storage
        self.campaign_results = {}
        self.campaign_metadata = {}
    
    def run_campaign(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Execute full campaign.
        
        Args:
            progress_callback: Optional callback for progress updates
                Signature: callback(current, total, config_name, elapsed_time)
        
        Returns:
            Dictionary mapping config_name -> {width -> statistics}
        """
        config_names = sorted(self.campaign_configs.keys())
        total_configs = len(config_names)
        
        start_time = time.time()
        
        for idx, config_name in enumerate(config_names, 1):
            config = self.campaign_configs[config_name]
            
            # Update progress
            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(idx - 1, total_configs, config_name, elapsed)
            
            # Run single configuration
            try:
                results = self._run_single_config(config_name, config)
                self.campaign_results[config_name] = results
                
                # Save intermediate results
                self._save_results(config_name, results)
                
            except Exception as e:
                print(f"\n[ERROR] Failed to run {config_name}: {e}")
                self.campaign_results[config_name] = None
                continue
        
        # Final progress update
        if progress_callback:
            elapsed = time.time() - start_time
            progress_callback(total_configs, total_configs, "Complete", elapsed)
        
        # Save final campaign results
        self._save_campaign_summary()
        
        return self.campaign_results
    
    def _run_single_config(
        self,
        config_name: str,
        config: Config,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run QV experiment for a single configuration.
        
        Args:
            config_name: Name of configuration
            config: Configuration object
        
        Returns:
            Dictionary mapping width -> statistics
        """
        # Import experiment runner
        from spinq_qv.experiments.run_qv import run_experiment
        
        # Create temporary output directory for this config
        config_output_dir = self.output_dir / "results" / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get seed
        seed = config.simulation.random_seed or 42
        
        # Run experiment and get aggregated results
        aggregated_results = run_experiment(
            config=config,
            output_dir=config_output_dir,
            seed=seed,
            return_aggregated=True,  # Return aggregated results
            parallel=self.parallel,
            n_workers=4,
            enable_profiling=False,
        )
        
        return aggregated_results
    
    def _save_results(self, config_name: str, results: Dict[int, Dict[str, Any]]) -> None:
        """Save results for a single configuration."""
        result_file = self.results_dir / f"{config_name}_results.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_campaign_summary(self) -> None:
        """Save campaign summary with all results."""
        summary_file = self.output_dir / "campaign_results.json"
        
        # Remove None results (failed configs)
        valid_results = {
            name: results
            for name, results in self.campaign_results.items()
            if results is not None
        }
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "n_configurations": len(self.campaign_configs),
            "n_successful": len(valid_results),
            "n_failed": len(self.campaign_results) - len(valid_results),
            "results": valid_results,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[✓] Campaign summary saved to: {summary_file}")


# Import numpy here to avoid issues
import numpy as np
