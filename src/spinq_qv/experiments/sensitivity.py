"""
High-level orchestrator for sensitivity and ablation experiments.

Manages parameter sweeps, grid runs, checkpointing, and result aggregation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from spinq_qv.config import Config
from spinq_qv.analysis.ablation import (
    AblationStudy,
    SensitivityAnalysis,
    compute_error_budget,
    export_error_budget_to_json,
    summarize_error_budget,
)


logger = logging.getLogger(__name__)


class SensitivityRunner:
    """
    Orchestrate sensitivity experiments with checkpointing and aggregation.
    """
    
    def __init__(
        self,
        base_config: Config,
        output_dir: Path,
        checkpoint_frequency: int = 5,
    ):
        """
        Initialize sensitivity runner.
        
        Args:
            base_config: Baseline configuration
            output_dir: Directory for outputs
            checkpoint_frequency: Save checkpoint every N runs
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = checkpoint_frequency
        
        self.results: List[Dict[str, Any]] = []
        self.checkpoint_file = self.output_dir / "checkpoint.json"
    
    def load_checkpoint(self) -> None:
        """Load results from checkpoint file if it exists."""
        if self.checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.results = data.get('results', [])
            logger.info(f"Loaded {len(self.results)} previous results")
    
    def save_checkpoint(self) -> None:
        """Save current results to checkpoint file."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'n_results': len(self.results),
            'results': self.results,
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved with {len(self.results)} results")
    
    def run_single_config(
        self,
        label: str,
        config: Config,
        qv_runner_func,
        **runner_kwargs
    ) -> Dict[str, Any]:
        """
        Run QV experiment for a single configuration.
        
        Args:
            label: Label for this configuration
            config: Configuration to run
            qv_runner_func: Function to run QV experiment (from run_qv.py)
            **runner_kwargs: Additional arguments for runner
        
        Returns:
            Dictionary with label, config metadata, and aggregated results
        """
        logger.info(f"Running configuration: {label}")
        
        # Run QV experiment
        aggregated_results = qv_runner_func(config, **runner_kwargs)
        
        # Extract key metrics
        result = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'config_metadata': config.metadata,
            'device_params': config.device.model_dump(),
            'aggregated_results': aggregated_results,
        }
        
        return result
    
    def run_ablation_sweep(
        self,
        qv_runner_func,
        **runner_kwargs
    ) -> pd.DataFrame:
        """
        Run full ablation sweep.
        
        Args:
            qv_runner_func: Function to run QV experiment
            **runner_kwargs: Additional arguments for runner
        
        Returns:
            DataFrame with all ablation results
        """
        self.load_checkpoint()
        
        ablation = AblationStudy(self.base_config)
        configs = ablation.generate_ablation_sweep()
        
        logger.info(f"Starting ablation sweep with {len(configs)} configurations")
        
        # Track which configs have been run
        completed_labels = {r['label'] for r in self.results}
        
        for i, (label, config) in enumerate(configs):
            if label in completed_labels:
                logger.info(f"Skipping {label} (already completed)")
                continue
            
            result = self.run_single_config(label, config, qv_runner_func, **runner_kwargs)
            self.results.append(result)
            
            # Checkpoint periodically
            if (i + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Convert to DataFrame
        df = self._results_to_dataframe()
        
        # Save CSV
        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        return df
    
    def run_parameter_sweep(
        self,
        parameter_path: List[str],
        values: List[float],
        qv_runner_func,
        labels: Optional[List[str]] = None,
        **runner_kwargs
    ) -> pd.DataFrame:
        """
        Run 1D parameter sweep.
        
        Args:
            parameter_path: Path to parameter (e.g., ['device', 'F2'])
            values: List of values to sweep
            qv_runner_func: Function to run QV experiment
            labels: Optional custom labels
            **runner_kwargs: Additional arguments for runner
        
        Returns:
            DataFrame with sweep results
        """
        self.load_checkpoint()
        
        sensitivity = SensitivityAnalysis(self.base_config)
        configs = sensitivity.sweep_parameter(parameter_path, values, labels)
        
        param_name = parameter_path[-1]
        logger.info(f"Starting {param_name} sweep with {len(values)} values")
        
        completed_labels = {r['label'] for r in self.results}
        
        for i, (label, config) in enumerate(configs):
            if label in completed_labels:
                logger.info(f"Skipping {label} (already completed)")
                continue
            
            result = self.run_single_config(label, config, qv_runner_func, **runner_kwargs)
            self.results.append(result)
            
            if (i + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()
        
        self.save_checkpoint()
        
        df = self._results_to_dataframe()
        csv_path = self.output_dir / f"{param_name}_sweep.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        return df
    
    def run_2d_grid(
        self,
        param1_path: List[str],
        param1_values: List[float],
        param2_path: List[str],
        param2_values: List[float],
        qv_runner_func,
        **runner_kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Run 2D parameter grid sweep.
        
        Args:
            param1_path: Path to first parameter
            param1_values: Values for first parameter
            param2_path: Path to second parameter
            param2_values: Values for second parameter
            qv_runner_func: Function to run QV experiment
            **runner_kwargs: Additional arguments for runner
        
        Returns:
            Tuple of (DataFrame with all results, 2D grid array for heatmap)
        """
        self.load_checkpoint()
        
        sensitivity = SensitivityAnalysis(self.base_config)
        configs = sensitivity.sweep_2d_grid(
            param1_path, param1_values,
            param2_path, param2_values
        )
        
        param1_name = param1_path[-1]
        param2_name = param2_path[-1]
        logger.info(
            f"Starting 2D grid: {param1_name} x {param2_name} "
            f"({len(param1_values)} x {len(param2_values)} = {len(configs)} points)"
        )
        
        completed_labels = {r['label'] for r in self.results}
        
        for i, (label, config) in enumerate(configs):
            if label in completed_labels:
                logger.info(f"Skipping {label} (already completed)")
                continue
            
            result = self.run_single_config(label, config, qv_runner_func, **runner_kwargs)
            self.results.append(result)
            
            if (i + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()
        
        self.save_checkpoint()
        
        df = self._results_to_dataframe()
        
        # Create 2D grid for heatmap
        grid = self._create_2d_grid(
            df, param1_name, param1_values, param2_name, param2_values
        )
        
        csv_path = self.output_dir / f"{param1_name}_vs_{param2_name}_grid.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Save grid as numpy array
        grid_path = self.output_dir / f"{param1_name}_vs_{param2_name}_grid.npy"
        np.save(grid_path, grid)
        
        return df, grid
    
    def compute_and_save_error_budget(
        self,
        metric: str = 'mean_hop',
        baseline_key: str = 'baseline',
        ideal_key: str = 'ideal',
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute error budget from ablation results.
        
        Args:
            metric: Metric to compute budget for
            baseline_key: Label for baseline configuration
            ideal_key: Label for ideal configuration
        
        Returns:
            Error budget dictionary
        """
        # Convert results to format expected by compute_error_budget
        ablation_results = {}
        for result in self.results:
            label = result['label']
            agg_results = result['aggregated_results']
            
            # Extract metrics for each width (use average across widths)
            mean_hops = []
            for width_data in agg_results.values():
                if isinstance(width_data, dict) and metric in width_data:
                    mean_hops.append(width_data[metric])
            
            if mean_hops:
                ablation_results[label] = {
                    metric: np.mean(mean_hops)
                }
        
        budget = compute_error_budget(
            ablation_results,
            baseline_key=baseline_key,
            ideal_key=ideal_key,
            metric=metric
        )
        
        # Save to JSON
        budget_path = self.output_dir / "error_budget.json"
        export_error_budget_to_json(budget, budget_path)
        logger.info(f"Saved error budget to {budget_path}")
        
        # Save human-readable summary
        summary = summarize_error_budget(budget, baseline_key, ideal_key)
        summary_path = self.output_dir / "error_budget_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        logger.info(f"Saved error budget summary to {summary_path}")
        
        print("\n" + summary)
        
        return budget
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results list to pandas DataFrame."""
        rows = []
        
        for result in self.results:
            label = result['label']
            device_params = result['device_params']
            
            # Flatten aggregated results by width
            for width, width_data in result['aggregated_results'].items():
                if isinstance(width_data, dict):
                    row = {
                        'label': label,
                        'width': width,
                        **device_params,
                        **width_data,
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_2d_grid(
        self,
        df: pd.DataFrame,
        param1_name: str,
        param1_values: List[float],
        param2_name: str,
        param2_values: List[float],
        metric: str = 'mean_hop',
    ) -> np.ndarray:
        """
        Create 2D grid array from DataFrame for heatmap plotting.
        
        Args:
            df: Results DataFrame
            param1_name: Name of first parameter (x-axis)
            param1_values: Values for first parameter
            param2_name: Name of second parameter (y-axis)
            param2_values: Values for second parameter
            metric: Metric to plot (default: 'mean_hop')
        
        Returns:
            2D numpy array with shape (len(param2_values), len(param1_values))
        """
        grid = np.full((len(param2_values), len(param1_values)), np.nan)
        
        for i, v2 in enumerate(param2_values):
            for j, v1 in enumerate(param1_values):
                # Find matching rows
                mask = (
                    (np.abs(df[param1_name] - v1) < 1e-10) &
                    (np.abs(df[param2_name] - v2) < 1e-10)
                )
                
                matching = df[mask]
                if not matching.empty and metric in matching.columns:
                    # Average over widths if multiple
                    grid[i, j] = matching[metric].mean()
        
        return grid


def create_default_param_ranges(config: Config) -> Dict[str, List[float]]:
    """
    Create default parameter ranges for sensitivity sweeps.
    
    Args:
        config: Base configuration
    
    Returns:
        Dictionary mapping parameter names to sweep ranges
    """
    device = config.device
    
    ranges = {
        'F1': np.linspace(0.995, 1.0, 4).tolist(),
        'F2': np.linspace(0.990, 1.0, 4).tolist(),
        'T1': np.logspace(-1, 0, 4).tolist(),  # 0.1s to 1s
        'T2': np.logspace(-5, -4, 4).tolist(),  # 10µs to 100µs
        'T2_star': np.logspace(-6, -5, 4).tolist(),  # 1µs to 10µs
    }
    
    return ranges
