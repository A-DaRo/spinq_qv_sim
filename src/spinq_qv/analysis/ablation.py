"""
Ablation studies and sensitivity analysis for QV experiments.

Enables systematic investigation of error sources by selectively
disabling different noise channels and varying parameters.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import itertools
import json

from spinq_qv.config import Config


class AblationStudy:
    """
    Orchestrate ablation experiments to identify dominant error sources.
    """
    
    def __init__(self, base_config: Config):
        """
        Initialize ablation study.
        
        Args:
            base_config: Baseline configuration to modify
        """
        self.base_config = base_config
    
    def create_ablated_config(
        self,
        disable_depolarizing: bool = False,
        disable_decoherence: bool = False,
        disable_coherent: bool = False,
        disable_readout_error: bool = False,
        disable_crosstalk: bool = False,
        disable_quasistatic: bool = False,
    ) -> Config:
        """
        Create config with selected noise sources disabled.
        
        Args:
            disable_depolarizing: Set gate fidelities to 1.0
            disable_decoherence: Set T1, T2 to infinity
            disable_coherent: Remove systematic rotations (ZZ coupling, over-rotation)
            disable_readout_error: Set readout fidelity to 1.0
            disable_crosstalk: Disable crosstalk effects
            disable_quasistatic: Disable quasi-static noise (1/f noise)
        
        Returns:
            Modified configuration
        """
        config_dict = self.base_config.model_dump()
        
        if disable_depolarizing:
            config_dict['device']['F1'] = 1.0
            config_dict['device']['F2'] = 1.0
        
        if disable_decoherence:
            config_dict['device']['T1'] = 1e6  # Very long (effectively infinite)
            config_dict['device']['T2'] = 1e6
            config_dict['device']['T2_star'] = 1e6
        
        if disable_coherent:
            # Mark in metadata to skip coherent error application in noise model
            if 'noise_toggles' not in config_dict['metadata']:
                config_dict['metadata']['noise_toggles'] = {}
            config_dict['metadata']['noise_toggles']['coherent_errors'] = False
        
        if disable_readout_error:
            config_dict['device']['state_prep_error'] = 0.0
            config_dict['device']['meas_error_1given0'] = 0.0
            config_dict['device']['meas_error_0given1'] = 0.0
        
        if disable_crosstalk:
            if 'noise_toggles' not in config_dict['metadata']:
                config_dict['metadata']['noise_toggles'] = {}
            config_dict['metadata']['noise_toggles']['crosstalk'] = False
        
        if disable_quasistatic:
            if 'noise_toggles' not in config_dict['metadata']:
                config_dict['metadata']['noise_toggles'] = {}
            config_dict['metadata']['noise_toggles']['quasistatic'] = False
        
        # Update metadata with ablation flags
        ablation_flags = {
            'depolarizing_disabled': disable_depolarizing,
            'decoherence_disabled': disable_decoherence,
            'coherent_disabled': disable_coherent,
            'readout_error_disabled': disable_readout_error,
            'crosstalk_disabled': disable_crosstalk,
            'quasistatic_disabled': disable_quasistatic,
        }
        
        config_dict['metadata']['ablation'] = ablation_flags
        
        return Config(**config_dict)
    
    def generate_ablation_sweep(self) -> List[Tuple[str, Config]]:
        """
        Generate full ablation sweep (all major error sources).
        
        Returns:
            List of (label, config) tuples
        """
        ablations = []
        
        # Baseline (all noise)
        ablations.append(("baseline", self.base_config))
        
        # Single ablations - each error source disabled individually
        ablations.append((
            "no_depolarizing",
            self.create_ablated_config(disable_depolarizing=True)
        ))
        ablations.append((
            "no_decoherence",
            self.create_ablated_config(disable_decoherence=True)
        ))
        ablations.append((
            "no_coherent",
            self.create_ablated_config(disable_coherent=True)
        ))
        ablations.append((
            "no_readout",
            self.create_ablated_config(disable_readout_error=True)
        ))
        ablations.append((
            "no_crosstalk",
            self.create_ablated_config(disable_crosstalk=True)
        ))
        ablations.append((
            "no_quasistatic",
            self.create_ablated_config(disable_quasistatic=True)
        ))
        
        # Pairwise ablations - most impactful combinations
        ablations.append((
            "no_dep_no_decoh",
            self.create_ablated_config(
                disable_depolarizing=True,
                disable_decoherence=True,
            )
        ))
        ablations.append((
            "no_dep_no_readout",
            self.create_ablated_config(
                disable_depolarizing=True,
                disable_readout_error=True,
            )
        ))
        
        # Ideal (no noise)
        ablations.append((
            "ideal",
            self.create_ablated_config(
                disable_depolarizing=True,
                disable_decoherence=True,
                disable_coherent=True,
                disable_readout_error=True,
                disable_crosstalk=True,
                disable_quasistatic=True,
            )
        ))
        
        return ablations


class SensitivityAnalysis:
    """
    Parameter sensitivity analysis via grid sweeps.
    """
    
    def __init__(self, base_config: Config):
        """
        Initialize sensitivity analysis.
        
        Args:
            base_config: Baseline configuration
        """
        self.base_config = base_config
    
    def sweep_parameter(
        self,
        parameter_path: List[str],
        values: List[float],
        labels: Optional[List[str]] = None,
    ) -> List[Tuple[str, Config]]:
        """
        Create configs sweeping a single parameter.
        
        Args:
            parameter_path: Path to parameter (e.g., ['device', 'F1'])
            values: List of values to sweep
            labels: Optional labels for each value
        
        Returns:
            List of (label, config) tuples
        """
        if labels is None:
            param_name = parameter_path[-1]
            labels = [f"{param_name}={v}" for v in values]
        
        configs = []
        
        for label, value in zip(labels, values):
            config_dict = self.base_config.model_dump()
            
            # Navigate to parameter
            target = config_dict
            for key in parameter_path[:-1]:
                target = target[key]
            
            target[parameter_path[-1]] = value
            
            # Update metadata
            config_dict['metadata']['sensitivity_param'] = '/'.join(parameter_path)
            config_dict['metadata']['sensitivity_value'] = value
            
            configs.append((label, Config(**config_dict)))
        
        return configs
    
    def sweep_2d_grid(
        self,
        param1_path: List[str],
        param1_values: List[float],
        param2_path: List[str],
        param2_values: List[float],
    ) -> List[Tuple[str, Config]]:
        """
        Create 2D parameter grid.
        
        Args:
            param1_path: Path to first parameter
            param1_values: Values for first parameter
            param2_path: Path to second parameter
            param2_values: Values for second parameter
        
        Returns:
            List of (label, config) tuples for all grid points
        """
        configs = []
        
        param1_name = param1_path[-1]
        param2_name = param2_path[-1]
        
        for v1, v2 in itertools.product(param1_values, param2_values):
            config_dict = self.base_config.model_dump()
            
            # Set first parameter
            target1 = config_dict
            for key in param1_path[:-1]:
                target1 = target1[key]
            target1[param1_path[-1]] = v1
            
            # Set second parameter
            target2 = config_dict
            for key in param2_path[:-1]:
                target2 = target2[key]
            target2[param2_path[-1]] = v2
            
            label = f"{param1_name}={v1}_{param2_name}={v2}"
            
            config_dict['metadata']['grid_params'] = {
                param1_name: v1,
                param2_name: v2,
            }
            
            configs.append((label, Config(**config_dict)))
        
        return configs


def compute_error_budget(
    ablation_results: Dict[str, Dict[str, Any]],
    baseline_key: str = "baseline",
    ideal_key: str = "ideal",
    metric: str = 'mean_hop',
) -> Dict[str, Dict[str, float]]:
    """
    Compute error budget from ablation study results.
    
    Error contribution for each source is estimated as the difference
    in HOP when that source is removed (improvement over baseline).
    
    Args:
        ablation_results: Dictionary mapping ablation label -> aggregated results
        baseline_key: Key for baseline (all noise) results
        ideal_key: Key for ideal (no noise) results
        metric: Metric to compute budget for (default: 'mean_hop')
    
    Returns:
        Dictionary mapping error source -> {delta_metric, fraction_of_total, pct_of_gap}
    """
    baseline_val = ablation_results[baseline_key][metric]
    ideal_val = ablation_results[ideal_key][metric]
    
    total_gap = ideal_val - baseline_val
    
    budget = {}
    
    # Single-source contributions
    for ablation_label, results in ablation_results.items():
        if ablation_label not in [baseline_key, ideal_key]:
            # Improvement when this source is removed
            ablated_val = results[metric]
            contribution = ablated_val - baseline_val
            
            budget[ablation_label] = {
                'delta_metric': float(contribution),
                'fraction_of_total': float(contribution / total_gap if total_gap != 0 else 0),
                'pct_of_gap': float(100 * contribution / total_gap if total_gap != 0 else 0),
                'ablated_value': float(ablated_val),
            }
    
    # Add baseline and ideal for reference
    budget[baseline_key] = {
        'delta_metric': 0.0,
        'fraction_of_total': 0.0,
        'pct_of_gap': 0.0,
        'ablated_value': float(baseline_val),
    }
    budget[ideal_key] = {
        'delta_metric': float(total_gap),
        'fraction_of_total': 1.0,
        'pct_of_gap': 100.0,
        'ablated_value': float(ideal_val),
    }
    
    return budget


def export_error_budget_to_json(
    budget: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Export error budget to JSON file.
    
    Args:
        budget: Error budget dictionary from compute_error_budget()
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(budget, f, indent=2)


def summarize_error_budget(
    budget: Dict[str, Dict[str, float]],
    baseline_key: str = "baseline",
    ideal_key: str = "ideal",
) -> str:
    """
    Create human-readable summary of error budget.
    
    Args:
        budget: Error budget dictionary
        baseline_key: Key for baseline results
        ideal_key: Key for ideal results
    
    Returns:
        Formatted string summary
    """
    # Filter out baseline and ideal
    contributions = {
        k: v for k, v in budget.items()
        if k not in [baseline_key, ideal_key]
    }
    
    # Sort by contribution (largest first)
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: x[1]['pct_of_gap'],
        reverse=True
    )
    
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR BUDGET SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Baseline HOP: {budget[baseline_key]['ablated_value']:.4f}")
    lines.append(f"Ideal HOP:    {budget[ideal_key]['ablated_value']:.4f}")
    lines.append(f"Total Gap:    {budget[ideal_key]['delta_metric']:.4f}")
    lines.append("")
    lines.append("Contributions (sorted by impact):")
    lines.append("-" * 60)
    
    for label, data in sorted_items:
        lines.append(
            f"  {label:25s}: {data['pct_of_gap']:6.2f}% "
            f"(delta={data['delta_metric']:+.4f}, "
            f"ablated HOP={data['ablated_value']:.4f})"
        )
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

