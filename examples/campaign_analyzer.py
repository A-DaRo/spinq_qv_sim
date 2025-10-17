"""
Campaign Analyzer

Analyzes campaign results and generates comprehensive visualizations
showing parameter effects on Quantum Volume performance.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinq_qv.analysis.plots import (
    save_all_campaign_plots,
    plot_campaign_comparison,
    plot_parameter_correlation_matrix,
    plot_3d_parameter_surface,
)


class CampaignAnalyzer:
    """Analyze and visualize campaign results."""
    
    def __init__(
        self,
        campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
        campaign_configs: Dict[str, Any],
        sweep_params: Dict[str, List[float]],
        output_dir: Path,
    ):
        """
        Initialize campaign analyzer.
        
        Args:
            campaign_results: Results dict mapping config_name -> {width -> stats}
            campaign_configs: Config dict mapping config_name -> Config
            sweep_params: Dict mapping parameter_name -> list of values
            output_dir: Base output directory
        """
        self.campaign_results = campaign_results
        self.campaign_configs = campaign_configs
        self.sweep_params = sweep_params
        self.output_dir = Path(output_dir)
        
        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create analysis directory
        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_analyses(self) -> None:
        """Generate all campaign analyses and plots."""
        print("\n[1/7] Generating campaign overview plots...")
        self._generate_overview_plots()
        
        print("[2/7] Generating parameter-specific plots...")
        self._generate_parameter_plots()
        
        print("[3/7] Analyzing parameter sensitivities...")
        self._analyze_parameter_sensitivity()
        
        print("[4/7] Generating correlation matrices...")
        self._generate_correlation_analysis()
        
        print("[5/7] Creating parameter sweep summaries...")
        self._generate_sweep_summaries()
        
        print("[6/7] Generating 3D parameter surfaces...")
        self._generate_3d_surfaces()
        
        print("[7/7] Creating HTML report...")
        self._generate_html_report()
        
        print("\n[✓] All analyses complete!")
    
    def _generate_overview_plots(self) -> None:
        """Generate standard campaign plots."""
        # Convert configs to dict format
        configs_dict = {
            name: config.model_dump() if hasattr(config, 'model_dump') else config
            for name, config in self.campaign_configs.items()
        }
        
        save_all_campaign_plots(
            self.campaign_results,
            campaign_configs=configs_dict,
            output_dir=self.plots_dir / "overview",
            formats=["png"],
        )
    
    def _generate_parameter_plots(self) -> None:
        """Generate plots for each swept parameter."""
        for param_name, param_values in self.sweep_params.items():
            # Filter results for this parameter
            param_results = self._filter_results_by_param(param_name)
            
            if not param_results:
                continue
            
            # Create parameter-specific directory
            param_dir = self.plots_dir / f"param_{param_name}"
            param_dir.mkdir(exist_ok=True)
            
            # Plot HOP vs parameter value
            self._plot_hop_vs_parameter(
                param_name,
                param_results,
                output_path=param_dir / f"{param_name}_vs_hop.png"
            )
            
            # Plot achieved QV vs parameter value
            self._plot_qv_vs_parameter(
                param_name,
                param_results,
                output_path=param_dir / f"{param_name}_vs_qv.png"
            )
    
    def _generate_correlation_analysis(self) -> None:
        """Generate parameter correlation analyses."""
        configs_dict = {
            name: config.model_dump() if hasattr(config, 'model_dump') else config
            for name, config in self.campaign_configs.items()
        }
        
        plot_parameter_correlation_matrix(
            self.campaign_results,
            configs_dict,
            output_path=self.plots_dir / "parameter_correlations.png",
        )
    
    def _analyze_parameter_sensitivity(self) -> None:
        """Compute and save parameter sensitivity metrics."""
        sensitivities = {}
        
        for param_name in self.sweep_params.keys():
            param_results = self._filter_results_by_param(param_name)
            
            if len(param_results) < 2:
                continue
            
            # Extract parameter values and achieved widths
            param_vals = []
            achieved_widths = []
            
            for config_name, results in param_results.items():
                # Get parameter value
                config = self.campaign_configs[config_name]
                if hasattr(config, 'device'):
                    param_val = getattr(config.device, param_name)
                else:
                    param_val = config['device'][param_name]
                param_vals.append(param_val)
                
                # Get achieved width
                passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
                max_width = max(passing_widths) if passing_widths else 0
                achieved_widths.append(max_width)
            
            # Compute sensitivity (correlation)
            if len(set(param_vals)) > 1 and len(set(achieved_widths)) > 1:
                correlation = np.corrcoef(param_vals, achieved_widths)[0, 1]
            else:
                correlation = 0.0
            
            # Compute range of achieved QV
            qv_range = max(achieved_widths) - min(achieved_widths)
            
            sensitivities[param_name] = {
                "correlation": float(correlation),
                "qv_range": int(qv_range),
                "max_qv": int(2 ** max(achieved_widths)) if max(achieved_widths) > 0 else 0,
                "min_qv": int(2 ** min(achieved_widths)) if min(achieved_widths) > 0 else 0,
            }
        
        # Save sensitivity analysis
        sensitivity_file = self.analysis_dir / "parameter_sensitivity.json"
        with open(sensitivity_file, 'w') as f:
            json.dump(sensitivities, f, indent=2)
        
        print(f"  [✓] Sensitivity analysis saved to {sensitivity_file}")
    
    def _generate_sweep_summaries(self) -> None:
        """Generate summary tables for each parameter sweep."""
        for param_name in self.sweep_params.keys():
            param_results = self._filter_results_by_param(param_name)
            
            if not param_results:
                continue
            
            summary_data = []
            
            for config_name in sorted(param_results.keys()):
                results = param_results[config_name]
                config = self.campaign_configs[config_name]
                
                # Get parameter value
                if hasattr(config, 'device'):
                    param_val = getattr(config.device, param_name)
                else:
                    param_val = config['device'][param_name]
                
                # Get achieved width
                passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
                max_width = max(passing_widths) if passing_widths else 0
                achieved_qv = 2 ** max_width if max_width > 0 else 0
                
                # Get HOP at max width
                if max_width > 0:
                    hop_at_max = results[max_width]["mean_hop"]
                else:
                    hop_at_max = 0.0
                
                summary_data.append({
                    "param_value": float(param_val),
                    "max_width": int(max_width),
                    "achieved_qv": int(achieved_qv),
                    "hop_at_max": float(hop_at_max),
                })
            
            # Save summary
            summary_file = self.analysis_dir / f"{param_name}_sweep_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
    
    def _generate_3d_surfaces(self) -> None:
        """Generate 3D parameter surfaces for parameter pairs."""
        param_names = list(self.sweep_params.keys())
        
        # Generate for key parameter pairs
        key_pairs = [
            ("F1", "F2"),
            ("T1", "T2"),
            ("t_single_gate", "t_two_gate"),
        ]
        
        for param1, param2 in key_pairs:
            if param1 not in param_names or param2 not in param_names:
                continue
            
            # This requires grid data - skip for now as we have 1D sweeps
            # In future, implement grid-based sweep support
            pass
    
    def _generate_html_report(self) -> None:
        """Generate HTML report with embedded plots and analysis."""
        html_content = self._create_html_report()
        
        report_file = self.output_dir / "campaign_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"  [✓] HTML report saved to {report_file}")
    
    def _filter_results_by_param(self, param_name: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Filter campaign results to configs that vary a specific parameter."""
        filtered = {}
        
        for config_name, results in self.campaign_results.items():
            if config_name == "baseline":
                filtered[config_name] = results
                continue
            
            config = self.campaign_configs[config_name]
            
            # Check if this config varies the target parameter
            if hasattr(config, 'metadata'):
                sweep_param = config.metadata.get("sweep_param")
            else:
                sweep_param = config.get('metadata', {}).get("sweep_param")
            
            if sweep_param == param_name:
                filtered[config_name] = results
        
        return filtered
    
    def _plot_hop_vs_parameter(
        self,
        param_name: str,
        param_results: Dict[str, Dict[int, Dict[str, Any]]],
        output_path: Path,
    ) -> None:
        """Plot HOP vs parameter value for different widths."""
        import matplotlib.pyplot as plt
        
        # Get all widths
        all_widths = set()
        for results in param_results.values():
            all_widths.update(results.keys())
        widths = sorted(all_widths)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each width as a separate line
        colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))
        
        for width, color in zip(widths, colors):
            param_vals = []
            hop_means = []
            hop_lowers = []
            hop_uppers = []
            
            for config_name in sorted(param_results.keys()):
                if width not in param_results[config_name]:
                    continue
                
                config = self.campaign_configs[config_name]
                
                # Get parameter value
                if hasattr(config, 'device'):
                    param_val = getattr(config.device, param_name)
                else:
                    param_val = config['device'][param_name]
                
                results = param_results[config_name][width]
                
                param_vals.append(param_val)
                hop_means.append(results["mean_hop"])
                hop_lowers.append(results["ci_lower"])
                hop_uppers.append(results["ci_upper"])
            
            if len(param_vals) < 2:
                continue
            
            # Sort by parameter value
            sorted_indices = np.argsort(param_vals)
            param_vals = np.array(param_vals)[sorted_indices]
            hop_means = np.array(hop_means)[sorted_indices]
            hop_lowers = np.array(hop_lowers)[sorted_indices]
            hop_uppers = np.array(hop_uppers)[sorted_indices]
            
            # Plot
            ax.plot(param_vals, hop_means, 'o-', label=f'Width {width}', 
                   color=color, linewidth=2, markersize=6)
            ax.fill_between(param_vals, hop_lowers, hop_uppers, alpha=0.2, color=color)
        
        # Threshold line
        ax.axhline(y=2/3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='QV Threshold')
        
        # Format
        ax.set_xlabel(param_name, fontweight='bold', fontsize=12)
        ax.set_ylabel('Heavy-Output Probability', fontweight='bold', fontsize=12)
        ax.set_title(f'HOP vs {param_name}', fontweight='bold', fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_qv_vs_parameter(
        self,
        param_name: str,
        param_results: Dict[str, Dict[int, Dict[str, Any]]],
        output_path: Path,
    ) -> None:
        """Plot achieved QV vs parameter value."""
        import matplotlib.pyplot as plt
        
        param_vals = []
        achieved_qvs = []
        
        for config_name in sorted(param_results.keys()):
            config = self.campaign_configs[config_name]
            results = param_results[config_name]
            
            # Get parameter value
            if hasattr(config, 'device'):
                param_val = getattr(config.device, param_name)
            else:
                param_val = config['device'][param_name]
            
            # Get achieved width
            passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
            max_width = max(passing_widths) if passing_widths else 0
            achieved_qv = 2 ** max_width if max_width > 0 else 0
            
            param_vals.append(param_val)
            achieved_qvs.append(achieved_qv)
        
        # Sort by parameter value
        sorted_indices = np.argsort(param_vals)
        param_vals = np.array(param_vals)[sorted_indices]
        achieved_qvs = np.array(achieved_qvs)[sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_vals, achieved_qvs, 'o-', linewidth=3, markersize=10, 
               color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
        
        # Format
        ax.set_xlabel(param_name, fontweight='bold', fontsize=12)
        ax.set_ylabel('Achieved Quantum Volume', fontweight='bold', fontsize=12)
        ax.set_title(f'Quantum Volume vs {param_name}', fontweight='bold', fontsize=14)
        
        if max(achieved_qvs) > 0:
            ax.set_yscale('log', base=2)
            qv_ticks = [2**i for i in range(1, int(np.log2(max(achieved_qvs))) + 2)]
            ax.set_yticks(qv_ticks)
            ax.set_yticklabels([str(qv) for qv in qv_ticks])
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QV Parameter Campaign Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2E86AB; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #e8f4f8; border-radius: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1>Quantum Volume Parameter Campaign Report</h1>
    
    <div class="summary">
        <h2>Campaign Overview</h2>
        <div class="metric">
            <div class="metric-value">{len(self.campaign_configs)}</div>
            <div class="metric-label">Total Configurations</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.sweep_params)}</div>
            <div class="metric-label">Parameters Swept</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self._get_max_achieved_qv()}</div>
            <div class="metric-label">Maximum QV Achieved</div>
        </div>
    </div>
    
    <h2>Overview Plots</h2>
    <div class="plot">
        <img src="plots/overview/campaign_comparison.png" alt="Campaign Comparison">
        <p><strong>Figure 1:</strong> HOP vs Width for all configurations</p>
    </div>
    
    <div class="plot">
        <img src="plots/overview/campaign_dashboard.png" alt="Campaign Dashboard">
        <p><strong>Figure 2:</strong> Comprehensive campaign dashboard</p>
    </div>
    
    <div class="plot">
        <img src="plots/parameter_correlations.png" alt="Parameter Correlations">
        <p><strong>Figure 3:</strong> Parameter correlation matrix</p>
    </div>
    
    <h2>Parameter-Specific Analysis</h2>
    {self._generate_param_sections_html()}
    
    <h2>Key Findings</h2>
    <div class="summary">
        {self._generate_findings_html()}
    </div>
    
    <p style="text-align: center; color: #999; margin-top: 50px;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</body>
</html>
"""
        return html
    
    def _get_max_achieved_qv(self) -> int:
        """Get maximum achieved QV across all configs."""
        max_qv = 0
        
        for results in self.campaign_results.values():
            if results is None:
                continue
            passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
            if passing_widths:
                max_width = max(passing_widths)
                qv = 2 ** max_width
                max_qv = max(max_qv, qv)
        
        return max_qv
    
    def _generate_param_sections_html(self) -> str:
        """Generate HTML sections for each parameter."""
        html_sections = []
        
        for param_name in self.sweep_params.keys():
            section = f"""
    <h3>{param_name}</h3>
    <div class="plot">
        <img src="plots/param_{param_name}/{param_name}_vs_hop.png" alt="{param_name} vs HOP">
        <p>HOP vs {param_name} for different circuit widths</p>
    </div>
    <div class="plot">
        <img src="plots/param_{param_name}/{param_name}_vs_qv.png" alt="{param_name} vs QV">
        <p>Achieved QV vs {param_name}</p>
    </div>
"""
            html_sections.append(section)
        
        return "\n".join(html_sections)
    
    def _generate_findings_html(self) -> str:
        """Generate key findings section."""
        # Load sensitivity analysis
        sensitivity_file = self.analysis_dir / "parameter_sensitivity.json"
        if sensitivity_file.exists():
            with open(sensitivity_file) as f:
                sensitivities = json.load(f)
            
            # Sort by correlation
            sorted_params = sorted(
                sensitivities.items(),
                key=lambda x: abs(x[1]["correlation"]),
                reverse=True
            )
            
            findings = "<ul>"
            for param, data in sorted_params[:3]:  # Top 3 most influential
                corr = data["correlation"]
                findings += f"<li><strong>{param}</strong>: Correlation with QV = {corr:.3f}</li>"
            findings += "</ul>"
            
            return findings
        else:
            return "<p>Sensitivity analysis not available.</p>"


from datetime import datetime
