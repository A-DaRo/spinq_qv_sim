"""
Professional report generation for Quantum Volume experiments.

Produces publication-quality PDF reports with:
- HOP vs m plots with confidence intervals
- Sensitivity analysis heatmaps
- RB validation plots
- Error budget breakdowns
- Hardware improvement recommendations
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import json

from spinq_qv.io.storage import QVResultsReader, load_run
from spinq_qv.analysis.plots import (
    plot_hop_vs_width,
    plot_error_budget_pie,
    plot_error_budget_bar,
    plot_sensitivity_heatmap,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive QV campaign reports.
    
    Produces multi-page PDF with all analysis artifacts, metrics,
    and actionable recommendations for hardware improvements.
    """
    
    def __init__(
        self,
        campaign_results_path: Path,
        output_dir: Path,
        title: Optional[str] = None,
    ):
        """
        Initialize report generator.
        
        Args:
            campaign_results_path: Path to campaign HDF5 results file
            output_dir: Output directory for report and figures
            title: Custom report title (auto-generated if None)
        """
        self.results_path = Path(campaign_results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if title is None:
            title = f"Quantum Volume Campaign Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        self.title = title
        
        # Load data
        logger.info(f"Loading campaign results from {self.results_path}")
        self.data = load_run(self.results_path)
        
        logger.info(f"Loaded data for widths: {self.data['widths']}")
    
    def generate_report(
        self,
        include_sensitivity: bool = False,
        sensitivity_dir: Optional[Path] = None,
        include_rb_validation: bool = False,
        rb_validation_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate comprehensive PDF report.
        
        Args:
            include_sensitivity: Include sensitivity analysis section
            sensitivity_dir: Directory containing sensitivity results
            include_rb_validation: Include RB validation section
            rb_validation_path: Path to RB validation results
        
        Returns:
            Path to generated PDF
        """
        pdf_path = self.output_dir / "QV_summary.pdf"
        
        logger.info(f"Generating report: {pdf_path}")
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title and summary
            self._create_title_page(pdf)
            
            # Page 2: HOP vs m with CI
            self._create_hop_page(pdf)
            
            # Page 3: QV decision and statistics
            self._create_statistics_page(pdf)
            
            # Page 4: Error budget (if available)
            if include_sensitivity and sensitivity_dir:
                self._create_error_budget_page(pdf, sensitivity_dir)
            
            # Page 5: Sensitivity heatmaps (if available)
            if include_sensitivity and sensitivity_dir:
                self._create_sensitivity_page(pdf, sensitivity_dir)
            
            # Page 6: RB validation (if available)
            if include_rb_validation and rb_validation_path:
                self._create_rb_validation_page(pdf, rb_validation_path)
            
            # Page 7: Hardware recommendations
            self._create_recommendations_page(pdf)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = self.title
            d['Author'] = 'spinq_qv_sim'
            d['Subject'] = 'Quantum Volume Simulation Results'
            d['CreationDate'] = datetime.now()
        
        logger.info(f"[OK] Report saved to {pdf_path}")
        return pdf_path
    
    def _create_title_page(self, pdf: PdfPages) -> None:
        """Create title page with executive summary."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(
            0.5, 0.9, self.title,
            ha='center', va='top', fontsize=20, fontweight='bold',
            transform=ax.transAxes,
        )
        
        # Metadata
        metadata = self.data['metadata']
        config = metadata.get('config', {})
        
        y_pos = 0.75
        line_height = 0.05
        
        # Campaign info
        info_lines = [
            f"Campaign ID: {metadata.get('campaign_id', 'N/A')}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Git Hash: {metadata.get('git_hash', 'unknown')}",
            f"",
            "DEVICE PARAMETERS:",
            f"  Single-qubit fidelity (F1): {config.get('device', {}).get('F1', 'N/A'):.5f}",
            f"  Two-qubit fidelity (F2): {config.get('device', {}).get('F2', 'N/A'):.5f}",
            f"  T1: {config.get('device', {}).get('T1', 'N/A')} s",
            f"  T2: {config.get('device', {}).get('T2', 'N/A')*1e6:.1f} μs",
            f"  T2*: {config.get('device', {}).get('T2_star', 'N/A')*1e6:.1f} μs",
            f"",
            "SIMULATION PARAMETERS:",
            f"  Backend: {config.get('simulation', {}).get('backend', 'N/A')}",
            f"  Circuits per width: {config.get('simulation', {}).get('n_circuits', 'N/A')}",
            f"  Shots per circuit: {config.get('simulation', {}).get('n_shots', 'N/A')}",
            f"  Widths tested: {', '.join(map(str, self.data['widths']))}",
        ]
        
        for line in info_lines:
            ax.text(
                0.1, y_pos, line,
                ha='left', va='top', fontsize=10, family='monospace',
                transform=ax.transAxes,
            )
            y_pos -= line_height
        
        # Executive summary box
        summary_y = 0.25
        ax.text(
            0.5, summary_y + 0.05, "EXECUTIVE SUMMARY",
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes,
        )
        
        # Compute highest passing width
        widths = self.data['widths']
        passing_widths = []
        
        for width in widths:
            agg = self.data['aggregated'].get(width, {})
            mean_hop = agg.get('mean_hop', 0)
            ci_lower = agg.get('ci_lower', 0)
            
            # IBM QV criterion: mean HOP > 2/3 AND lower CI > 2/3
            if mean_hop > 2/3 and ci_lower > 2/3:
                passing_widths.append(width)
        
        if passing_widths:
            max_passing = max(passing_widths)
            estimated_qv = 2 ** max_passing
            summary_text = f"Estimated QV = 2^{max_passing} = {estimated_qv}"
            color = 'green'
        else:
            summary_text = "QV criterion not achieved at any width"
            color = 'red'
        
        ax.text(
            0.5, summary_y - 0.05, summary_text,
            ha='center', va='top', fontsize=16, fontweight='bold',
            color=color, transform=ax.transAxes,
        )
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_hop_page(self, pdf: PdfPages) -> None:
        """Create HOP vs m plot with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        widths = np.array(self.data['widths'])
        mean_hops = []
        ci_lowers = []
        ci_uppers = []
        
        for width in widths:
            agg = self.data['aggregated'].get(width, {})
            mean_hops.append(agg.get('mean_hop', 0))
            ci_lowers.append(agg.get('ci_lower', 0))
            ci_uppers.append(agg.get('ci_upper', 0))
        
        mean_hops = np.array(mean_hops)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        
        # Plot HOP vs m
        ax.plot(widths, mean_hops, 'o-', linewidth=2, markersize=8, label='Mean HOP')
        ax.fill_between(
            widths, ci_lowers, ci_uppers,
            alpha=0.3, label='95% Bootstrap CI'
        )
        
        # QV threshold line
        ax.axhline(2/3, color='red', linestyle='--', linewidth=1.5, label='QV Threshold (2/3)')
        
        # Formatting
        ax.set_xlabel('Circuit Width (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Heavy-Output Probability (HOP)', fontsize=12, fontweight='bold')
        ax.set_title('Quantum Volume Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        
        # Add text annotations for passing widths
        for i, width in enumerate(widths):
            if mean_hops[i] > 2/3 and ci_lowers[i] > 2/3:
                ax.annotate(
                    f'PASS\nQV={2**width}',
                    xy=(width, mean_hops[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                )
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _create_statistics_page(self, pdf: PdfPages) -> None:
        """Create page with detailed statistics table."""
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(
            0.5, 0.95, "Statistical Analysis Summary",
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes,
        )
        
        # Create table data
        table_data = [['Width (m)', 'QV', 'Mean HOP', 'Std HOP', 'CI Lower', 'CI Upper', 'Pass?']]
        
        for width in self.data['widths']:
            agg = self.data['aggregated'].get(width, {})
            mean_hop = agg.get('mean_hop', 0)
            std_hop = agg.get('std_hop', 0)
            ci_lower = agg.get('ci_lower', 0)
            ci_upper = agg.get('ci_upper', 0)
            
            qv = 2 ** width
            passes = (mean_hop > 2/3 and ci_lower > 2/3)
            pass_str = '✓ PASS' if passes else '✗ FAIL'
            
            table_data.append([
                str(width),
                str(qv),
                f"{mean_hop:.4f}",
                f"{std_hop:.4f}",
                f"{ci_lower:.4f}",
                f"{ci_upper:.4f}",
                pass_str,
            ])
        
        # Create table
        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.3, 0.8, 0.6],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(7):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Color-code pass/fail
        for i in range(1, len(table_data)):
            cell = table[(i, 6)]
            if 'PASS' in cell.get_text().get_text():
                cell.set_facecolor('#C8E6C9')
            else:
                cell.set_facecolor('#FFCDD2')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_error_budget_page(self, pdf: PdfPages, sensitivity_dir: Path) -> None:
        """Create error budget visualization page."""
        # Load error budget if available
        budget_file = sensitivity_dir / "ablation" / "error_budget.json"
        
        if not budget_file.exists():
            logger.warning(f"Error budget file not found: {budget_file}")
            return
        
        with open(budget_file, 'r') as f:
            budget = json.load(f)
        
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Error Budget Analysis', fontsize=16, fontweight='bold')
        
        # Pie chart
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_error_budget_pie_custom(ax1, budget)
        
        # Bar chart
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_error_budget_bar_custom(ax2, budget)
        
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _plot_error_budget_pie_custom(self, ax, budget: Dict[str, Any]) -> None:
        """Plot error budget as pie chart."""
        labels = []
        sizes = []
        
        for label, data in budget.items():
            if label not in ['baseline', 'ideal']:
                pct = abs(data.get('pct_of_gap', 0))
                if pct > 1:  # Only show contributions > 1%
                    labels.append(label.replace('_', ' ').title())
                    sizes.append(pct)
        
        if sizes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Error Source Contributions (%)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No error budget data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Source Contributions', fontsize=12, fontweight='bold')
    
    def _plot_error_budget_bar_custom(self, ax, budget: Dict[str, Any]) -> None:
        """Plot error budget as horizontal bar chart."""
        labels = []
        values = []
        
        for label, data in budget.items():
            if label not in ['baseline', 'ideal']:
                pct = abs(data.get('pct_of_gap', 0))
                labels.append(label.replace('_', ' ').title())
                values.append(pct)
        
        if values:
            # Sort by contribution
            sorted_idx = np.argsort(values)[::-1]
            labels = [labels[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(values)))
            ax.barh(labels, values, color=colors)
            ax.set_xlabel('Contribution to Gap (%)', fontsize=11, fontweight='bold')
            ax.set_title('Error Budget Breakdown', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No error budget data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Budget Breakdown', fontsize=12, fontweight='bold')
    
    def _create_sensitivity_page(self, pdf: PdfPages, sensitivity_dir: Path) -> None:
        """Create sensitivity analysis heatmap page."""
        # Look for 2D grid results
        grid_files = list(sensitivity_dir.glob("sensitivity_2d/*.csv"))
        
        if not grid_files:
            logger.warning("No 2D sensitivity grid files found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # This is a placeholder - in production, load actual heatmap data
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Heatmap placeholder\n(Load from CSV)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter 1 vs Parameter 2', fontsize=11)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _create_rb_validation_page(self, pdf: PdfPages, rb_path: Path) -> None:
        """Create RB validation plots page."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Randomized Benchmarking Validation', fontsize=16, fontweight='bold')
        
        # Placeholder for RB plots
        for i, (ax, gate_type) in enumerate(zip(axes, ['Single-Qubit', 'Two-Qubit'])):
            ax.text(0.5, 0.5, f'{gate_type} RB\nPlaceholder',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Sequence Length', fontsize=11)
            ax.set_ylabel('Survival Probability', fontsize=11)
            ax.set_title(f'{gate_type} Fidelity Validation', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf: PdfPages) -> None:
        """Create hardware improvement recommendations page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(
            0.5, 0.95, "Hardware Improvement Recommendations",
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes,
        )
        
        # Analyze current performance
        metadata = self.data['metadata']
        config = metadata.get('config', {})
        device = config.get('device', {})
        
        F1 = device.get('F1', 0)
        F2 = device.get('F2', 0)
        T2 = device.get('T2', 0) * 1e6  # Convert to μs
        T2_star = device.get('T2_star', 0) * 1e6
        
        # Determine limiting factors
        recommendations = []
        
        if F2 < 0.999:
            recommendations.append(
                f"• Two-qubit gate fidelity (F2 = {F2:.5f}) is the primary limitation.\n"
                f"  Target: F2 > 0.9990 to achieve higher QV.\n"
                f"  Actions: Optimize gate calibration, reduce crosstalk, improve gate pulse shapes."
            )
        
        if F1 < 0.9995:
            recommendations.append(
                f"• Single-qubit gate fidelity (F1 = {F1:.5f}) can be improved.\n"
                f"  Target: F1 > 0.9995 for QV > 64.\n"
                f"  Actions: Refine single-qubit control, minimize leakage."
            )
        
        if T2 < 100:
            recommendations.append(
                f"• Coherence time T2 = {T2:.1f} μs limits circuit depth.\n"
                f"  Target: T2 > 200 μs for QV > 128.\n"
                f"  Actions: Improve material purity, reduce charge noise, optimize device geometry."
            )
        
        if T2_star < 50:
            recommendations.append(
                f"• Dephasing time T2* = {T2_star:.1f} μs indicates quasi-static noise.\n"
                f"  Target: T2* > T2/2 (currently T2*/T2 = {T2_star/T2:.2f}).\n"
                f"  Actions: Reduce low-frequency noise sources, improve magnetic field stability."
            )
        
        if not recommendations:
            recommendations.append(
                "• Device performance is excellent across all metrics.\n"
                "  Continue current optimization strategies to push QV higher."
            )
        
        # Add general recommendations
        recommendations.append("\nGENERAL RECOMMENDATIONS:")
        recommendations.append(
            "• Prioritize two-qubit gate optimization (typically dominant error source)"
        )
        recommendations.append(
            "• Implement dynamical decoupling to extend effective T2"
        )
        recommendations.append(
            "• Use error mitigation techniques (readout error correction, ZNE)"
        )
        recommendations.append(
            "• Consider gate compilation optimization to reduce circuit depth"
        )
        
        # Display recommendations
        y_pos = 0.85
        line_height = 0.04
        
        for rec in recommendations:
            ax.text(
                0.1, y_pos, rec,
                ha='left', va='top', fontsize=10, wrap=True,
                transform=ax.transAxes,
            )
            y_pos -= line_height * (rec.count('\n') + 1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def generate_campaign_report(
    campaign_results_path: Path,
    output_dir: Path,
    title: Optional[str] = None,
    include_sensitivity: bool = False,
    sensitivity_dir: Optional[Path] = None,
    include_rb: bool = False,
    rb_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function to generate a complete campaign report.
    
    Args:
        campaign_results_path: Path to campaign HDF5 file
        output_dir: Output directory for report
        title: Custom title
        include_sensitivity: Include sensitivity analysis
        sensitivity_dir: Directory with sensitivity results
        include_rb: Include RB validation
        rb_path: Path to RB results
    
    Returns:
        Path to generated PDF
    """
    generator = ReportGenerator(campaign_results_path, output_dir, title)
    
    return generator.generate_report(
        include_sensitivity=include_sensitivity,
        sensitivity_dir=sensitivity_dir,
        include_rb_validation=include_rb,
        rb_validation_path=rb_path,
    )
