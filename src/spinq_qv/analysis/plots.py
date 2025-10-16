"""
Plotting utilities for Quantum Volume analysis.

Generates publication-quality plots for HOP vs width, confidence intervals,
probability distributions, and error budgets.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


def plot_hop_vs_width(
    aggregated_results: Dict[int, Dict[str, Any]],
    output_path: Optional[Path] = None,
    show_ci: bool = True,
    show_threshold: bool = True,
    threshold: float = 2.0 / 3.0,
) -> plt.Figure:
    """
    Plot Heavy-Output Probability vs circuit width.
    
    Args:
        aggregated_results: Dictionary mapping width -> statistics
        output_path: Optional path to save figure
        show_ci: Whether to show confidence intervals
        show_threshold: Whether to show QV threshold line
        threshold: QV success threshold (default: 2/3)
    
    Returns:
        Matplotlib figure object
    """
    # Extract data
    widths = sorted(aggregated_results.keys())
    means = [aggregated_results[w]["mean_hop"] for w in widths]
    
    if show_ci:
        ci_lower = [aggregated_results[w]["ci_lower"] for w in widths]
        ci_upper = [aggregated_results[w]["ci_upper"] for w in widths]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot mean HOP
    ax.plot(widths, means, 'o-', linewidth=2, markersize=8, 
            label='Mean HOP', color='#2E86AB', zorder=3)
    
    # Plot confidence intervals
    if show_ci:
        ax.fill_between(widths, ci_lower, ci_upper, alpha=0.3, 
                        color='#2E86AB', label='95% CI')
        
        # Also plot CI bounds as lines
        ax.plot(widths, ci_lower, '--', color='#2E86AB', alpha=0.7, linewidth=1)
        ax.plot(widths, ci_upper, '--', color='#2E86AB', alpha=0.7, linewidth=1)
    
    # Plot threshold line
    if show_threshold:
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'QV Threshold ({threshold:.3f})', zorder=2)
    
    # Formatting
    ax.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax.set_ylabel('Heavy-Output Probability', fontweight='bold')
    ax.set_title('Quantum Volume: HOP vs Circuit Width', fontweight='bold', pad=15)
    
    ax.set_xticks(widths)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='best', framealpha=0.9)
    
    # Add pass/fail markers
    for w in widths:
        if aggregated_results[w].get("pass_qv", False):
            # Green check mark
            ax.plot(w, aggregated_results[w]["mean_hop"], 'g*', 
                   markersize=15, zorder=4)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved HOP plot to {output_path}")
    
    return fig


def plot_hop_distribution(
    hops: np.ndarray,
    width: int,
    output_path: Optional[Path] = None,
    threshold: float = 2.0 / 3.0,
) -> plt.Figure:
    """
    Plot distribution of HOP values for a given width.
    
    Args:
        hops: Array of HOP values
        width: Circuit width
        output_path: Optional path to save figure
        threshold: QV threshold to mark
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    n_bins = min(30, len(hops) // 3) if len(hops) > 10 else 10
    ax.hist(hops, bins=n_bins, density=True, alpha=0.7, 
           color='#A23B72', edgecolor='black', linewidth=0.8)
    
    # Mean line
    mean_hop = np.mean(hops)
    ax.axvline(x=mean_hop, color='blue', linestyle='-', 
              linewidth=2, label=f'Mean = {mean_hop:.3f}')
    
    # Threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', 
              linewidth=2, label=f'Threshold = {threshold:.3f}')
    
    # Formatting
    ax.set_xlabel('Heavy-Output Probability', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'HOP Distribution (Width m={width})', fontweight='bold', pad=15)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved distribution plot to {output_path}")
    
    return fig


def plot_ideal_vs_measured_probs(
    ideal_probs: np.ndarray,
    measured_counts: Dict[str, int],
    width: int,
    output_path: Optional[Path] = None,
    max_states: int = 64,
) -> plt.Figure:
    """
    Plot ideal vs measured probability distributions.
    
    Args:
        ideal_probs: Ideal probability array (length 2^m)
        measured_counts: Measured bitstring counts
        width: Circuit width
        output_path: Optional save path
        max_states: Maximum number of states to show (for readability)
    
    Returns:
        Matplotlib figure object
    """
    n_states = len(ideal_probs)
    
    # Convert measured counts to probabilities
    total_shots = sum(measured_counts.values())
    measured_probs = np.zeros(n_states)
    
    for bitstring, count in measured_counts.items():
        index = int(bitstring, 2)
        measured_probs[index] = count / total_shots
    
    # Limit to most probable states if too many
    if n_states > max_states:
        # Show top states by ideal probability
        top_indices = np.argsort(ideal_probs)[-max_states:]
        ideal_probs = ideal_probs[top_indices]
        measured_probs = measured_probs[top_indices]
        state_labels = [f"{i}" for i in top_indices]
    else:
        state_labels = [f"{i}" for i in range(n_states)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(state_labels))
    width_bar = 0.35
    
    ax.bar(x - width_bar/2, ideal_probs, width_bar, 
          label='Ideal', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width_bar/2, measured_probs, width_bar, 
          label='Measured', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Computational Basis State', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title(f'Ideal vs Measured Probabilities (m={width})', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, rotation=45 if len(state_labels) > 20 else 0)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved probability comparison to {output_path}")
    
    return fig


def plot_qv_summary(
    aggregated_results: Dict[int, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a comprehensive QV summary plot with multiple panels.
    
    Args:
        aggregated_results: Aggregated results by width
        output_path: Optional save path
    
    Returns:
        Matplotlib figure with multiple subplots
    """
    widths = sorted(aggregated_results.keys())
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: HOP vs Width
    ax1 = fig.add_subplot(gs[0, :])
    
    means = [aggregated_results[w]["mean_hop"] for w in widths]
    ci_lower = [aggregated_results[w]["ci_lower"] for w in widths]
    ci_upper = [aggregated_results[w]["ci_upper"] for w in widths]
    
    ax1.plot(widths, means, 'o-', linewidth=2, markersize=8, 
            label='Mean HOP', color='#2E86AB')
    ax1.fill_between(widths, ci_lower, ci_upper, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=2/3, color='red', linestyle='--', linewidth=2, label='QV Threshold')
    
    # Mark passing widths
    for w in widths:
        if aggregated_results[w].get("pass_qv", False):
            ax1.plot(w, means[widths.index(w)], 'g*', markersize=15)
    
    ax1.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax1.set_ylabel('Heavy-Output Probability', fontweight='bold')
    ax1.set_title('Quantum Volume Results', fontweight='bold', pad=15)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Number of circuits per width
    ax2 = fig.add_subplot(gs[1, 0])
    n_circuits = [aggregated_results[w]["n_circuits"] for w in widths]
    ax2.bar(widths, n_circuits, color='#95B8D1', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Width (m)', fontweight='bold')
    ax2.set_ylabel('Number of Circuits', fontweight='bold')
    ax2.set_title('Circuits per Width', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Pass/Fail summary
    ax3 = fig.add_subplot(gs[1, 1])
    pass_count = sum(1 for w in widths if aggregated_results[w].get("pass_qv", False))
    fail_count = len(widths) - pass_count
    
    colors = ['#81C784', '#E57373']
    labels = [f'Pass ({pass_count})', f'Fail ({fail_count})']
    sizes = [pass_count, fail_count]
    
    if pass_count > 0 or fail_count > 0:
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontweight': 'bold'})
        ax3.set_title('QV Pass/Fail Summary', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved summary plot to {output_path}")
    
    return fig


def save_all_plots(
    aggregated_results: Dict[int, Dict[str, Any]],
    output_dir: Path,
    formats: List[str] = ['png', 'svg'],
) -> None:
    """
    Generate and save all standard QV plots.
    
    Args:
        aggregated_results: Aggregated results by width
        output_dir: Directory to save plots
        formats: List of file formats to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    
    for fmt in formats:
        # HOP vs width
        plot_hop_vs_width(
            aggregated_results,
            output_path=output_dir / f"hop_vs_width.{fmt}",
        )
        plt.close()
        
        # Summary
        plot_qv_summary(
            aggregated_results,
            output_path=output_dir / f"qv_summary.{fmt}",
        )
        plt.close()
        
        # Individual width distributions
        for width, results in aggregated_results.items():
            if "hops" in results:
                plot_hop_distribution(
                    np.array(results["hops"]),
                    width=width,
                    output_path=output_dir / f"hop_dist_m{width}.{fmt}",
                )
                plt.close()
    
    print(f"[OK] All plots saved to {output_dir}/")


def plot_sensitivity_heatmap(
    grid: np.ndarray,
    param1_values: List[float],
    param2_values: List[float],
    param1_name: str,
    param2_name: str,
    output_path: Optional[Path] = None,
    title: str = "Sensitivity Heatmap",
    cmap: str = 'RdYlGn',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Plot 2D parameter sensitivity heatmap.
    
    Args:
        grid: 2D array with shape (len(param2_values), len(param1_values))
        param1_values: Values for x-axis parameter
        param2_values: Values for y-axis parameter
        param1_name: Name of x-axis parameter
        param2_name: Name of y-axis parameter
        output_path: Optional path to save figure
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(
        grid,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            param1_values[0], param1_values[-1],
            param2_values[0], param2_values[-1]
        ]
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean HOP', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(title)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"[OK] Saved heatmap to {output_path}")
    
    return fig


def plot_error_budget_pie(
    budget: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    title: str = "Error Budget",
    baseline_key: str = "baseline",
    ideal_key: str = "ideal",
    min_percentage: float = 1.0,
) -> plt.Figure:
    """
    Plot error budget as pie chart.
    
    Args:
        budget: Error budget dictionary from compute_error_budget()
        output_path: Optional path to save figure
        title: Plot title
        baseline_key: Key for baseline (excluded from chart)
        ideal_key: Key for ideal (excluded from chart)
        min_percentage: Minimum percentage to show separately (others grouped as "Other")
    
    Returns:
        Matplotlib figure object
    """
    # Filter out baseline and ideal
    contributions = {
        k: v for k, v in budget.items()
        if k not in [baseline_key, ideal_key] and v['pct_of_gap'] > 0
    }
    
    # Sort by contribution
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: x[1]['pct_of_gap'],
        reverse=True
    )
    
    # Group small contributions
    labels = []
    sizes = []
    other_pct = 0.0
    
    for label, data in sorted_items:
        pct = data['pct_of_gap']
        if pct >= min_percentage:
            labels.append(f"{label}\n({pct:.1f}%)")
            sizes.append(pct)
        else:
            other_pct += pct
    
    if other_pct > 0:
        labels.append(f"Other\n({other_pct:.1f}%)")
        sizes.append(other_pct)
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9}
    )
    
    # Equal aspect ratio ensures pie is circular
    ax.axis('equal')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"[OK] Saved error budget pie chart to {output_path}")
    
    return fig


def plot_error_budget_bar(
    budget: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    title: str = "Error Budget Breakdown",
    baseline_key: str = "baseline",
    ideal_key: str = "ideal",
) -> plt.Figure:
    """
    Plot error budget as horizontal bar chart.
    
    Args:
        budget: Error budget dictionary from compute_error_budget()
        output_path: Optional path to save figure
        title: Plot title
        baseline_key: Key for baseline (excluded from chart)
        ideal_key: Key for ideal (excluded from chart)
    
    Returns:
        Matplotlib figure object
    """
    # Filter and sort
    contributions = {
        k: v for k, v in budget.items()
        if k not in [baseline_key, ideal_key]
    }
    
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: x[1]['pct_of_gap'],
        reverse=False  # Ascending for horizontal bar (bottom to top)
    )
    
    labels = [item[0] for item in sorted_items]
    percentages = [item[1]['pct_of_gap'] for item in sorted_items]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.barh(labels, percentages, color=colors)
    
    ax.set_xlabel('Contribution to Error Gap (%)')
    ax.set_ylabel('Error Source')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax.text(
            pct + 1,
            i,
            f'{pct:.1f}%',
            va='center',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"[OK] Saved error budget bar chart to {output_path}")
    
    return fig


def plot_parameter_sweep(
    param_values: List[float],
    mean_hops: List[float],
    ci_lower: Optional[List[float]] = None,
    ci_upper: Optional[List[float]] = None,
    param_name: str = "Parameter",
    output_path: Optional[Path] = None,
    title: str = "Parameter Sensitivity",
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot 1D parameter sweep with confidence intervals.
    
    Args:
        param_values: Parameter values (x-axis)
        mean_hops: Mean HOP values (y-axis)
        ci_lower: Lower confidence interval bounds (optional)
        ci_upper: Upper confidence interval bounds (optional)
        param_name: Name of parameter
        output_path: Optional path to save figure
        title: Plot title
        log_scale: Use log scale for x-axis
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Main line
    ax.plot(param_values, mean_hops, 'o-', linewidth=2, markersize=8, label='Mean HOP')
    
    # Confidence intervals
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            param_values,
            ci_lower,
            ci_upper,
            alpha=0.3,
            label='95% CI'
        )
    
    # Threshold line
    ax.axhline(y=2.0/3.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='QV Threshold')
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Heavy-Output Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"[OK] Saved parameter sweep plot to {output_path}")
    
    return fig


def save_all_sensitivity_plots(
    aggregated_results: Dict[int, Dict[str, Any]],
    budget: Optional[Dict[str, Dict[str, float]]] = None,
    grid_data: Optional[Tuple[np.ndarray, List[float], List[float], str, str]] = None,
    output_dir: Path = Path("sensitivity_plots"),
    formats: List[str] = ["png", "svg"],
) -> None:
    """
    Generate and save all sensitivity analysis plots.
    
    Args:
        aggregated_results: QV results dictionary
        budget: Error budget dictionary (optional)
        grid_data: Tuple of (grid, param1_vals, param2_vals, param1_name, param2_name) for heatmap
        output_dir: Directory to save plots
        formats: List of file formats to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Error budget plots
    if budget is not None:
        for fmt in formats:
            # Pie chart
            plot_error_budget_pie(
                budget,
                output_path=output_dir / f"error_budget_pie.{fmt}",
            )
            plt.close()
            
            # Bar chart
            plot_error_budget_bar(
                budget,
                output_path=output_dir / f"error_budget_bar.{fmt}",
            )
            plt.close()
    
    # Heatmap
    if grid_data is not None:
        grid, param1_vals, param2_vals, param1_name, param2_name = grid_data
        for fmt in formats:
            plot_sensitivity_heatmap(
                grid,
                param1_vals,
                param2_vals,
                param1_name,
                param2_name,
                output_path=output_dir / f"heatmap_{param1_name}_vs_{param2_name}.{fmt}",
                title=f"Sensitivity: {param1_name} vs {param2_name}",
            )
            plt.close()
    
    print(f"[OK] All sensitivity plots saved to {output_dir}/")
