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


# ============================================================================
# CAMPAIGN-LEVEL PLOTS: Multi-Configuration Analysis & Comparison
# ============================================================================


def plot_campaign_comparison(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Optional[Path] = None,
    show_ci: bool = True,
    threshold: float = 2.0 / 3.0,
    colors: Optional[List[str]] = None,
    title: str = "Campaign Comparison: HOP vs Width",
) -> plt.Figure:
    """
    Compare HOP curves across multiple configurations in a single plot.
    
    This is the primary campaign visualization - shows all configurations
    overlaid to directly compare performance.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        output_path: Optional save path
        show_ci: Whether to show confidence intervals
        threshold: QV threshold line
        colors: Optional list of colors for each config
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Default color palette if not provided
    if colors is None:
        # Use distinct colors from colorblind-friendly palette
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
                  '#0072B2', '#D55E00', '#CC79A7', '#999999']
    
    config_names = sorted(campaign_results.keys())
    
    for idx, config_name in enumerate(config_names):
        results = campaign_results[config_name]
        widths = sorted(results.keys())
        means = [results[w]["mean_hop"] for w in widths]
        
        color = colors[idx % len(colors)]
        
        # Plot mean line
        ax.plot(widths, means, 'o-', linewidth=2.5, markersize=9, 
                label=config_name, color=color, alpha=0.9, zorder=3)
        
        # Plot confidence intervals if requested
        if show_ci:
            ci_lower = [results[w]["ci_lower"] for w in widths]
            ci_upper = [results[w]["ci_upper"] for w in widths]
            ax.fill_between(widths, ci_lower, ci_upper, 
                          alpha=0.15, color=color, zorder=1)
        
        # Mark passing widths with stars
        for w in widths:
            if results[w].get("pass_qv", False):
                ax.plot(w, means[widths.index(w)], '*', 
                       markersize=18, color=color, 
                       markeredgecolor='darkgreen', markeredgewidth=1.5,
                       zorder=4)
    
    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', 
              linewidth=2.5, label=f'QV Threshold ({threshold:.3f})', 
              zorder=2, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Circuit Width (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Heavy-Output Probability', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.0)
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved campaign comparison plot to {output_path}")
    
    return fig


def plot_achieved_qv_comparison(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Optional[Path] = None,
    colors: Optional[List[str]] = None,
    show_values: bool = True,
) -> plt.Figure:
    """
    Bar chart showing maximum achieved QV for each configuration.
    
    Key insight: Shows the final QV metric (2^m) for each device config,
    making it easy to compare overall performance.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        output_path: Optional save path
        colors: Optional color list
        show_values: Whether to annotate bars with QV values
    
    Returns:
        Matplotlib figure object
    """
    # Calculate achieved QV for each config
    config_names = []
    achieved_qvs = []
    max_widths = []
    
    for config_name, results in campaign_results.items():
        # Find maximum passing width
        passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
        
        if passing_widths:
            max_width = max(passing_widths)
            achieved_qv = 2 ** max_width
        else:
            max_width = 0
            achieved_qv = 0
        
        config_names.append(config_name)
        achieved_qvs.append(achieved_qv)
        max_widths.append(max_width)
    
    # Sort by achieved QV
    sorted_indices = np.argsort(achieved_qvs)[::-1]
    config_names = [config_names[i] for i in sorted_indices]
    achieved_qvs = [achieved_qvs[i] for i in sorted_indices]
    max_widths = [max_widths[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Default colors
    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
                  '#6A994E', '#BC4B51', '#8D5B4C', '#5E548E']
    
    bar_colors = [colors[i % len(colors)] for i in range(len(config_names))]
    
    # Create bars
    y_pos = np.arange(len(config_names))
    bars = ax.barh(y_pos, achieved_qvs, color=bar_colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Annotate with QV values
    if show_values:
        for i, (bar, qv, width) in enumerate(zip(bars, achieved_qvs, max_widths)):
            if qv > 0:
                label = f"QV = {qv} (m={width})"
                ax.text(qv + max(achieved_qvs) * 0.02, i, label, 
                       va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(max(achieved_qvs) * 0.02, i, "No QV achieved", 
                       va='center', fontsize=10, style='italic', color='red')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(config_names, fontsize=10)
    ax.set_xlabel('Quantum Volume (QV)', fontweight='bold', fontsize=12)
    ax.set_title('Achieved Quantum Volume by Configuration', 
                fontweight='bold', fontsize=14, pad=20)
    
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.8)
    
    # Add vertical lines at powers of 2
    if achieved_qvs:
        max_qv = max(achieved_qvs) if max(achieved_qvs) > 0 else 4
        qv_ticks = [2**i for i in range(1, int(np.log2(max_qv)) + 2)]
        ax.set_xticks(qv_ticks)
        ax.set_xticklabels([f"{qv}" for qv in qv_ticks])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved achieved QV comparison to {output_path}")
    
    return fig


def plot_parameter_correlation_matrix(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    campaign_configs: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
    params_to_analyze: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Correlation heatmap between device parameters and achieved QV.
    
    Key insight: Identifies which parameters most strongly correlate with QV,
    helping focus optimization efforts.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        campaign_configs: Dict mapping config_name -> device parameters
        output_path: Optional save path
        params_to_analyze: List of parameter names (default: all device params)
    
    Returns:
        Matplotlib figure object
    """
    # Extract achieved QV for each config
    config_names = list(campaign_results.keys())
    achieved_qvs = []
    
    for config_name in config_names:
        results = campaign_results[config_name]
        passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
        max_width = max(passing_widths) if passing_widths else 0
        achieved_qvs.append(max_width)  # Use width, not 2^width for linearity
    
    # Default parameters to analyze
    if params_to_analyze is None:
        params_to_analyze = ['F1', 'F2', 'T1', 'T2', 'T2_star', 
                            't_single_gate', 't_two_gate', 
                            'F_readout', 'F_init']
    
    # Extract parameter values
    param_matrix = []
    actual_params = []
    
    for param_name in params_to_analyze:
        values = []
        for config_name in config_names:
            if config_name in campaign_configs:
                device_config = campaign_configs[config_name].get('device', {})
                if param_name in device_config:
                    values.append(float(device_config[param_name]))
        
        if len(values) == len(config_names):
            param_matrix.append(values)
            actual_params.append(param_name)
    
    if not param_matrix:
        print("[WARNING] No valid parameters found for correlation analysis")
        return None
    
    # Add achieved QV as last column
    param_matrix.append(achieved_qvs)
    actual_params.append('Max Width (m)')
    
    # Compute correlation matrix
    param_array = np.array(param_matrix)
    
    # Normalize each parameter to [0, 1] for fair correlation
    param_normalized = np.zeros_like(param_array)
    for i in range(param_array.shape[0]):
        row = param_array[i]
        if np.std(row) > 1e-10:  # Check for variance
            param_normalized[i] = (row - np.min(row)) / (np.max(row) - np.min(row) + 1e-10)
        else:
            param_normalized[i] = row
    
    # Compute correlation
    corr_matrix = np.corrcoef(param_normalized)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontweight='bold')
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(actual_params)))
    ax.set_yticks(np.arange(len(actual_params)))
    ax.set_xticklabels(actual_params, rotation=45, ha='right')
    ax.set_yticklabels(actual_params)
    
    # Annotate cells with correlation values
    for i in range(len(actual_params)):
        for j in range(len(actual_params)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', 
                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                          fontsize=9, fontweight='bold')
    
    ax.set_title('Parameter Correlation Matrix\n(vs Maximum Achieved Width)', 
                fontweight='bold', fontsize=13, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved correlation matrix to {output_path}")
    
    return fig


def plot_qv_waterfall(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Optional[Path] = None,
    threshold: float = 2.0 / 3.0,
) -> plt.Figure:
    """
    Waterfall plot showing HOP degradation across widths for each config.
    
    Key insight: Visualizes where each configuration starts to fail,
    making critical width transitions clear.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        output_path: Optional save path
        threshold: QV threshold
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    config_names = sorted(campaign_results.keys())
    n_configs = len(config_names)
    
    # Determine global width range
    all_widths = set()
    for results in campaign_results.values():
        all_widths.update(results.keys())
    widths = sorted(all_widths)
    
    # Create color map
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_configs - 1, 1)) for i in range(n_configs)]
    
    # Plot each config as a separate line with offset
    y_offset = 0
    y_step = 0.15
    
    for idx, config_name in enumerate(config_names):
        results = campaign_results[config_name]
        config_widths = sorted(results.keys())
        means = [results[w]["mean_hop"] for w in config_widths]
        
        # Offset y-values for waterfall effect
        offset_means = [m + y_offset for m in means]
        
        # Plot line
        ax.plot(config_widths, offset_means, 'o-', 
               linewidth=2.5, markersize=8, 
               color=colors[idx], label=config_name, alpha=0.85)
        
        # Highlight critical width (last passing)
        passing_widths = [w for w in config_widths if results[w].get("pass_qv", False)]
        if passing_widths:
            critical_width = max(passing_widths)
            critical_idx = config_widths.index(critical_width)
            ax.plot(critical_width, offset_means[critical_idx], '*', 
                   markersize=20, color='gold', 
                   markeredgecolor='darkred', markeredgewidth=2, zorder=5)
        
        # Add config label
        if config_widths:
            ax.text(config_widths[0] - 0.3, offset_means[0], config_name, 
                   ha='right', va='center', fontsize=9, fontweight='bold',
                   color=colors[idx])
        
        y_offset += y_step
    
    # Add threshold reference line
    ax.axhline(y=threshold, color='red', linestyle='--', 
              linewidth=2, alpha=0.6, label=f'Threshold ({threshold:.3f})', zorder=0)
    
    # Formatting
    ax.set_xlabel('Circuit Width (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Heavy-Output Probability (with offset)', fontweight='bold', fontsize=12)
    ax.set_title('QV Waterfall: Performance Degradation Across Configurations', 
                fontweight='bold', fontsize=14, pad=20)
    
    ax.set_xticks(widths)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved waterfall plot to {output_path}")
    
    return fig


def plot_campaign_dashboard(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    campaign_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    threshold: float = 2.0 / 3.0,
) -> plt.Figure:
    """
    Comprehensive dashboard summarizing entire campaign results.
    
    Multi-panel plot combining:
    1. HOP comparison across configs
    2. Achieved QV bar chart
    3. Critical width analysis
    4. Statistical summary table
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        campaign_configs: Optional dict with config parameters
        output_path: Optional save path
        threshold: QV threshold
    
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    config_names = sorted(campaign_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    # ========== Panel 1: HOP Comparison ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    for idx, config_name in enumerate(config_names):
        results = campaign_results[config_name]
        widths = sorted(results.keys())
        means = [results[w]["mean_hop"] for w in widths]
        ci_lower = [results[w]["ci_lower"] for w in widths]
        ci_upper = [results[w]["ci_upper"] for w in widths]
        
        ax1.plot(widths, means, 'o-', linewidth=2, markersize=7, 
                label=config_name, color=colors[idx], alpha=0.9)
        ax1.fill_between(widths, ci_lower, ci_upper, 
                        alpha=0.15, color=colors[idx])
    
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax1.set_ylabel('Heavy-Output Probability', fontweight='bold')
    ax1.set_title('HOP vs Width: All Configurations', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    
    # ========== Panel 2: Achieved QV ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    achieved_qvs = []
    for config_name in config_names:
        results = campaign_results[config_name]
        passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
        max_width = max(passing_widths) if passing_widths else 0
        achieved_qvs.append(2 ** max_width if max_width > 0 else 0)
    
    y_pos = np.arange(len(config_names))
    bars = ax2.barh(y_pos, achieved_qvs, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(config_names, fontsize=9)
    ax2.set_xlabel('Quantum Volume', fontweight='bold')
    ax2.set_title('Achieved QV by Configuration', fontweight='bold', fontsize=12)
    
    if max(achieved_qvs) > 0:
        ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ========== Panel 3: Critical Width Analysis ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    critical_widths = []
    for config_name in config_names:
        results = campaign_results[config_name]
        passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
        critical_widths.append(max(passing_widths) if passing_widths else 0)
    
    ax3.bar(range(len(config_names)), critical_widths, 
           color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Maximum Passing Width', fontweight='bold')
    ax3.set_title('Critical Width Analysis', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== Panel 4: Statistics Summary ==========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Build summary table
    table_data = [['Configuration', 'Max Width', 'QV', 'Mean HOP @Max', '# Circuits', 'Pass Rate']]
    
    for config_name in config_names:
        results = campaign_results[config_name]
        passing_widths = [w for w, r in results.items() if r.get("pass_qv", False)]
        
        if passing_widths:
            max_width = max(passing_widths)
            qv = 2 ** max_width
            mean_hop = results[max_width]["mean_hop"]
            n_circuits = results[max_width]["n_circuits"]
            pass_rate = f"{len(passing_widths)}/{len(results)}"
        else:
            max_width = 0
            qv = 0
            mean_hop = 0
            n_circuits = 0
            pass_rate = "0/" + str(len(results))
        
        table_data.append([
            config_name,
            f"{max_width}",
            f"{qv}",
            f"{mean_hop:.4f}",
            f"{n_circuits}",
            pass_rate
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.15, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
    
    ax4.set_title('Campaign Summary Statistics', fontweight='bold', 
                 fontsize=13, pad=20, loc='left')
    
    plt.suptitle('Campaign Dashboard: Comprehensive QV Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved campaign dashboard to {output_path}")
    
    return fig


def plot_3d_parameter_surface(
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    qv_grid: np.ndarray,
    param1_name: str = "Parameter 1",
    param2_name: str = "Parameter 2",
    output_path: Optional[Path] = None,
    title: str = "QV Parameter Surface",
    elevation: float = 30,
    azimuth: float = 45,
) -> plt.Figure:
    """
    3D surface plot showing QV as function of two parameters.
    
    Key insight: Reveals interaction effects and optimal parameter regions
    in 3D space.
    
    Args:
        param1_values: 1D array of first parameter values
        param2_values: 1D array of second parameter values
        qv_grid: 2D grid of achieved QV (or max width)
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        output_path: Optional save path
        title: Plot title
        elevation: View elevation angle
        azimuth: View azimuth angle
    
    Returns:
        Matplotlib figure object
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    X, Y = np.meshgrid(param1_values, param2_values)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, qv_grid, cmap='viridis', 
                          alpha=0.9, edgecolor='none', 
                          linewidth=0, antialiased=True)
    
    # Contours on bottom plane
    ax.contour(X, Y, qv_grid, zdir='z', offset=np.min(qv_grid) - 1, 
              cmap='viridis', alpha=0.5, linewidths=1)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Achieved Width (m)', rotation=270, labelpad=20, fontweight='bold')
    
    # Labels
    ax.set_xlabel(param1_name, fontweight='bold', labelpad=10)
    ax.set_ylabel(param2_name, fontweight='bold', labelpad=10)
    ax.set_zlabel('Max Width', fontweight='bold', labelpad=10)
    ax.set_title(title, fontweight='bold', fontsize=13, pad=20)
    
    # View angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved 3D parameter surface to {output_path}")
    
    return fig


def plot_critical_transitions(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Optional[Path] = None,
    threshold: float = 2.0 / 3.0,
) -> plt.Figure:
    """
    Identify and visualize critical width transitions (last pass -> first fail).
    
    Key insight: Shows the "cliff" where each configuration drops below threshold,
    highlighting the stability margin.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        output_path: Optional save path
        threshold: QV threshold
    
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    config_names = sorted(campaign_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))
    
    transition_data = []
    
    for idx, config_name in enumerate(config_names):
        results = campaign_results[config_name]
        widths = sorted(results.keys())
        means = [results[w]["mean_hop"] for w in widths]
        
        # Find transition point
        passing_widths = [w for w in widths if results[w].get("pass_qv", False)]
        
        if passing_widths:
            last_pass = max(passing_widths)
            failing_widths = [w for w in widths if w > last_pass]
            first_fail = min(failing_widths) if failing_widths else None
            
            transition_width = last_pass
            transition_hop = results[last_pass]["mean_hop"]
            
            # Margin to threshold
            margin = transition_hop - threshold
            
            transition_data.append({
                'config': config_name,
                'last_pass': last_pass,
                'first_fail': first_fail,
                'margin': margin,
                'hop_at_transition': transition_hop
            })
        else:
            transition_data.append({
                'config': config_name,
                'last_pass': 0,
                'first_fail': widths[0] if widths else None,
                'margin': 0,
                'hop_at_transition': 0
            })
        
        # Plot full curve
        ax1.plot(widths, means, 'o-', linewidth=2, markersize=6, 
                label=config_name, color=colors[idx], alpha=0.8)
        
        # Highlight transition region
        if transition_data[-1]['last_pass'] > 0:
            trans_w = transition_data[-1]['last_pass']
            trans_idx = widths.index(trans_w)
            ax1.plot(trans_w, means[trans_idx], 'D', 
                    markersize=15, color=colors[idx], 
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    # Threshold line
    ax1.axhline(y=threshold, color='red', linestyle='--', 
               linewidth=2.5, alpha=0.7, label='Threshold')
    
    ax1.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax1.set_ylabel('Heavy-Output Probability', fontweight='bold')
    ax1.set_title('Critical Width Transitions (Diamonds)', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    
    # ========== Lower panel: Margin analysis ==========
    margins = [t['margin'] for t in transition_data]
    last_passes = [t['last_pass'] for t in transition_data]
    
    # Scatter plot: last pass width vs margin
    for idx, (config_name, margin, last_pass) in enumerate(zip(config_names, margins, last_passes)):
        ax2.scatter(last_pass, margin, s=150, color=colors[idx], 
                   edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)
        ax2.text(last_pass + 0.1, margin, config_name, 
                fontsize=9, va='center', ha='left')
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Last Passing Width', fontweight='bold')
    ax2.set_ylabel('Margin to Threshold', fontweight='bold')
    ax2.set_title('Stability Margin at Critical Width', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Saved critical transitions plot to {output_path}")
    
    return fig


def save_all_campaign_plots(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    campaign_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Path = Path("campaign_plots"),
    formats: List[str] = ["png", "svg"],
) -> None:
    """
    Generate and save all campaign-level plots.
    
    Args:
        campaign_results: Dict mapping config_name -> {width -> statistics}
        campaign_configs: Optional dict mapping config_name -> full config dict
        output_dir: Directory to save plots
        formats: List of file formats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERATING CAMPAIGN PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Configurations analyzed: {len(campaign_results)}")
    print()
    
    for fmt in formats:
        # 1. Campaign comparison
        print(f"[1/7] Generating campaign comparison plot...")
        plot_campaign_comparison(
            campaign_results,
            output_path=output_dir / f"campaign_comparison.{fmt}",
        )
        plt.close()
        
        # 2. Achieved QV bar chart
        print(f"[2/7] Generating achieved QV comparison...")
        plot_achieved_qv_comparison(
            campaign_results,
            output_path=output_dir / f"achieved_qv_comparison.{fmt}",
        )
        plt.close()
        
        # 3. Waterfall plot
        print(f"[3/7] Generating waterfall plot...")
        plot_qv_waterfall(
            campaign_results,
            output_path=output_dir / f"qv_waterfall.{fmt}",
        )
        plt.close()
        
        # 4. Critical transitions
        print(f"[4/7] Generating critical transitions plot...")
        plot_critical_transitions(
            campaign_results,
            output_path=output_dir / f"critical_transitions.{fmt}",
        )
        plt.close()
        
        # 5. Campaign dashboard
        print(f"[5/7] Generating campaign dashboard...")
        plot_campaign_dashboard(
            campaign_results,
            campaign_configs=campaign_configs,
            output_path=output_dir / f"campaign_dashboard.{fmt}",
        )
        plt.close()
        
        # 6. Parameter correlation (if configs provided)
        if campaign_configs is not None:
            print(f"[6/7] Generating parameter correlation matrix...")
            plot_parameter_correlation_matrix(
                campaign_results,
                campaign_configs,
                output_path=output_dir / f"parameter_correlation.{fmt}",
            )
            plt.close()
        else:
            print(f"[6/7] Skipping correlation matrix (no configs provided)")
        
        # 7. 3D surface (if we have grid data)
        # This requires structured grid data, skip for now
        print(f"[7/7] Skipping 3D surface (requires grid structure)")
    
    print()
    print(f"{'='*60}")
    print(f"[✓] ALL CAMPAIGN PLOTS SAVED TO: {output_dir}/")
    print(f"{'='*60}\n")
