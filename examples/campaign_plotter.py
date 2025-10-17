"""
Campaign Plotting Script

Creates width-grouped plots showing QV results across different device configurations,
with both per-config detail plots and global comparison plots.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def plot_hop_by_width(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
    threshold: float = 2.0 / 3.0,
) -> None:
    """
    Create individual plots for each width showing HOP across all configurations.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
        threshold: QV success threshold
    """
    plots_dir = output_dir / "plots" / "by_width"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all widths (skip None results)
    all_widths = set()
    for config_results in campaign_results.values():
        if config_results is not None:  # Skip failed configs
            all_widths.update(config_results.keys())
    
    if not all_widths:
        print("  [WARNING] No successful results to plot")
        return
    
    widths = sorted(all_widths)
    
    print(f"\n[Plotting] Creating per-width HOP comparison plots...")
    
    for width in widths:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        config_names = []
        mean_hops = []
        ci_lowers = []
        ci_uppers = []
        pass_flags = []
        
        # Collect data for this width across all configs
        for config_name in sorted(campaign_results.keys()):
            if campaign_results[config_name] is None:  # Skip failed configs
                continue
            if width in campaign_results[config_name]:
                stats = campaign_results[config_name][width]
                config_names.append(config_name)
                mean_hops.append(stats["mean_hop"])
                ci_lowers.append(stats["ci_lower"])
                ci_uppers.append(stats["ci_upper"])
                pass_flags.append(stats.get("pass_qv", False))
        
        # Create bar plot
        x_pos = np.arange(len(config_names))
        colors = ['#2E86AB' if passed else '#A23B72' for passed in pass_flags]
        
        bars = ax.bar(x_pos, mean_hops, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
        
        # Add error bars (confidence intervals)
        errors_lower = [mean - ci_low for mean, ci_low in zip(mean_hops, ci_lowers)]
        errors_upper = [ci_up - mean for mean, ci_up in zip(mean_hops, ci_uppers)]
        
        ax.errorbar(x_pos, mean_hops, 
                   yerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='black', capsize=5, 
                   linewidth=1.5, capthick=1.5)
        
        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'QV Threshold ({threshold:.3f})', zorder=2)
        
        # Formatting
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Heavy-Output Probability', fontweight='bold')
        ax.set_title(f'HOP Comparison at Width m={width}', fontweight='bold', pad=15)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', alpha=0.7, label='Passed QV'),
            Patch(facecolor='#A23B72', alpha=0.7, label='Failed QV'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold:.3f})')
        ]
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        output_path = plots_dir / f"hop_width_{width}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"  ✓ Saved width m={width} plot: {output_path}")


def plot_qv_by_width(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """
    Create individual plots for each width showing achieved QV across configurations.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots" / "by_width"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Plotting] Creating per-width achieved QV plots...")
    
    # Get all widths (skip None results)
    all_widths = set()
    for config_results in campaign_results.values():
        if config_results is not None:  # Skip failed configs
            all_widths.update(config_results.keys())
    
    if not all_widths:
        print("  [WARNING] No successful results to plot")
        return
    
    widths = sorted(all_widths)
    
    for width in widths:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        config_names = []
        qv_values = []
        colors_list = []
        
        # Collect data
        for config_name in sorted(campaign_results.keys()):
            if campaign_results[config_name] is None:  # Skip failed configs
                continue
            if width in campaign_results[config_name]:
                stats = campaign_results[config_name][width]
                config_names.append(config_name)
                
                # QV = 2^m if passed, else 0
                if stats.get("pass_qv", False):
                    qv_values.append(2 ** width)
                    colors_list.append('#2E86AB')
                else:
                    qv_values.append(0)
                    colors_list.append('#A23B72')
        
        # Create bar plot
        x_pos = np.arange(len(config_names))
        bars = ax.bar(x_pos, qv_values, color=colors_list, alpha=0.7,
                     edgecolor='black', linewidth=1.2)
        
        # Formatting
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Quantum Volume (QV)', fontweight='bold')
        ax.set_title(f'Achieved Quantum Volume at Width m={width}', fontweight='bold', pad=15)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        
        # Use log scale if values vary greatly
        max_qv = max(qv_values) if qv_values else 1
        if max_qv > 0:
            ax.set_ylim([0, max_qv * 1.1])
            if max_qv >= 16:
                ax.set_yscale('log', base=2)
                ax.set_ylabel('Quantum Volume (QV) - log₂ scale', fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', alpha=0.7, label='Passed QV'),
            Patch(facecolor='#A23B72', alpha=0.7, label='Failed QV'),
        ]
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        output_path = plots_dir / f"qv_width_{width}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"  ✓ Saved width m={width} QV plot: {output_path}")


def plot_global_width_average(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
    threshold: float = 2.0 / 3.0,
) -> None:
    """
    Create global plot showing average HOP across all widths for each configuration.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
        threshold: QV success threshold
    """
    plots_dir = output_dir / "plots" / "global"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Plotting] Creating global width-average plots...")
    
    config_names = []
    avg_hops = []
    std_hops = []
    max_widths = []
    
    # Calculate averages for each config
    for config_name in sorted(campaign_results.keys()):
        config_results = campaign_results[config_name]
        
        if config_results is None:  # Skip failed configs
            continue
        
        if not config_results:
            continue
        
        # Average HOP across all widths
        hops = [stats["mean_hop"] for stats in config_results.values()]
        config_names.append(config_name)
        avg_hops.append(np.mean(hops))
        std_hops.append(np.std(hops))
        
        # Maximum width that passed QV
        passed_widths = [w for w, stats in config_results.items() 
                        if stats.get("pass_qv", False)]
        max_widths.append(max(passed_widths) if passed_widths else 0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average HOP
    x_pos = np.arange(len(config_names))
    colors = ['#2E86AB' if mw > 0 else '#A23B72' for mw in max_widths]
    
    ax1.bar(x_pos, avg_hops, color=colors, alpha=0.7,
           edgecolor='black', linewidth=1.2, yerr=std_hops,
           capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    ax1.axhline(y=threshold, color='red', linestyle='--', 
               linewidth=2, label=f'QV Threshold ({threshold:.3f})')
    
    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Average HOP (across all widths)', fontweight='bold')
    ax1.set_title('Global Performance: Width-Averaged HOP', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
    ax1.legend(loc='best', framealpha=0.9)
    
    # Plot 2: Maximum achieved width
    max_qv = [2 ** w if w > 0 else 0 for w in max_widths]
    
    ax2.bar(x_pos, max_widths, color=colors, alpha=0.7,
           edgecolor='black', linewidth=1.2)
    
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Maximum Width Passed', fontweight='bold')
    ax2.set_title('Global Performance: Maximum QV Achieved', fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    
    # Add secondary y-axis for QV values
    ax2_qv = ax2.twinx()
    ax2_qv.set_ylabel('Quantum Volume (QV = 2^m)', fontweight='bold', color='#555555')
    ax2_qv.set_ylim([0, max(max_qv) * 1.1 if max_qv else 1])
    
    # Add QV labels on bars
    for i, (w, qv) in enumerate(zip(max_widths, max_qv)):
        if w > 0:
            ax2.text(i, w + 0.15, f'QV={int(qv)}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
    
    plt.tight_layout()
    
    output_path = plots_dir / "global_width_average.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved global average plot: {output_path}")


def plot_config_trajectories(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
    threshold: float = 2.0 / 3.0,
) -> None:
    """
    Create line plot showing HOP trajectories across widths for all configurations.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
        threshold: QV success threshold
    """
    plots_dir = output_dir / "plots" / "global"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Plotting] Creating configuration trajectory plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color map for configs
    n_configs = len(campaign_results)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_configs))
    
    for idx, (config_name, config_results) in enumerate(sorted(campaign_results.items())):
        if config_results is None:  # Skip failed configs
            continue
        
        widths = sorted(config_results.keys())
        means = [config_results[w]["mean_hop"] for w in widths]
        ci_lowers = [config_results[w]["ci_lower"] for w in widths]
        ci_uppers = [config_results[w]["ci_upper"] for w in widths]
        
        # Plot line with markers
        ax.plot(widths, means, 'o-', linewidth=2, markersize=6,
               label=config_name, color=colors[idx], alpha=0.8)
        
        # Add confidence interval
        ax.fill_between(widths, ci_lowers, ci_uppers, 
                       alpha=0.15, color=colors[idx])
    
    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', 
              linewidth=2.5, label=f'QV Threshold ({threshold:.3f})', zorder=1)
    
    # Formatting
    ax.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax.set_ylabel('Heavy-Output Probability', fontweight='bold')
    ax.set_title('HOP Trajectories Across Widths: All Configurations', 
                fontweight='bold', pad=15)
    
    all_widths = set()
    for config_results in campaign_results.values():
        all_widths.update(config_results.keys())
    ax.set_xticks(sorted(all_widths))
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Legend - place outside plot if many configs
    if n_configs > 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 framealpha=0.95, fontsize=8)
    else:
        ax.legend(loc='best', framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    output_path = plots_dir / "hop_trajectories.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved trajectory plot: {output_path}")


def plot_qv_heatmap(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """
    Create heatmap showing QV pass/fail status for all configs and widths.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots" / "global"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Plotting] Creating QV pass/fail heatmap...")
    
    # Build matrix
    config_names = sorted(campaign_results.keys())
    all_widths = set()
    for config_results in campaign_results.values():
        if config_results is not None:  # Skip failed configs
            all_widths.update(config_results.keys())
    
    if not all_widths:
        print("  [WARNING] No successful results to plot heatmap")
        return
    
    widths = sorted(all_widths)
    
    # Create matrix: 1 = pass, 0 = fail, -1 = not tested
    matrix = np.full((len(config_names), len(widths)), -1.0)
    
    for i, config_name in enumerate(config_names):
        config_results = campaign_results[config_name]
        if config_results is None:  # Failed config - mark all as not tested
            continue
        
        for j, width in enumerate(widths):
            if width in config_results:
                stats = config_results[width]
                matrix[i, j] = 1.0 if stats.get("pass_qv", False) else 0.0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(widths) * 0.8), 
                                    max(6, len(config_names) * 0.4)))
    
    # Custom colormap: gray (not tested), red (fail), green (pass)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#CCCCCC', '#E63946', '#06D6A0']  # gray, red, green
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(len(widths)):
            val = matrix[i, j]
            if val == 1.0:
                text = '✓'
                color = 'white'
            elif val == 0.0:
                text = '✗'
                color = 'white'
            else:
                text = '—'
                color = 'black'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xticks(np.arange(len(widths)))
    ax.set_yticks(np.arange(len(config_names)))
    ax.set_xticklabels(widths)
    ax.set_yticklabels(config_names)
    
    ax.set_xlabel('Circuit Width (m)', fontweight='bold')
    ax.set_ylabel('Configuration', fontweight='bold')
    ax.set_title('QV Pass/Fail Matrix: All Configurations', fontweight='bold', pad=15)
    
    # Colorbar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#06D6A0', label='Passed QV'),
        Patch(facecolor='#E63946', label='Failed QV'),
        Patch(facecolor='#CCCCCC', label='Not Tested'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
             loc='upper left', framealpha=0.95)
    
    plt.tight_layout()
    
    output_path = plots_dir / "qv_heatmap.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"  ✓ Saved QV heatmap: {output_path}")


def create_all_campaign_plots(
    campaign_results: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: Path,
    threshold: float = 2.0 / 3.0,
) -> None:
    """
    Generate all campaign plots: per-width and global views.
    
    Args:
        campaign_results: Results dict {config_name -> {width -> stats}}
        output_dir: Directory to save plots
        threshold: QV success threshold
    """
    print("\n" + "=" * 80)
    print("GENERATING CAMPAIGN PLOTS".center(80))
    print("=" * 80)
    
    # Per-width plots
    print("\n[1/5] Per-Width HOP Comparisons")
    plot_hop_by_width(campaign_results, output_dir, threshold)
    
    print("\n[2/5] Per-Width QV Achievement")
    plot_qv_by_width(campaign_results, output_dir)
    
    # Global plots
    print("\n[3/5] Global Width-Averaged Performance")
    plot_global_width_average(campaign_results, output_dir, threshold)
    
    print("\n[4/5] Configuration Trajectories")
    plot_config_trajectories(campaign_results, output_dir, threshold)
    
    print("\n[5/5] QV Pass/Fail Heatmap")
    plot_qv_heatmap(campaign_results, output_dir)
    
    print("\n" + "=" * 80)
    print(f"✓ All campaign plots saved to: {output_dir / 'plots'}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test with mock data
    import json
    
    mock_results = {
        "baseline": {
            2: {"mean_hop": 0.85, "ci_lower": 0.82, "ci_upper": 0.88, "pass_qv": True},
            3: {"mean_hop": 0.75, "ci_lower": 0.71, "ci_upper": 0.79, "pass_qv": True},
            4: {"mean_hop": 0.68, "ci_lower": 0.63, "ci_upper": 0.73, "pass_qv": True},
            5: {"mean_hop": 0.58, "ci_lower": 0.52, "ci_upper": 0.64, "pass_qv": False},
        },
        "high_fidelity": {
            2: {"mean_hop": 0.90, "ci_lower": 0.87, "ci_upper": 0.93, "pass_qv": True},
            3: {"mean_hop": 0.82, "ci_lower": 0.78, "ci_upper": 0.86, "pass_qv": True},
            4: {"mean_hop": 0.72, "ci_lower": 0.67, "ci_upper": 0.77, "pass_qv": True},
            5: {"mean_hop": 0.65, "ci_lower": 0.59, "ci_upper": 0.71, "pass_qv": False},
        },
        "low_coherence": {
            2: {"mean_hop": 0.80, "ci_lower": 0.76, "ci_upper": 0.84, "pass_qv": True},
            3: {"mean_hop": 0.65, "ci_lower": 0.60, "ci_upper": 0.70, "pass_qv": False},
            4: {"mean_hop": 0.52, "ci_lower": 0.46, "ci_upper": 0.58, "pass_qv": False},
        }
    }
    
    output_dir = Path("test_campaign_plots")
    output_dir.mkdir(exist_ok=True)
    
    create_all_campaign_plots(mock_results, output_dir)
    
    print("\n✓ Test plots generated successfully!")
