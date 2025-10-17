"""
Demo script for campaign-level plotting functions.

Demonstrates how to use the new campaign analysis plots to compare
multiple QV configurations and visualize parameter correlations.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinq_qv.analysis.plots import (
    plot_campaign_comparison,
    plot_achieved_qv_comparison,
    plot_parameter_correlation_matrix,
    plot_qv_waterfall,
    plot_campaign_dashboard,
    plot_critical_transitions,
    plot_3d_parameter_surface,
    save_all_campaign_plots,
)


def generate_mock_campaign_results():
    """
    Generate mock campaign results for demonstration.
    
    Simulates 4 different device configurations with varying performance.
    """
    np.random.seed(42)
    
    campaign_results = {}
    campaign_configs = {}
    
    # Configuration 1: Baseline (moderate performance)
    baseline = {}
    for width in [2, 3, 4, 5, 6, 7, 8]:
        # HOP degrades with width
        mean_hop = 0.95 - 0.08 * (width - 2)
        mean_hop = max(0.5, mean_hop)  # Floor at 0.5
        
        baseline[width] = {
            "mean_hop": mean_hop,
            "ci_lower": mean_hop - 0.03,
            "ci_upper": mean_hop + 0.03,
            "n_circuits": 50,
            "pass_qv": mean_hop > 2/3 and mean_hop - 0.03 > 2/3,
            "hops": np.random.normal(mean_hop, 0.02, 50),
        }
    
    campaign_results["baseline"] = baseline
    campaign_configs["baseline"] = {
        "device": {
            "F1": 0.99926,
            "F2": 0.998,
            "T1": 1.0,
            "T2": 99e-6,
            "T2_star": 20e-6,
            "t_single_gate": 60e-9,
            "t_two_gate": 40e-9,
            "F_readout": 0.9997,
            "F_init": 0.994,
        }
    }
    
    # Configuration 2: High fidelity (better performance)
    high_fidelity = {}
    for width in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        mean_hop = 0.98 - 0.06 * (width - 2)
        mean_hop = max(0.55, mean_hop)
        
        high_fidelity[width] = {
            "mean_hop": mean_hop,
            "ci_lower": mean_hop - 0.025,
            "ci_upper": mean_hop + 0.025,
            "n_circuits": 50,
            "pass_qv": mean_hop > 2/3 and mean_hop - 0.025 > 2/3,
            "hops": np.random.normal(mean_hop, 0.015, 50),
        }
    
    campaign_results["high_fidelity"] = high_fidelity
    campaign_configs["high_fidelity"] = {
        "device": {
            "F1": 0.9999,
            "F2": 0.9995,
            "T1": 2.0,
            "T2": 200e-6,
            "T2_star": 50e-6,
            "t_single_gate": 40e-9,
            "t_two_gate": 150e-9,
            "F_readout": 0.999,
            "F_init": 0.998,
        }
    }
    
    # Configuration 3: Long coherence (intermediate)
    long_coherence = {}
    for width in [2, 3, 4, 5, 6, 7, 8, 9]:
        mean_hop = 0.96 - 0.07 * (width - 2)
        mean_hop = max(0.52, mean_hop)
        
        long_coherence[width] = {
            "mean_hop": mean_hop,
            "ci_lower": mean_hop - 0.028,
            "ci_upper": mean_hop + 0.028,
            "n_circuits": 50,
            "pass_qv": mean_hop > 2/3 and mean_hop - 0.028 > 2/3,
            "hops": np.random.normal(mean_hop, 0.018, 50),
        }
    
    campaign_results["long_coherence"] = long_coherence
    campaign_configs["long_coherence"] = {
        "device": {
            "F1": 0.999,
            "F2": 0.997,
            "T1": 3.0,
            "T2": 300e-6,
            "T2_star": 60e-6,
            "t_single_gate": 60e-9,
            "t_two_gate": 40e-9,
            "F_readout": 0.998,
            "F_init": 0.995,
        }
    }
    
    # Configuration 4: Fast gates (lower performance due to fidelity tradeoff)
    fast_gates = {}
    for width in [2, 3, 4, 5, 6, 7]:
        mean_hop = 0.93 - 0.09 * (width - 2)
        mean_hop = max(0.48, mean_hop)
        
        fast_gates[width] = {
            "mean_hop": mean_hop,
            "ci_lower": mean_hop - 0.035,
            "ci_upper": mean_hop + 0.035,
            "n_circuits": 50,
            "pass_qv": mean_hop > 2/3 and mean_hop - 0.035 > 2/3,
            "hops": np.random.normal(mean_hop, 0.022, 50),
        }
    
    campaign_results["fast_gates"] = fast_gates
    campaign_configs["fast_gates"] = {
        "device": {
            "F1": 0.998,
            "F2": 0.995,
            "T1": 1.0,
            "T2": 99e-6,
            "T2_star": 20e-6,
            "t_single_gate": 30e-9,
            "t_two_gate": 20e-9,
            "F_readout": 0.9995,
            "F_init": 0.992,
        }
    }
    
    return campaign_results, campaign_configs


def demo_individual_plots(campaign_results, campaign_configs, output_dir):
    """Generate individual campaign plots with custom options."""
    
    print("\n" + "="*70)
    print("DEMO: INDIVIDUAL CAMPAIGN PLOTS")
    print("="*70)
    
    # 1. Campaign comparison
    print("\n[1/7] Campaign Comparison Plot")
    print("   Purpose: Overlay all configurations to directly compare HOP curves")
    plot_campaign_comparison(
        campaign_results,
        output_path=output_dir / "demo_comparison.png",
        show_ci=True,
        title="QV Campaign: Device Configuration Comparison",
    )
    print("   ✓ Saved to demo_comparison.png")
    
    # 2. Achieved QV bar chart
    print("\n[2/7] Achieved QV Comparison")
    print("   Purpose: Show maximum QV achieved by each configuration")
    plot_achieved_qv_comparison(
        campaign_results,
        output_path=output_dir / "demo_achieved_qv.png",
        show_values=True,
    )
    print("   ✓ Saved to demo_achieved_qv.png")
    
    # 3. Waterfall plot
    print("\n[3/7] QV Waterfall Plot")
    print("   Purpose: Visualize degradation cascade across configurations")
    plot_qv_waterfall(
        campaign_results,
        output_path=output_dir / "demo_waterfall.png",
    )
    print("   ✓ Saved to demo_waterfall.png")
    
    # 4. Critical transitions
    print("\n[4/7] Critical Transitions Analysis")
    print("   Purpose: Identify where each config drops below threshold")
    plot_critical_transitions(
        campaign_results,
        output_path=output_dir / "demo_critical_transitions.png",
    )
    print("   ✓ Saved to demo_critical_transitions.png")
    
    # 5. Campaign dashboard
    print("\n[5/7] Campaign Dashboard (Multi-panel)")
    print("   Purpose: Comprehensive overview with statistics table")
    plot_campaign_dashboard(
        campaign_results,
        campaign_configs=campaign_configs,
        output_path=output_dir / "demo_dashboard.png",
    )
    print("   ✓ Saved to demo_dashboard.png")
    
    # 6. Parameter correlation
    print("\n[6/7] Parameter Correlation Matrix")
    print("   Purpose: Identify which parameters correlate with QV success")
    plot_parameter_correlation_matrix(
        campaign_results,
        campaign_configs,
        output_path=output_dir / "demo_correlation.png",
    )
    print("   ✓ Saved to demo_correlation.png")
    
    # 7. 3D parameter surface
    print("\n[7/7] 3D Parameter Surface")
    print("   Purpose: Visualize QV as function of two parameters")
    
    # Generate mock grid data
    F1_values = np.linspace(0.995, 0.9999, 10)
    F2_values = np.linspace(0.990, 0.9995, 10)
    qv_grid = np.zeros((len(F2_values), len(F1_values)))
    
    for i, f2 in enumerate(F2_values):
        for j, f1 in enumerate(F1_values):
            # Mock relationship: QV improves with fidelities
            base_width = 3.0
            f1_contribution = 8.0 * (f1 - 0.995) / (0.9999 - 0.995)
            f2_contribution = 5.0 * (f2 - 0.990) / (0.9995 - 0.990)
            qv_grid[i, j] = base_width + f1_contribution + f2_contribution
    
    plot_3d_parameter_surface(
        F1_values,
        F2_values,
        qv_grid,
        param1_name="F1 (Single-Qubit Fidelity)",
        param2_name="F2 (Two-Qubit Fidelity)",
        output_path=output_dir / "demo_3d_surface.png",
        title="QV Parameter Space: F1 vs F2",
    )
    print("   ✓ Saved to demo_3d_surface.png")
    
    print("\n" + "="*70)
    print("✓ ALL INDIVIDUAL PLOTS GENERATED")
    print("="*70)


def demo_batch_generation(campaign_results, campaign_configs, output_dir):
    """Demonstrate batch generation of all campaign plots."""
    
    print("\n" + "="*70)
    print("DEMO: BATCH PLOT GENERATION")
    print("="*70)
    print("\nGenerating all campaign plots in PNG and SVG formats...")
    
    batch_dir = output_dir / "batch"
    save_all_campaign_plots(
        campaign_results,
        campaign_configs=campaign_configs,
        output_dir=batch_dir,
        formats=["png"],  # Use only PNG for demo (faster)
    )
    
    print("\n✓ Batch generation complete!")


def print_campaign_summary(campaign_results):
    """Print textual summary of campaign results."""
    
    print("\n" + "="*70)
    print("CAMPAIGN SUMMARY")
    print("="*70)
    
    for config_name, results in sorted(campaign_results.items()):
        widths = sorted(results.keys())
        passing_widths = [w for w in widths if results[w].get("pass_qv", False)]
        
        if passing_widths:
            max_width = max(passing_widths)
            qv = 2 ** max_width
            mean_hop = results[max_width]["mean_hop"]
            status = "✓ PASS"
        else:
            max_width = 0
            qv = 0
            mean_hop = 0
            status = "✗ FAIL"
        
        print(f"\n{config_name.upper()}")
        print(f"  Status: {status}")
        print(f"  Max passing width: {max_width}")
        print(f"  Achieved QV: {qv}")
        if max_width > 0:
            print(f"  HOP at max width: {mean_hop:.4f}")
        print(f"  Widths tested: {len(widths)}")
        print(f"  Pass rate: {len(passing_widths)}/{len(widths)}")
    
    print("\n" + "="*70)


def main():
    """Run campaign plotting demo."""
    
    print("\n" + "="*70)
    print("SPINQ QV CAMPAIGN PLOTTING DEMO")
    print("="*70)
    print("\nThis demo showcases advanced campaign-level plotting functions")
    print("for comparing multiple Quantum Volume configurations.\n")
    
    # Setup output directory
    output_dir = Path(__file__).parent / "demo_campaign_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    # Generate mock data
    print("Generating mock campaign data (4 configurations)...")
    campaign_results, campaign_configs = generate_mock_campaign_results()
    print(f"  ✓ Generated results for {len(campaign_results)} configurations")
    
    # Print summary
    print_campaign_summary(campaign_results)
    
    # Generate individual plots
    demo_individual_plots(campaign_results, campaign_configs, output_dir)
    
    # Generate batch plots
    demo_batch_generation(campaign_results, campaign_configs, output_dir)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nKey plots to review:")
    print("  • demo_comparison.png - Overlaid HOP curves")
    print("  • demo_dashboard.png - Comprehensive overview")
    print("  • demo_correlation.png - Parameter correlations")
    print("  • demo_3d_surface.png - 3D parameter space")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
