"""
Quick Demo: Parameter Campaign System

Demonstrates the campaign system with a small test run.
Uses reduced settings for fast execution.
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from campaign_config_generator import generate_parameter_sweep_configs
from spinq_qv.config.schemas import Config


def create_test_config():
    """Create a lightweight test configuration."""
    # Load production config
    prod_config = Config.from_yaml(Path("examples/configs/production.yaml"))
    
    # Modify for quick testing
    prod_config.simulation.widths = [2, 3, 4]  # Only small widths
    prod_config.simulation.n_circuits = 5      # Few circuits
    prod_config.simulation.n_shots = 100       # Few shots
    
    # Save test config
    test_config_path = Path("examples/configs/test_campaign.yaml")
    prod_config.to_yaml(test_config_path)
    
    print(f"[✓] Created test config: {test_config_path}")
    return test_config_path


def test_config_generation():
    """Test configuration generation."""
    print("\n" + "="*70)
    print("TEST 1: Configuration Generation")
    print("="*70)
    
    # Create test config
    test_config = create_test_config()
    
    # Test each sweep type
    sweep_types = ["comprehensive", "fidelity_focus", "coherence_focus"]
    
    for sweep_type in sweep_types:
        print(f"\n[{sweep_type}]")
        
        configs, params = generate_parameter_sweep_configs(
            base_config_path=test_config,
            sweep_type=sweep_type,
            n_points=3,  # Small for testing
            output_dir=Path("test_campaign_output") / sweep_type,
        )
        
        print(f"  ✓ Generated {len(configs)} configurations")
        print(f"  ✓ Parameters swept: {list(params.keys())}")
        
        # Show sample config names
        sample_names = list(configs.keys())[:5]
        print(f"  ✓ Sample configs: {', '.join(sample_names)}")


def test_campaign_structure():
    """Test campaign execution structure (dry run)."""
    print("\n" + "="*70)
    print("TEST 2: Campaign Structure (Dry Run)")
    print("="*70)
    
    test_config = Path("examples/configs/test_campaign.yaml")
    
    # Generate configs
    configs, params = generate_parameter_sweep_configs(
        base_config_path=test_config,
        sweep_type="fidelity_focus",
        n_points=3,
        output_dir=Path("test_campaign_output/structure_test"),
    )
    
    print(f"\n[Campaign Info]")
    print(f"  Total configs: {len(configs)}")
    print(f"  Parameters: {list(params.keys())}")
    
    # Show what would be run
    base_config = configs["baseline"]
    n_widths = len(base_config.simulation.widths)
    n_circuits = base_config.simulation.n_circuits
    n_shots = base_config.simulation.n_shots
    
    total_circuits = len(configs) * n_widths * n_circuits
    
    print(f"\n[Simulation Stats]")
    print(f"  Widths per config: {n_widths}")
    print(f"  Circuits per width: {n_circuits}")
    print(f"  Shots per circuit: {n_shots}")
    print(f"  Total circuits to run: {total_circuits}")
    print(f"  Estimated time: ~{total_circuits * 0.1:.1f} seconds")


def print_usage_example():
    """Print example usage of the campaign system."""
    print("\n" + "="*70)
    print("USAGE EXAMPLE")
    print("="*70)
    
    example = """
# Run a full parameter campaign:

python examples/run_parameter_campaign.py \\
    --base-config examples/configs/production.yaml \\
    --sweep-type comprehensive \\
    --n-points 5 \\
    --output campaigns/my_campaign

# Options:
  --sweep-type: comprehensive, fidelity_focus, coherence_focus, timing_focus
  --n-points: Number of values per parameter (default: 5)
  --dry-run: Generate configs without running (test first!)
  --parallel: Enable parallel execution (experimental)

# After completion, results will be in:
  campaigns/my_campaign/
    ├── configs/              # Individual config files
    ├── results/              # Raw results (JSON)
    ├── plots/                # All visualizations
    ├── analysis/             # Sensitivity analysis
    ├── campaign_manifest.json
    ├── campaign_results.json
    └── campaign_report.html  # Main report
"""
    print(example)


def main():
    """Run demo tests."""
    print("\n" + "="*70)
    print("PARAMETER CAMPAIGN SYSTEM - QUICK DEMO")
    print("="*70)
    
    # Test 1: Config generation
    test_config_generation()
    
    # Test 2: Campaign structure
    test_campaign_structure()
    
    # Show usage
    print_usage_example()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review generated test configs in: test_campaign_output/")
    print("  2. Run actual campaign with: python examples/run_parameter_campaign.py --dry-run")
    print("  3. Once satisfied, run without --dry-run to execute")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
