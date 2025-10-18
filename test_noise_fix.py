"""
Quick test to verify noise is being applied correctly.

Compare two runs with vastly different F1 values - they should give different results.
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spinq_qv.config.schemas import Config
from spinq_qv.experiments.run_qv import run_experiment
import yaml

# Test 1: High fidelity (F1=0.99926, F2=0.998)
config_high = Config.from_yaml("examples/configs/baseline.yaml")

# Set to only test width=2 for speed
config_high.simulation.widths = [2]
config_high.simulation.n_circuits = 10
config_high.simulation.n_shots = 1000

print("Testing HIGH fidelity: F1=0.99926, F2=0.998")
print(f"  Device params: {config_high.device}")

with tempfile.TemporaryDirectory() as tmpdir:
    result_high = run_experiment(
        config=config_high,
        output_dir=Path(tmpdir),
        seed=12345,
        return_aggregated=True,
        parallel=False
    )

hop_high = result_high[2]['mean_hop']
print(f"  Result: mean_hop = {hop_high:.4f}\n")

# Test 2: Low fidelity (F1=0.95, F2=0.90) - much worse
config_low_dict = yaml.safe_load(open("examples/configs/baseline.yaml"))
config_low_dict['device']['F1'] = 0.95
config_low_dict['device']['F2'] = 0.90

config_low = Config(**config_low_dict)
config_low.simulation.widths = [2]
config_low.simulation.n_circuits = 10
config_low.simulation.n_shots = 1000

print("Testing LOW fidelity: F1=0.95, F2=0.90")
print(f"  Device params: {config_low.device}")

with tempfile.TemporaryDirectory() as tmpdir:
    result_low = run_experiment(
        config=config_low,
        output_dir=Path(tmpdir),
        seed=12345,
        return_aggregated=True,
        parallel=False
    )

hop_low = result_low[2]['mean_hop']
print(f"  Result: mean_hop = {hop_low:.4f}\n")

# Verify they differ significantly
print("=" * 60)
print(f"HIGH fidelity HOP: {hop_high:.4f}")
print(f"LOW  fidelity HOP: {hop_low:.4f}")
print(f"Difference: {hop_high - hop_low:.4f}")

if abs(hop_high - hop_low) < 0.05:
    print("\n⚠️  WARNING: Results are too similar! Noise may not be applied.")
    sys.exit(1)
else:
    print(f"\n✅ SUCCESS: High fidelity gives {hop_high:.4f} vs low fidelity {hop_low:.4f}")
    print("   Noise is being applied correctly!")
    sys.exit(0)
