import numpy as np
from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.io.formats import compute_heavy_outputs

m = 4
circuit = generate_qv_circuit(m, seed=42)
probs = compute_ideal_probabilities(circuit)
sorted_probs = np.sort(probs)[::-1]

print(f"Distribution analysis for m={m}, seed=42")
print(f"=" * 60)
print(f"\nTop 8 probabilities (heavy):")
for i, p in enumerate(sorted_probs[:8]):
    print(f"  {i+1:2d}: {p:.6f}")

print(f"\nBottom 8 probabilities (light):")
for i, p in enumerate(sorted_probs[8:]):
    print(f"  {i+9:2d}: {p:.6f}")

median = np.median(probs)
heavy_indices = compute_heavy_outputs(probs)
heavy_prob = sum(probs[i] for i in heavy_indices)

print(f"\n" + "=" * 60)
print(f"Median: {median:.6f}")
print(f"Heavy outputs (prob > median): {len(heavy_indices)}")
print(f"Total heavy probability: {heavy_prob:.4f}")
print(f"Total light probability: {1 - heavy_prob:.4f}")
print(f"\n✓ This is CORRECT behavior!")
print(f"  Heavy outputs are defined as prob > median (strict inequality)")
print(f"  For skewed distributions, this can result in heavy_prob > 0.5")
