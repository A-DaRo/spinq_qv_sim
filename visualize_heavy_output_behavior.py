"""
Visualization to demonstrate why heavy output probability > 0.5 is correct.
This creates a clear diagram showing the skewed distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from spinq_qv.circuits import generate_qv_circuit, compute_ideal_probabilities
from spinq_qv.io.formats import compute_heavy_outputs

# Generate example circuit
m = 4
circuit = generate_qv_circuit(m, seed=42)
probs = compute_ideal_probabilities(circuit)

# Compute heavy outputs
median = np.median(probs)
heavy_indices = compute_heavy_outputs(probs)
heavy_prob = sum(probs[i] for i in heavy_indices)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Distribution with median line
indices = np.arange(2**m)
colors = ['#ff4444' if i in heavy_indices else '#4444ff' for i in indices]

ax1.bar(indices, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax1.axhline(median, color='green', linestyle='--', linewidth=3, 
            label=f'Median = {median:.5f}', zorder=10)

# Add annotations
ax1.text(8, 0.25, f'Heavy Outputs\n(prob > median)\n{len(heavy_indices)} states\n{heavy_prob:.1%} total prob', 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='#ff4444', alpha=0.3),
         ha='center')
ax1.text(8, 0.01, f'Light Outputs\n(prob ≤ median)\n{16-len(heavy_indices)} states\n{1-heavy_prob:.1%} total prob', 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='#4444ff', alpha=0.3),
         ha='center')

ax1.set_xlabel('Output State Index', fontsize=14, fontweight='bold')
ax1.set_ylabel('Probability', fontsize=14, fontweight='bold')
ax1.set_title(f'QV Circuit Output Distribution (m={m}, seed=42)\n"Heavy" vs "Light" Split at Median', 
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Cumulative probability
sorted_indices = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_indices]
cumulative = np.cumsum(sorted_probs)

ax2.plot(range(len(cumulative)), cumulative, 'o-', linewidth=3, markersize=8, color='navy')
ax2.axhline(0.5, color='orange', linestyle=':', linewidth=2, label='50% mark')
ax2.axhline(heavy_prob, color='red', linestyle='--', linewidth=2, 
            label=f'Heavy prob = {heavy_prob:.1%}')
ax2.axvline(len(heavy_indices), color='green', linestyle='--', linewidth=2, 
            label=f'Heavy set size = {len(heavy_indices)}')

# Add shaded region
ax2.fill_between(range(len(heavy_indices)), 0, 1, alpha=0.2, color='red', 
                  label='Heavy output region')

ax2.set_xlabel('Number of Top Outputs Included', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
ax2.set_title('Cumulative Probability Distribution\nShowing Heavy Outputs Dominate', 
              fontsize=16, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('heavy_output_explanation.png', dpi=200, bbox_inches='tight')
print(f"✓ Saved visualization to: heavy_output_explanation.png")

print(f"\n" + "="*70)
print(f"KEY INSIGHT:")
print(f"="*70)
print(f"The top {len(heavy_indices)} outputs (heavy set) contain {heavy_prob:.1%} of total probability")
print(f"The bottom {16-len(heavy_indices)} outputs (light set) contain only {1-heavy_prob:.1%}")
print(f"\nThis is CORRECT and EXPECTED for random quantum circuits!")
print(f"Random unitaries create highly non-uniform distributions.")
print(f"="*70)

plt.show()
