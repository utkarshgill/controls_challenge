"""
Test hypothesis: Decaying weights should beat single-step FF
Manually set w[i] = 0.3 * exp(-i/10) for i=0..49
"""

import numpy as np
from pathlib import Path

# Generate decaying weights
# Start from w[0]=0.267 (from BC experiment) and decay
weights = np.array([0.267 * np.exp(-i/5) for i in range(50)], dtype=np.float32)

print("="*60)
print("Testing Decay Hypothesis")
print("="*60)
print("Weights pattern: w[i] = 0.267 * exp(-i/5)")
print()
print("Sample weights:")
for i in [0, 1, 2, 5, 10, 20, 30, 40, 49]:
    print(f"  w[{i:2d}] (t={i*0.1:.1f}s): {weights[i]:.6f}")
print()

# Save for controller
output_file = Path(__file__).parent / 'decay_weights.npy'
np.save(output_file, weights)
print(f"Saved to: {output_file}")
print()
print("="*60)
print("NEXT: Test with batch metrics")
print("Expected: < 82.89 if decay hypothesis is correct")
print("="*60)
