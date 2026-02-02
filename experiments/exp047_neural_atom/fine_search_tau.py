"""
Fine-grained search around tau=1 to find absolute optimum
Test tau in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
"""

import numpy as np
from pathlib import Path
import sys
import subprocess

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Fine search around tau=1
tau_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
base_weight = 0.267

results = []

for tau in tau_values:
    # Generate weights
    weights = np.array([base_weight * np.exp(-i/tau) for i in range(50)], dtype=np.float32)
    
    # Save
    np.save(Path(__file__).parent / 'decay_weights.npy', weights)
    
    # Show weight pattern
    print(f"tau={tau:.1f}: w[0]={weights[0]:.6f}, w[1]={weights[1]:.6f}, w[2]={weights[2]:.6f}, w[3]={weights[3]:.6f}")
    
    result = subprocess.run(
        [
            'python', 'tinyphysics.py',
            '--model_path', './models/tinyphysics.onnx',
            '--data_path', './data',
            '--num_segs', '100',
            '--controller', 'exp047_decay'
        ],
        cwd=Path(__file__).parent.parent.parent,
        capture_output=True,
        text=True
    )
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'average total_cost' in line:
            cost = float(line.split('average total_cost:')[1].strip())
            results.append((tau, cost))
            print(f"  → Cost: {cost:.2f}\n")
            break

print("="*60)
print("RESULTS (sorted by cost):")
print("="*60)
for tau, cost in sorted(results, key=lambda x: x[1]):
    improvement = 82.89 - cost
    print(f"tau={tau:4.1f}: cost={cost:7.2f}  (Δ={improvement:+5.2f} vs exp046_v3)")

best_tau, best_cost = min(results, key=lambda x: x[1])
print()
print(f"🎯 OPTIMAL: tau={best_tau:.1f}, cost={best_cost:.2f}")
print(f"   Improvement: {82.89 - best_cost:.2f} better than exp046_v3")
print()

# Show optimal weights
weights = np.array([base_weight * np.exp(-i/best_tau) for i in range(50)], dtype=np.float32)
print("Optimal weight pattern:")
for i in [0, 1, 2, 3, 5, 10]:
    print(f"  w[{i}] (t={i*0.1:.1f}s): {weights[i]:.6f}")

# Save
np.save(Path(__file__).parent / 'decay_weights_optimal.npy', weights)
print(f"\nSaved to decay_weights_optimal.npy")
