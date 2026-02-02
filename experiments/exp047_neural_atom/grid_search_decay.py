"""
Grid search different decay rates to find optimal pattern
"""

import numpy as np
from pathlib import Path
import sys
import subprocess

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Test different decay rates
decay_rates = [1, 2, 5, 10, 20, 50, 100]  # tau in exp(-i/tau)
base_weight = 0.267  # From BC experiment

results = []

for tau in decay_rates:
    # Generate weights
    weights = np.array([base_weight * np.exp(-i/tau) for i in range(50)], dtype=np.float32)
    
    # Save
    np.save(Path(__file__).parent / 'decay_weights.npy', weights)
    
    # Test
    print(f"Testing tau={tau}...")
    print(f"  w[0]={weights[0]:.6f}, w[5]={weights[5]:.6f}, w[10]={weights[10]:.6f}, w[49]={weights[49]:.6f}")
    
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
            print(f"  Cost: {cost:.2f}")
            break
    print()

print("="*60)
print("RESULTS:")
print("="*60)
for tau, cost in sorted(results, key=lambda x: x[1]):
    print(f"tau={tau:3d}: cost={cost:7.2f}")

best_tau, best_cost = min(results, key=lambda x: x[1])
print()
print(f"BEST: tau={best_tau}, cost={best_cost:.2f}")
print(f"Baseline: exp046_v3 = 82.89")

# Save best
weights = np.array([base_weight * np.exp(-i/best_tau) for i in range(50)], dtype=np.float32)
np.save(Path(__file__).parent / 'decay_weights_best.npy', weights)
print(f"\nSaved best to decay_weights_best.npy")
