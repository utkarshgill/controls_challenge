#!/usr/bin/env python3
"""Evaluate exp037 controller on test set using official interface"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyphysics import run_rollout
import numpy as np

model_path = "../../models/tinyphysics.onnx"
data_files = sorted(Path("../../data").glob("*.csv"))[15000:15020]

print("Evaluating NNFF-style temporal probes (exp037) on 20 routes")
print("=" * 80)

costs = []
for i, file_path in enumerate(data_files):
    cost_dict, _, _ = run_rollout(file_path, 'exp037_temporal', model_path, debug=False)
    costs.append(cost_dict['total_cost'])
    print(f"Route {i+1:2d}: {cost_dict['total_cost']:7.2f}")

print("=" * 80)
print(f"\nExp037 (NNFF-style temporal probes):")
print(f"  Mean: {np.mean(costs):.2f}")
print(f"  Std:  {np.std(costs):.2f}")

print(f"\nPID baseline:")
print(f"  Mean: 101.31")

print(f"\nDifference: {np.mean(costs) - 101.31:+.2f}")

if np.mean(costs) < 75:
    print("✅✅✅ TEMPORAL STRUCTURE HELPS! Cost < 75!")
elif np.mean(costs) < 90:
    print("✅ Temporal probes help significantly")
elif np.mean(costs) < 101:
    print("✅ Temporal probes help slightly")
else:
    print("❌ Temporal probes don't help")

print("\n**THE ANSWER:**")
if np.mean(costs) < 101:
    print("NNFF-style temporal probes beat PID → preview + PPO works with right features!")
else:
    print("Even with NNFF structure, preview from PID doesn't help → need better teacher")

