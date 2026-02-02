#!/usr/bin/env python3
"""Evaluate learned controller on test set"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyphysics import run_rollout
import numpy as np

model_path = "../../models/tinyphysics.onnx"
data_files = sorted(Path("../../data").glob("*.csv"))[15000:15020]

print("Evaluating learned 3-parameter linear controller on 20 routes")
print("=" * 80)

costs = []
for i, file_path in enumerate(data_files):
    cost_dict, _, _ = run_rollout(file_path, 'exp034_linear', model_path, debug=False)
    costs.append(cost_dict['total_cost'])
    print(f"Route {i+1:2d}: {cost_dict['total_cost']:7.2f}")

print("=" * 80)
print(f"\nLearned controller:")
print(f"  Mean: {np.mean(costs):.2f}")
print(f"  Std:  {np.std(costs):.2f}")

print(f"\nPID baseline (from earlier test):")
print(f"  Mean: 101.31")

print(f"\nDifference: {np.mean(costs) - 101.31:+.2f}")
if np.mean(costs) < 101.31:
    print("✅ Learned controller BEATS PID!")
else:
    print("❌ Learned controller WORSE than PID")

