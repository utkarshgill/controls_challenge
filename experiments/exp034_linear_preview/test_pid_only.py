#!/usr/bin/env python3
"""Test PID-only to establish baseline"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, run_rollout
from controllers.pid import Controller as PIDController
import numpy as np

model_path = "../../models/tinyphysics.onnx"
data_files = sorted(Path("../../data").glob("*.csv"))[15000:15020]  # First 20 test files

print("Testing PID-only baseline on 20 routes")
print("=" * 80)

costs = []
for i, file_path in enumerate(data_files):
    cost_dict, _, _ = run_rollout(file_path, 'pid', model_path, debug=False)
    costs.append(cost_dict['total_cost'])
    print(f"Route {i+1:2d}: {cost_dict['total_cost']:7.2f}")

print("=" * 80)
print(f"Mean cost: {np.mean(costs):.2f}")
print(f"Std cost:  {np.std(costs):.2f}")
print(f"Min cost:  {np.min(costs):.2f}")
print(f"Max cost:  {np.max(costs):.2f}")

