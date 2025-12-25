#!/usr/bin/env python3
"""Quick test to verify PPO costs"""

from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.ppo_parallel import Controller
import glob
import numpy as np

model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)
controller = Controller()

# Test on first 100 files
files = sorted(glob.glob('./data/*.csv'))[:100]
costs = []

print("Evaluating PPO controller on 100 files...")
for i, f in enumerate(files):
    sim = TinyPhysicsSimulator(model, f, controller=controller, debug=False)
    sim.rollout()
    cost = sim.compute_cost()
    costs.append(cost['total_cost'])
    if (i+1) % 20 == 0:
        print(f"  {i+1}/100 files processed...")

print("\n" + "="*60)
print("PPO Controller Results (100 files)")
print("="*60)
print(f"Mean cost:   {np.mean(costs):.2f}")
print(f"Median cost: {np.median(costs):.2f}")
print(f"Std dev:     {np.std(costs):.2f}")
print(f"Min cost:    {np.min(costs):.2f}")
print(f"Max cost:    {np.max(costs):.2f}")
print("="*60)

if np.mean(costs) < 45:
    print("✅ PASSED: Mean cost < 45 (competition threshold)")
else:
    print(f"❌ FAILED: Mean cost {np.mean(costs):.2f} >= 45")

if np.mean(costs) < 100:
    print("✅ PASSED: Mean cost < 100 (leaderboard threshold)")

