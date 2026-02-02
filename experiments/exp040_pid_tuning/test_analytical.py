"""
MINIMAL test: Just load cache and solve analytically
"""
import numpy as np
import pickle
from pathlib import Path

print("Loading cache...")
cache_path = Path(__file__).parent / 'pid_demonstrations_1000.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
    states = data['states']
    actions = data['actions']

print(f"Loaded: {states.shape}, {actions.shape}")

print("\nSolving analytically...")
XtX = states.T @ states
Xty = states.T @ actions
weights = np.linalg.solve(XtX, Xty).flatten()

print(f"\n✅ Results:")
print(f"   P: {weights[0]:+.9f}  (PID: +0.195000)")
print(f"   I: {weights[1]:+.9f}  (PID: +0.100000)")
print(f"   D: {weights[2]:+.9f}  (PID: -0.053000)")

total_error = (abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053))
print(f"\n   Total Δ: {total_error:.9f}")

if total_error < 0.001:
    print(f"   ✅ PERFECT! Single neuron WITHOUT bias cloned PID!")
else:
    print(f"   ❌ FAILED")

