#!/usr/bin/env python3
"""
Problem #2: Why is PID getting 107 instead of 85?
Test PID on different file sets to understand variance.
"""

import numpy as np
import glob
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid

print("\n" + "="*60)
print("PROBLEM #2: PID baseline variance")
print("="*60)

# Load all files
all_files = sorted(glob.glob("./data/*.csv"))
print(f"Total files: {len(all_files)}")

# Test sets
np.random.seed(42)
np.random.shuffle(all_files)

sets = {
    'First 100 files (sorted)': sorted(glob.glob("./data/*.csv"))[:100],
    'Random 100 files': np.random.choice(all_files, 100, replace=False),
    'Validation 100 (seed 42)': all_files[int(len(all_files)*0.9):][:100],
}

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

print("\nTesting PID on different file sets:\n")

for name, files in sets.items():
    costs = []
    for data_file in tqdm(files, desc=name[:20], leave=False):
        controller = pid.Controller()
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        sim.rollout()
        costs.append(sim.compute_cost()['total_cost'])
    
    mean_cost = np.mean(costs)
    median_cost = np.median(costs)
    std_cost = np.std(costs)
    
    print(f"{name:30s}: mean={mean_cost:6.2f}, median={median_cost:6.2f}, std={std_cost:6.2f}")

print("\n" + "="*60)
print("If all sets ~107: Dataset is just harder than expected")
print("If only val set ~107: We picked hard files for validation")
print("="*60)

