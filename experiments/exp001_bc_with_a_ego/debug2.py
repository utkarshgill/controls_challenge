#!/usr/bin/env python3
"""Check training data stats"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController
from train import build_state

# Collect 10 files to check action distribution
files = sorted(glob.glob("../../data/*.csv"))[:10]

all_actions = []
all_states = []

for file_path in tqdm(files):
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    pid = PIDController()
    sim = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
    
    prev_error = 0.0
    error_integral = 0.0
    
    max_steps = len(sim.data) - 50
    for _ in range(max_steps):
        state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
        current_lataccel = sim.current_lataccel
        
        action = pid.update(target, current_lataccel, state, futureplan)
        
        error = target - current_lataccel
        error_integral = np.clip(error_integral + error, -14, 14)
        state_vec = build_state(target, current_lataccel, state, futureplan, prev_error, error_integral)
        
        all_actions.append(action)
        all_states.append(state_vec)
        
        prev_error = error
        sim.current_steer = action
        sim.step()

actions = np.array(all_actions)
states = np.array(all_states)

print("Action stats:")
print(f"  Mean: {actions.mean():.6f}")
print(f"  Std:  {actions.std():.6f}")
print(f"  Min:  {actions.min():.6f}")
print(f"  Max:  {actions.max():.6f}")
print(f"  90th: {np.percentile(actions, 90):.6f}")
print(f"  99th: {np.percentile(actions, 99):.6f}")

print("\nState[5] (a_ego) stats:")
print(f"  Mean: {states[:, 5].mean():.6f}")
print(f"  Std:  {states[:, 5].std():.6f}")
print(f"  Min:  {states[:, 5].min():.6f}")
print(f"  Max:  {states[:, 5].max():.6f}")

print("\nState[2] (error_integral) stats:")
print(f"  Mean: {states[:, 2].mean():.6f}")
print(f"  Std:  {states[:, 2].std():.6f}")
print(f"  Min:  {states[:, 2].min():.6f}")
print(f"  Max:  {states[:, 2].max():.6f}")

