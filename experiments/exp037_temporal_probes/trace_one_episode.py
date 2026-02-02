#!/usr/bin/env python3
"""Trace exactly what happens in one evaluation episode"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Copy the environment class locally and add logging
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController
import random

# Load checkpoint
checkpoint = torch.load('ppo_best.pth', map_location='cpu', weights_only=False)
weights = checkpoint['model_state_dict']['actor.weight'].numpy()[0]
bias = checkpoint['model_state_dict']['actor.bias'].numpy()[0]

model_path = "../../models/tinyphysics.onnx"
model = TinyPhysicsModel(model_path, debug=False)
data_file = "../../data/15000.csv"  # First test file

# Create simulator with PID
from controllers.pid import Controller as PIDController
pid = PIDController()
sim = TinyPhysicsSimulator(model, data_file, controller=pid, debug=False)

print("Running PID-only on route 15000...")
cost_pid = sim.rollout()
print(f"PID cost: {cost_dict['total_cost']:.2f}")
print()

# Now let's manually simulate with residual
print("Now simulating with residual...")
print(f"Weights: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}], bias: {bias:.4f}")
print()

# Reset and run first 110 steps with logging
sim.reset()
pid.error_integral = 0.0
pid.prev_error = 0.0
filtered_residual = 0.0

for step_i in range(110):
    if step_i >= len(sim.data):
        break
        
    # Get features (like in train_ppo.py _get_state)
    future_lat = []
    for i in range(50):
        future_idx = sim.step_idx + i
        if future_idx < len(sim.data):
            future_lat.append(sim.data['target_lataccel'].values[future_idx])
        else:
            future_lat.append(future_lat[-1] if future_lat else 0.0)
    
    baseline = future_lat[0]
    delta = np.array([lat - baseline for lat in future_lat])
    f1 = np.mean(delta[1:6]) / 3.0
    f2 = np.mean(delta[10:25]) / 3.0
    f3 = np.mean(delta) / 3.0
    
    # Compute residual
    raw_residual = weights[0] * f1 + weights[1] * f2 + weights[2] * f3 + bias
    residual = np.tanh(raw_residual) * 0.5  # RESIDUAL_CLIP
    filtered_residual = 0.05 * residual + 0.95 * filtered_residual  # LOWPASS_ALPHA=0.05
    scaled_residual = filtered_residual * 0.1  # RESIDUAL_SCALE
    
    # Compute PID
    target = sim.target_lataccel_history[sim.step_idx]
    current = sim.current_lataccel_history[sim.step_idx]
    error = target - current
    pid.error_integral += error
    error_diff = error - pid.prev_error
    pid.prev_error = error
    pid_action = pid.p * error + pid.i * pid.error_integral + pid.d * error_diff
    pid_action = np.clip(pid_action, -2.0, 2.0)
    
    # Combined
    combined = np.clip(pid_action + scaled_residual, -2.0, 2.0)
    
    if step_i % 10 == 0 or step_i >= 100:
        print(f"Step {sim.step_idx:3d}: f=[{f1:6.3f},{f2:6.3f},{f3:6.3f}] raw_res={raw_residual:6.3f} filtered={filtered_residual:6.3f} scaled={scaled_residual:6.3f} pid={pid_action:6.3f} combined={combined:6.3f}")
    
    # Set action and step
    pid.update(target, current, None, None)  # Just to match interface
    sim.step()

print()
cost_dict = sim.compute_cost()
print(f"Final cost with residual: {cost_dict['total_cost']:.2f}")

