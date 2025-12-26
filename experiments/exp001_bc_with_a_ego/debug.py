#!/usr/bin/env python3
"""Debug what's going wrong"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from train import BCNetwork, build_state, OBS_SCALE
from controllers.pid import Controller as PIDController

# Test on file 00000 (should be easy, but BC gets 45k cost)
model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
pid = PIDController()

# Load BC
bc_model = BCNetwork(state_dim=57)
bc_model.load_state_dict(torch.load("results/checkpoints/bc_with_a_ego.pt"))
bc_model.eval()

# Run first 10 steps
sim = TinyPhysicsSimulator(model, "../../data/00000.csv", controller=None, debug=False)

prev_error = 0.0
error_integral = 0.0

print("Step | PID action | BC action | a_ego | State[5] (a_ego raw) | State[5] (normalized)")
print("-" * 100)

for i in range(10):
    state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
    current_lataccel = sim.current_lataccel
    
    # PID
    pid_action = pid.update(target, current_lataccel, state, futureplan)
    
    # BC
    error = target - current_lataccel
    error_integral = np.clip(error_integral + error, -14, 14)
    state_vec = build_state(target, current_lataccel, state, futureplan, prev_error, error_integral)
    
    # Check what a_ego looks like
    a_ego_raw = state_vec[5]
    state_normalized = state_vec / OBS_SCALE
    a_ego_normalized = state_normalized[5]
    
    state_tensor = torch.from_numpy(state_normalized).float().unsqueeze(0)
    with torch.no_grad():
        bc_action = bc_model(state_tensor).item()
    
    print(f"{i:4d} | {pid_action:10.4f} | {bc_action:10.4f} | {state.a_ego:6.3f} | {a_ego_raw:10.4f} | {a_ego_normalized:10.4f}")
    
    prev_error = error
    sim.current_steer = pid_action
    sim.step()

print("\nOBS_SCALE:", OBS_SCALE[:7])

