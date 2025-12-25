#!/usr/bin/env python3
"""
Debug: Check if BC state/action collection is synchronized
Run PID and BC side-by-side on same file, compare actions
"""

import numpy as np
import torch
import torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import BaseController, pid

# Load trained BC model (if exists)
state_dim = 55
action_dim = 1
hidden_dim = 128
trunk_layers, head_layers = 1, 3
STEER_RANGE = (-2.0, 2.0)

OBS_SCALE = np.array(
    [10.0, 2.0, 0.03, 20.0, 1000.0] + [1000.0] * 50,
    dtype=np.float32
)

def build_state(target_lataccel, current_lataccel, state, future_plan):
    eps = 1e-6
    error = target_lataccel - current_lataccel
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, current_lataccel, state.v_ego, state.a_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

# Test: Does PID's actual computation match what we feed BC?
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
data_file = "./data/00000.csv"

print("Testing PID action generation vs BC state collection")
print("="*60)

# Method 1: Direct PID call
pid_ctrl = pid.Controller()

# Method 2: Collect via logging controller
states_collected = []
actions_collected = []

class LoggingPIDController(BaseController):
    def __init__(self):
        self.pid = pid.Controller()
        self.calls = []
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build state like BC does
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        
        # Get PID action
        action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        
        # Log
        self.calls.append({
            'state_vec': state_vec,
            'action': action,
            'target': target_lataccel,
            'current': current_lataccel,
            'v_ego': state.v_ego,
            'error': target_lataccel - current_lataccel
        })
        
        return action

logging_ctrl = LoggingPIDController()
sim = TinyPhysicsSimulator(model, data_file, controller=logging_ctrl, debug=False)
cost = sim.rollout()

print(f"PID cost: {cost['total_cost']:.2f}")
print(f"Collected {len(logging_ctrl.calls)} state-action pairs")
print()

# Analyze first 10 calls
print("First 10 state-action pairs:")
print("-" * 60)
for i, call in enumerate(logging_ctrl.calls[:10]):
    state = call['state_vec']
    action = call['action']
    print(f"Step {i}:")
    print(f"  Error: {call['error']:.4f}, V: {call['v_ego']:.2f}")
    print(f"  State[0:5]: {state[:5]}")  # error, lataccel, v, a, curv
    print(f"  Action: {action:.4f}")
    print()

# Check state magnitudes
states = np.array([c['state_vec'] for c in logging_ctrl.calls])
actions = np.array([c['action'] for c in logging_ctrl.calls])

print("State statistics:")
print(f"  Error (dim 0): mean={states[:,0].mean():.4f}, std={states[:,0].std():.4f}")
print(f"  Lataccel (dim 1): mean={states[:,1].mean():.4f}, std={states[:,1].std():.4f}")
print(f"  V_ego (dim 2): mean={states[:,2].mean():.2f}, std={states[:,2].std():.2f}")
print(f"  Future curv mean: mean={states[:,5:].mean():.6f}, std={states[:,5:].std():.6f}")
print()
print("Action statistics:")
print(f"  Mean: {actions.mean():.4f}, Std: {actions.std():.4f}")
print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
print()

# Test if we can fit a simple model to this data
print("Testing if state→action mapping is learnable...")
print("(Simple linear regression as sanity check)")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Normalize states
states_norm = states * OBS_SCALE

# Split train/test
split = int(0.8 * len(states_norm))
X_train, X_test = states_norm[:split], states_norm[split:]
y_train, y_test = actions[:split], actions[split:]

# Fit linear model
model_linear = Ridge(alpha=0.1)
model_linear.fit(X_train, y_train)

y_pred_train = model_linear.predict(X_train)
y_pred_test = model_linear.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Linear model MSE:")
print(f"  Train: {mse_train:.6f}")
print(f"  Test: {mse_test:.6f}")
print()

# Check which state dims are most important
importances = np.abs(model_linear.coef_)
top_5 = np.argsort(importances)[-5:][::-1]
print("Top 5 most important state dimensions:")
dim_names = ['error', 'lataccel', 'v_ego', 'a_ego', 'curv_now'] + [f'fut{i}' for i in range(50)]
for idx in top_5:
    print(f"  {dim_names[idx]}: {importances[idx]:.6f}")

