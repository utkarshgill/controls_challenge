#!/usr/bin/env python3
"""
Deep diagnosis: Why does BC diverge from PID on file 00069?

Hypothesis: BC actions != PID actions on this file, leading to compounding errors.

Test: Compare BC vs PID action-by-action on 00069.
"""

import numpy as np
import torch
import torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import pid

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        state_dim, hidden_dim = 56, 128
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.ones(1) * np.log(0.05))
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.critic_out = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value

OBS_SCALE = np.array([10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0] * 50, dtype=np.float32)

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
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
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class BCController_NoAntiwindup:
    """BC matching training exactly (no anti-windup)"""
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.actions = []
        self.error_integrals = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        error = target_lataccel - current_lataccel
        action_value = float(action.item())
        
        # Log
        self.actions.append(action_value)
        self.error_integrals.append(self.error_integral)
        
        # Update (NO anti-windup, match training)
        self.error_integral += error
        self.prev_error = error
        return action_value

class PIDController_Instrumented:
    def __init__(self):
        self.pid = pid.Controller()
        self.actions = []
        self.error_integrals = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Log BEFORE action
        self.error_integrals.append(self.pid.error_integral)
        action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        self.actions.append(action)
        return action

print("\n" + "="*60)
print("BC vs PID ACTION COMPARISON")
print("="*60)
print("File: 00069.csv (failure case)")

# Load
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

failure_file = "./data/00069.csv"

# Run BC
print("\n[1] Running BC...")
bc_ctrl = BCController_NoAntiwindup(network_bc)
sim_bc = TinyPhysicsSimulator(model, failure_file, controller=bc_ctrl, debug=False)
sim_bc.rollout()
bc_cost = sim_bc.compute_cost()['total_cost']
print(f"BC cost: {bc_cost:.1f}")

# Run PID
print("\n[2] Running PID...")
pid_ctrl = PIDController_Instrumented()
sim_pid = TinyPhysicsSimulator(model, failure_file, controller=pid_ctrl, debug=False)
sim_pid.rollout()
pid_cost = sim_pid.compute_cost()['total_cost']
print(f"PID cost: {pid_cost:.1f}")

# Analysis
bc_actions = np.array(bc_ctrl.actions)
pid_actions = np.array(pid_ctrl.actions)
bc_integrals = np.array(bc_ctrl.error_integrals)
pid_integrals = np.array(pid_ctrl.error_integrals)

print("\n" + "="*60)
print("ACTION STATISTICS")
print("="*60)

# Action MSE (how different are BC actions from PID?)
action_mse = np.mean((bc_actions - pid_actions)**2)
action_mae = np.mean(np.abs(bc_actions - pid_actions))

print(f"\nAction difference:")
print(f"  MSE: {action_mse:.4f}")
print(f"  MAE: {action_mae:.4f}")
print(f"  Max diff: {np.max(np.abs(bc_actions - pid_actions)):.4f}")

if action_mae > 0.1:
    print("\n⚠️  BC ACTIONS DIVERGE FROM PID!")
    print("   BC is NOT successfully cloning PID on this file")

# Error integral comparison
integral_mse = np.mean((bc_integrals - pid_integrals)**2)
print(f"\nError integral difference:")
print(f"  MSE: {integral_mse:.2f}")
print(f"  BC max: {np.max(np.abs(bc_integrals)):.1f}")
print(f"  PID max: {np.max(np.abs(pid_integrals)):.1f}")

# Find where they start to diverge
action_diff = np.abs(bc_actions - pid_actions)
diverge_step = np.argmax(action_diff > 0.1)  # First step where |diff| > 0.1

if diverge_step > 0:
    print(f"\nDivergence starts at step: {diverge_step}")
    print(f"  BC action: {bc_actions[diverge_step]:.3f}")
    print(f"  PID action: {pid_actions[diverge_step]:.3f}")
    print(f"  BC integral: {bc_integrals[diverge_step]:.1f}")
    print(f"  PID integral: {pid_integrals[diverge_step]:.1f}")

# Check if BC ever sees OOD integral values
print("\n" + "="*60)
print("OUT-OF-DISTRIBUTION CHECK")
print("="*60)
print("Hypothesis: BC sees integral values during rollout that it never saw in training")
print("\nFile 00069 integral range:")
print(f"  BC: [{np.min(bc_integrals):.1f}, {np.max(bc_integrals):.1f}]")
print(f"  PID: [{np.min(pid_integrals):.1f}, {np.max(pid_integrals):.1f}]")
print("\nTypical training data integral range: likely [-20, +20]")
print("(Need to check training data distribution to confirm)")

if np.max(np.abs(bc_integrals)) > 40 or np.max(np.abs(pid_integrals)) > 40:
    print("\n⚠️  INTEGRAL VALUES EXCEED TYPICAL TRAINING RANGE!")
    print("   BC network is out-of-distribution → unpredictable actions")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if action_mae < 0.05:
    print("✅ BC successfully clones PID actions")
    print("   Problem is elsewhere (simulator dynamics? jerk calculation?)")
elif action_mae < 0.2 and diverge_step > 100:
    print("⚠️  BC matches PID initially, then diverges")
    print("   Likely: compounding errors from small action differences")
elif action_mae < 0.2:
    print("⚠️  BC has small but consistent action errors")
    print("   Likely: BC quality issue (need more data or better training)")
else:
    print("❌ BC fails to clone PID on this file")
    print("   Likely: OOD state → network has no idea what to do")

print("="*60)

