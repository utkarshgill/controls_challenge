#!/usr/bin/env python3
"""
BACK TO BASICS: Feynman's approach

Question 1: Does BC actually clone PID on EASY files?
Question 2: What makes file 00069 HARD?
Question 3: Is there a state construction bug?

Method: Compare BC vs PID action-by-action on EASY and HARD files
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

class InstrumentedBC:
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.actions = []
        self.states = []
        self.integrals = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan, self.prev_error, self.error_integral)
        self.states.append(obs.copy())
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        error = target_lataccel - current_lataccel
        action_value = float(action.item())
        self.actions.append(action_value)
        self.integrals.append(self.error_integral)
        self.error_integral = np.clip(self.error_integral + error, -14, 14)
        self.prev_error = error
        return action_value

class InstrumentedPID:
    def __init__(self):
        self.pid = pid.Controller()
        self.actions = []
        self.integrals = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.integrals.append(self.pid.error_integral)
        action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        self.actions.append(action)
        return action

print("\n" + "="*80)
print("BACK TO BASICS: Feynman's Approach")
print("="*80)

# Load
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

# Test on EASY file (first file, should be easy)
print("\n[1] EASY FILE: ./data/00000.csv")
print("-" * 80)

bc_ctrl = InstrumentedBC(network_bc)
sim_bc = TinyPhysicsSimulator(model, "./data/00000.csv", controller=bc_ctrl, debug=False)
sim_bc.rollout()
bc_cost_easy = sim_bc.compute_cost()['total_cost']

pid_ctrl = InstrumentedPID()
sim_pid = TinyPhysicsSimulator(model, "./data/00000.csv", controller=pid_ctrl, debug=False)
sim_pid.rollout()
pid_cost_easy = sim_pid.compute_cost()['total_cost']

bc_actions_easy = np.array(bc_ctrl.actions)
pid_actions_easy = np.array(pid_ctrl.actions)
action_mae_easy = np.mean(np.abs(bc_actions_easy - pid_actions_easy))

print(f"PID cost:  {pid_cost_easy:.1f}")
print(f"BC cost:   {bc_cost_easy:.1f}")
print(f"Ratio:     {bc_cost_easy/pid_cost_easy:.2f}x")
print(f"Action MAE: {action_mae_easy:.4f}")

if action_mae_easy < 0.05:
    print("✅ BC successfully clones PID on easy file")
else:
    print("❌ BC FAILS to clone PID even on easy file!")

# Test on HARD file (00069)
print("\n[2] HARD FILE: ./data/00069.csv")
print("-" * 80)

bc_ctrl = InstrumentedBC(network_bc)
sim_bc = TinyPhysicsSimulator(model, "./data/00069.csv", controller=bc_ctrl, debug=False)
sim_bc.rollout()
bc_cost_hard = sim_bc.compute_cost()['total_cost']

pid_ctrl = InstrumentedPID()
sim_pid = TinyPhysicsSimulator(model, "./data/00069.csv", controller=pid_ctrl, debug=False)
sim_pid.rollout()
pid_cost_hard = sim_pid.compute_cost()['total_cost']

bc_actions_hard = np.array(bc_ctrl.actions)
pid_actions_hard = np.array(pid_ctrl.actions)
bc_integrals_hard = np.array(bc_ctrl.integrals)
pid_integrals_hard = np.array(pid_ctrl.integrals)
action_mae_hard = np.mean(np.abs(bc_actions_hard - pid_actions_hard))

print(f"PID cost:  {pid_cost_hard:.1f}")
print(f"BC cost:   {bc_cost_hard:.1f}")
print(f"Ratio:     {bc_cost_hard/pid_cost_hard:.2f}x")
print(f"Action MAE: {action_mae_hard:.4f}")

print(f"\nIntegral stats:")
print(f"  PID: mean={np.mean(pid_integrals_hard):.2f}, max={np.max(np.abs(pid_integrals_hard)):.2f}")
print(f"  BC:  mean={np.mean(bc_integrals_hard):.2f}, max={np.max(np.abs(bc_integrals_hard)):.2f}")

# CRITICAL QUESTION: If BC actions ≈ PID actions, why is cost so different?
print("\n" + "="*80)
print("CRITICAL INSIGHT")
print("="*80)

if action_mae_easy < 0.1 and action_mae_hard < 0.3:
    print("\n✅ BC actions are CLOSE to PID actions on both files")
    print("   But cost is still 3.7× worse on hard file!")
    print("\n🤔 This suggests:")
    print("   1. Small action errors compound exponentially in simulator")
    print("   2. BC might be producing high-frequency oscillations (jerk penalty)")
    print("   3. There's a subtle bug in state construction or simulator interaction")
    
    # Check for oscillations
    action_diff_bc = np.diff(bc_actions_hard)
    action_diff_pid = np.diff(pid_actions_hard)
    print(f"\n   Action smoothness (std of action changes):")
    print(f"     PID: {np.std(action_diff_pid):.4f}")
    print(f"     BC:  {np.std(action_diff_bc):.4f}")
    
    if np.std(action_diff_bc) > 1.5 * np.std(action_diff_pid):
        print("   ⚠️  BC oscillates more → higher jerk cost!")
else:
    print(f"\n❌ BC actions DIVERGE from PID:")
    print(f"   Easy file MAE: {action_mae_easy:.4f}")
    print(f"   Hard file MAE: {action_mae_hard:.4f}")
    print("\n🤔 BC is NOT successfully cloning PID!")
    print("   Possible causes:")
    print("   1. Training data doesn't cover hard scenarios")
    print("   2. Network capacity too small")
    print("   3. State construction bug")

# What makes file 00069 hard?
print("\n" + "="*80)
print("WHAT MAKES FILE 00069 HARD?")
print("="*80)

# Load both files and compare
import pandas as pd
easy_data = pd.read_csv("./data/00000.csv")
hard_data = pd.read_csv("./data/00069.csv")

print(f"\nFile statistics:")
print(f"  Easy (00000): {len(easy_data)} steps, PID cost = {pid_cost_easy:.1f}")
print(f"  Hard (00069): {len(hard_data)} steps, PID cost = {pid_cost_hard:.1f}")

# Check for extreme values
print(f"\nTarget lataccel range:")
print(f"  Easy: [{easy_data['targetLateralAcceleration'].min():.2f}, {easy_data['targetLateralAcceleration'].max():.2f}]")
print(f"  Hard: [{hard_data['targetLateralAcceleration'].min():.2f}, {hard_data['targetLateralAcceleration'].max():.2f}]")

print(f"\nVelocity range:")
print(f"  Easy: [{easy_data['vEgo'].min():.2f}, {easy_data['vEgo'].max():.2f}]")
print(f"  Hard: [{hard_data['vEgo'].min():.2f}, {hard_data['vEgo'].max():.2f}]")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. If BC clones PID well (MAE < 0.1) but cost is worse:")
print("   → Problem is COMPOUNDING ERRORS or JERK")
print("   → Need better BC training (more epochs, regularization)")
print("\n2. If BC fails to clone PID (MAE > 0.3):")
print("   → Problem is FUNDAMENTAL (network, training data, state bug)")
print("   → Need to debug BC training itself")
print("\n3. If file 00069 has extreme values:")
print("   → BC wasn't trained on such scenarios")
print("   → Need more diverse training data or data augmentation")
print("="*80)

