#!/usr/bin/env python3
"""
TEST: Anti-windup fix for error_integral

Hypothesis: Clamping error_integral to [-10, 10] will fix catastrophic failures.

Test: Re-evaluate BC with anti-windup on file 00069.csv
"""

import numpy as np
import torch
import torch.nn as nn

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE

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
    """Original BC (no anti-windup)"""
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        error = target_lataccel - current_lataccel
        self.error_integral += error  # NO CLAMPING
        self.prev_error = error
        return float(action.item())

class BCController_WithAntiwindup:
    """Fixed BC (with anti-windup)"""
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        error = target_lataccel - current_lataccel
        self.error_integral = np.clip(self.error_integral + error, -10, 10)  # ANTI-WINDUP
        self.prev_error = error
        return float(action.item())

print("\n" + "="*60)
print("TESTING ANTI-WINDUP FIX")
print("="*60)

# Load BC network
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

# Load model
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

# Test on failure file
failure_file = "./data/00069.csv"

print(f"\nTesting on failure file: {failure_file}")
print("="*60)

# Without anti-windup
print("\n[1] BC WITHOUT anti-windup...")
controller_old = BCController_NoAntiwindup(network_bc)
sim_old = TinyPhysicsSimulator(model, failure_file, controller=controller_old, debug=False)
sim_old.rollout()
cost_old = sim_old.compute_cost()['total_cost']
print(f"Cost: {cost_old:.1f}")

# With anti-windup
print("\n[2] BC WITH anti-windup...")
controller_new = BCController_WithAntiwindup(network_bc)
sim_new = TinyPhysicsSimulator(model, failure_file, controller=controller_new, debug=False)
sim_new.rollout()
cost_new = sim_new.compute_cost()['total_cost']
print(f"Cost: {cost_new:.1f}")

print("\n" + "="*60)
print("RESULT:")
print("="*60)
print(f"Without anti-windup: {cost_old:.1f}")
print(f"With anti-windup:    {cost_new:.1f}")
print(f"Improvement:         {cost_old - cost_new:.1f} ({100*(cost_old-cost_new)/cost_old:.1f}%)")

if cost_new < cost_old * 0.5:
    print("\n✅ HYPOTHESIS CONFIRMED!")
    print("   Anti-windup fixes catastrophic failure")
elif cost_new < cost_old * 0.9:
    print("\n✅ PARTIAL IMPROVEMENT")
    print("   Anti-windup helps but not complete fix")
else:
    print("\n❌ HYPOTHESIS REJECTED")
    print("   Anti-windup doesn't fix the problem")

print("="*60)

