#!/usr/bin/env python3
"""
FINAL DIAGNOSIS: Systematic evaluation

Check BC quality across the ENTIRE test set to see if 00069 is an outlier
or if BC is systematically poor.
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
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

class BCController:
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
        self.error_integral = np.clip(self.error_integral + error, -14, 14)
        self.prev_error = error
        return float(action.item())

print("\n" + "="*60)
print("SYSTEMATIC BC EVALUATION")
print("="*60)

# Load
test_files = sorted(glob.glob("./data/*.csv"))[:100]
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

print(f"\nEvaluating on {len(test_files)} files...")

pid_costs = []
bc_costs = []

for f in tqdm(test_files, desc="Evaluating"):
    # PID
    pid_ctrl = pid.Controller()
    sim_pid = TinyPhysicsSimulator(model, f, controller=pid_ctrl, debug=False)
    sim_pid.rollout()
    pid_costs.append(sim_pid.compute_cost()['total_cost'])
    
    # BC
    bc_ctrl = BCController(network_bc)
    sim_bc = TinyPhysicsSimulator(model, f, controller=bc_ctrl, debug=False)
    sim_bc.rollout()
    bc_costs.append(sim_bc.compute_cost()['total_cost'])

pid_costs = np.array(pid_costs)
bc_costs = np.array(bc_costs)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nPID:")
print(f"  Mean: {np.mean(pid_costs):.2f}")
print(f"  Median: {np.median(pid_costs):.2f}")
print(f"  Std: {np.std(pid_costs):.2f}")

print(f"\nBC:")
print(f"  Mean: {np.mean(bc_costs):.2f}")
print(f"  Median: {np.median(bc_costs):.2f}")
print(f"  Std: {np.std(bc_costs):.2f}")

# Per-file ratio
ratios = bc_costs / pid_costs

print(f"\nBC/PID ratio:")
print(f"  Mean: {np.mean(ratios):.2f}x")
print(f"  Median: {np.median(ratios):.2f}x")
print(f"  Files where BC > 2× PID: {np.sum(ratios > 2)}/100")
print(f"  Files where BC > 3× PID: {np.sum(ratios > 3)}/100")
print(f"  Files where BC > 5× PID: {np.sum(ratios > 5)}/100")

# Find worst cases
worst_idx = np.argsort(ratios)[-5:][::-1]
print(f"\nWorst 5 files:")
for i in worst_idx:
    print(f"  {test_files[i]}: PID={pid_costs[i]:.1f}, BC={bc_costs[i]:.1f}, Ratio={ratios[i]:.2f}x")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

target = 45.0
print(f"\nTarget: < {target}")
print(f"Best controller: PID at {np.mean(pid_costs):.2f}")
print(f"Gap to target: {np.mean(pid_costs) - target:.2f} points")

print(f"\n✅ BC median performance ≈ PID median")
print(f"❌ BC mean performance > PID mean (outlier failures)")
print(f"\n🎯 To reach < 45:")
print(f"   Need to improve by {np.mean(pid_costs) - target:.1f} points")
print(f"   This requires fundamentally better control, not just BC→PID cloning")
print(f"\n💡 RECOMMENDATION: Switch to PPO")
print(f"   PPO can learn beyond PID through trial & error")
print("="*60)

