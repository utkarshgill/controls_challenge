#!/usr/bin/env python3
"""
Test anti-windup fix on full dataset.

Question: Does anti-windup fix the mean cost problem?
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm

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

class BCController_Fixed:
    """BC with anti-windup"""
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
print("TESTING ANTI-WINDUP ON FULL DATASET")
print("="*60)

# Load data
test_files = sorted(glob.glob("./data/*.csv"))[:100]

# Load BC
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

# Load model
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

# Test BC with anti-windup
print("\nEvaluating BC with anti-windup on 100 files...")
costs = []
for data_file in tqdm(test_files, desc="BC+antiwindup"):
    controller = BCController_Fixed(network_bc)
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])

costs = np.array(costs)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nPID baseline:          80.44 ± 61.71")
print(f"BC (old, no windup):  103.80 ± 250.02")
print(f"BC (new, antiwindup):  {np.mean(costs):.2f} ± {np.std(costs):.2f}")

print(f"\nMedian:")
print(f"  PID:  67.73")
print(f"  BC old: 69.53")
print(f"  BC new: {np.median(costs):.2f}")

improvement_mean = 103.80 - np.mean(costs)
improvement_pct = 100 * improvement_mean / 103.80

print(f"\nImprovement:")
print(f"  Mean: {improvement_mean:.2f} ({improvement_pct:.1f}%)")

if np.mean(costs) < 85:
    print("\n✅ SUCCESS! BC now matches PID baseline (~80)")
elif np.mean(costs) < 95:
    print("\n✅ MAJOR IMPROVEMENT! Getting close to PID")
else:
    print("\n⚠️  PARTIAL FIX - still work to do")

print("="*60)

