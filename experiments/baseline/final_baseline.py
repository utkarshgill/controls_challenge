#!/usr/bin/env python3
"""
FINAL BASELINE RUN
One definitive evaluation on 100 EASY files (first 100 sorted).
Get numbers we can trust.
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import pid

# Architecture
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

class NNController:
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
        self.error_integral += error
        self.prev_error = error
        return float(action.item())

print("\n" + "="*60)
print("FINAL BASELINE RUN")
print("="*60)
print("Test set: First 100 files (sorted, easier)")
print("Controllers: PID, BC, PPO")
print("="*60)

# Use FIRST 100 sorted files (easier than last 10%)
test_files = sorted(glob.glob("./data/*.csv"))[:100]
print(f"\nTest files: {len(test_files)}")

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

results = {}

# 1. PID
print("\n[1/3] PID...")
costs = []
for data_file in tqdm(test_files, desc="PID", leave=False):
    controller = pid.Controller()
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['PID'] = {
    'costs': costs,
    'mean': np.mean(costs),
    'median': np.median(costs),
    'std': np.std(costs),
    'min': np.min(costs),
    'max': np.max(costs),
}
print(f"PID: {results['PID']['mean']:.2f} ± {results['PID']['std']:.2f}")

# 2. BC
print("\n[2/3] BC...")
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()
costs = []
for data_file in tqdm(test_files, desc="BC", leave=False):
    controller = NNController(network_bc)
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['BC'] = {
    'costs': costs,
    'mean': np.mean(costs),
    'median': np.median(costs),
    'std': np.std(costs),
    'min': np.min(costs),
    'max': np.max(costs),
}
print(f"BC:  {results['BC']['mean']:.2f} ± {results['BC']['std']:.2f}")

# 3. PPO
print("\n[3/3] PPO...")
network_ppo = ActorCritic()
checkpoint = torch.load('ppo_parallel_best.pth', map_location='cpu', weights_only=False)
network_ppo.load_state_dict(checkpoint, strict=False)
network_ppo.eval()
costs = []
for data_file in tqdm(test_files, desc="PPO", leave=False):
    controller = NNController(network_ppo)
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['PPO'] = {
    'costs': costs,
    'mean': np.mean(costs),
    'median': np.median(costs),
    'std': np.std(costs),
    'min': np.min(costs),
    'max': np.max(costs),
}
print(f"PPO: {results['PPO']['mean']:.2f} ± {results['PPO']['std']:.2f}")

# FINAL REPORT
print("\n" + "="*60)
print("FINAL BASELINE RESULTS (100 easy files)")
print("="*60)

for name in ['PID', 'BC', 'PPO']:
    r = results[name]
    print(f"\n{name}:")
    print(f"  Mean:   {r['mean']:6.2f} ± {r['std']:.2f}")
    print(f"  Median: {r['median']:6.2f}")
    print(f"  Range:  {r['min']:.2f} - {r['max']:.2f}")

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
bc_vs_pid = results['BC']['mean'] - results['PID']['mean']
ppo_vs_pid = results['PPO']['mean'] - results['PID']['mean']
ppo_vs_bc = results['PPO']['mean'] - results['BC']['mean']

print(f"BC  vs PID: {bc_vs_pid:+.2f}  {'✅ matching' if abs(bc_vs_pid) < 5 else '❌ diverged'}")
print(f"PPO vs PID: {ppo_vs_pid:+.2f}  {'✅ better' if ppo_vs_pid < -5 else '⚠️ not improving' if abs(ppo_vs_pid) < 5 else '❌ worse'}")
print(f"PPO vs BC:  {ppo_vs_bc:+.2f}  {'✅ improving' if ppo_vs_bc < -2 else '⚠️ similar' if abs(ppo_vs_bc) < 2 else '❌ degraded'}")

print("\n" + "="*60)
print("TARGET: < 45.00")
print(f"BEST:   {min(results['PID']['mean'], results['BC']['mean'], results['PPO']['mean']):.2f}")
print(f"GAP:    {min(results['PID']['mean'], results['BC']['mean'], results['PPO']['mean']) - 45:.2f}")
print("="*60)

# Save results
np.savez('baseline_results.npz',
         pid_costs=results['PID']['costs'],
         bc_costs=results['BC']['costs'],
         ppo_costs=results['PPO']['costs'],
         test_files=test_files)
print("\n✅ Saved: baseline_results.npz")

