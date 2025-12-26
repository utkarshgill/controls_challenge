#!/usr/bin/env python3
"""
FINAL EVALUATION: PID vs BC vs PPO

All with anti-windup fix applied.
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import pid

class ActorCritic(nn.Module):
    def __init__(self, state_dim=56, action_dim=1, hidden_dim=128, trunk_layers=1, head_layers=3):
        super(ActorCritic, self).__init__()
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.05))
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
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
        self.error_integral = np.clip(self.error_integral + error, -14, 14)  # ANTI-WINDUP
        self.prev_error = error
        return float(action.item())

print("\n" + "="*60)
print("FINAL EVALUATION: PID vs BC vs PPO")
print("="*60)
print("All controllers have anti-windup (±14) applied")
print("="*60)

# Load models
test_files = sorted(glob.glob("./data/*.csv"))[:100]
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

print("\nLoading networks...")
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()
print("✅ BC loaded")

network_ppo = ActorCritic()
checkpoint = torch.load('ppo_parallel_best.pth', map_location='cpu', weights_only=False)
network_ppo.load_state_dict(checkpoint, strict=False)
network_ppo.eval()
print("✅ PPO loaded")

# Evaluate
results = {'PID': [], 'BC': [], 'PPO': []}

print(f"\nEvaluating on {len(test_files)} files...")

for f in tqdm(test_files, desc="Testing"):
    # PID
    pid_ctrl = pid.Controller()
    sim = TinyPhysicsSimulator(model, f, controller=pid_ctrl, debug=False)
    sim.rollout()
    results['PID'].append(sim.compute_cost()['total_cost'])
    
    # BC
    bc_ctrl = NNController(network_bc)
    sim = TinyPhysicsSimulator(model, f, controller=bc_ctrl, debug=False)
    sim.rollout()
    results['BC'].append(sim.compute_cost()['total_cost'])
    
    # PPO
    ppo_ctrl = NNController(network_ppo)
    sim = TinyPhysicsSimulator(model, f, controller=ppo_ctrl, debug=False)
    sim.rollout()
    results['PPO'].append(sim.compute_cost()['total_cost'])

# Convert to arrays
for k in results:
    results[k] = np.array(results[k])

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n{'Controller':<10} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
print("-" * 60)

for name in ['PID', 'BC', 'PPO']:
    r = results[name]
    print(f"{name:<10} {np.mean(r):<8.2f} {np.median(r):<8.2f} {np.std(r):<8.2f} {np.min(r):<8.2f} {np.max(r):<8.2f}")

print("\n" + "="*60)
print("COMPARISON TO TARGET")
print("="*60)

target = 45.0
best_mean = min(np.mean(results['PID']), np.mean(results['BC']), np.mean(results['PPO']))
best_controller = [k for k in results if np.mean(results[k]) == best_mean][0]

print(f"\nTarget:           < {target:.1f}")
print(f"Best controller:  {best_controller} at {best_mean:.2f}")
print(f"Gap to target:    {best_mean - target:.2f} points")

if best_mean < target:
    print("\n✅✅✅ TARGET ACHIEVED!")
elif best_mean < 60:
    print(f"\n🔥 CLOSE! Only {best_mean - target:.1f} points away")
else:
    print(f"\n⚠️  Need {best_mean - target:.1f} more points of improvement")

print("\n" + "="*60)
print("FAILURE ANALYSIS")
print("="*60)

# Count failures (cost > 2× median)
for name in ['PID', 'BC', 'PPO']:
    r = results[name]
    median = np.median(r)
    failures = np.sum(r > 2 * median)
    catastrophic = np.sum(r > 5 * median)
    print(f"\n{name}:")
    print(f"  Failures (> 2× median):       {failures}/100")
    print(f"  Catastrophic (> 5× median):   {catastrophic}/100")

print("="*60)

# Save results
np.savez('final_results.npz',
         pid_costs=results['PID'],
         bc_costs=results['BC'],
         ppo_costs=results['PPO'],
         test_files=test_files)
print("\n✅ Results saved to final_results.npz")

