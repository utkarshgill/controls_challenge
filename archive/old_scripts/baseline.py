#!/usr/bin/env python3
"""
Problem #2: Establish proper baseline (PID, BC, PPO) on same 100 files
ONE NUMBER for each controller - clean comparison
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import pid

# Architecture (same as before)
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
print("BASELINE: PID vs BC vs PPO (100 files)")
print("="*60)

# Load data (SAME 100 files for fair comparison)
all_files = sorted(glob.glob("./data/*.csv"))
np.random.seed(42)
np.random.shuffle(all_files)
val_files = all_files[int(len(all_files)*0.9):][:100]  # First 100 val files
print(f"Testing on: {len(val_files)} files")

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

results = {}

# Test PID
print("\n[1/3] Testing PID...")
costs = []
for data_file in tqdm(val_files, desc="PID"):
    controller = pid.Controller()
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['PID'] = np.mean(costs)
print(f"  PID: {results['PID']:.2f}")

# Test BC
print("\n[2/3] Testing BC...")
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()
costs = []
for data_file in tqdm(val_files, desc="BC"):
    controller = NNController(network_bc)
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['BC'] = np.mean(costs)
print(f"  BC:  {results['BC']:.2f}")

# Test PPO
print("\n[3/3] Testing PPO...")
network_ppo = ActorCritic()
checkpoint = torch.load('ppo_parallel_best.pth', map_location='cpu', weights_only=False)
network_ppo.load_state_dict(checkpoint, strict=False)
network_ppo.eval()
costs = []
for data_file in tqdm(val_files, desc="PPO"):
    controller = NNController(network_ppo)
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    costs.append(sim.compute_cost()['total_cost'])
results['PPO'] = np.mean(costs)
print(f"  PPO: {results['PPO']:.2f}")

print("\n" + "="*60)
print("BASELINE ESTABLISHED")
print("="*60)
print(f"PID:    {results['PID']:6.2f}")
print(f"BC:     {results['BC']:6.2f}  (should match PID ~85)")
print(f"PPO:    {results['PPO']:6.2f}  (should beat BC)")
print(f"Target: <45.00")
print("="*60)

if results['BC'] > 100:
    print("\n⚠️  BC cost too high! Something wrong with:")
    print("   - Expert data collection")
    print("   - State building")
    print("   - Network loading")
elif results['PPO'] > results['BC'] + 10:
    print("\n⚠️  PPO worse than BC! Problems:")
    print("   - PPO training diverged")
    print("   - Reward function wrong")
    print("   - Gym env broken")
elif abs(results['PPO'] - results['BC']) < 5:
    print("\n⚠️  PPO not improving! Needs:")
    print("   - More exploration")
    print("   - Better learning rate")
    print("   - Longer training")
else:
    print("\n✅ Both models working!")

