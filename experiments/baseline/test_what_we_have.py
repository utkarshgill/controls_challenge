#!/usr/bin/env python3
"""
Problem #1: Test what we actually have right now.
Simple, fast, with clear output.
"""

import numpy as np
import torch
import torch.nn as nn
import glob

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE

# Copy the minimal architecture we need
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        state_dim, hidden_dim = 56, 128
        
        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU()
        )
        
        # Actor
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.ones(1) * np.log(0.05))
        
        # Critic (unused for eval)
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

class TestController:
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        
        # Deterministic action
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.prev_error = error
        
        return float(action.item())

print("\n" + "="*60)
print("PROBLEM #1: What do we have?")
print("="*60)

# Check files exist
import os
files_exist = {
    'bc_pid_checkpoint.pth': os.path.exists('bc_pid_checkpoint.pth'),
    'ppo_parallel_best.pth': os.path.exists('ppo_parallel_best.pth'),
}

print("\nFiles:")
for name, exists in files_exist.items():
    status = "✅" if exists else "❌"
    print(f"  {status} {name}")

if not any(files_exist.values()):
    print("\n❌ No model files found! Need to train first.")
    exit(1)

# Load data
all_files = sorted(glob.glob("./data/*.csv"))
np.random.seed(42)
np.random.shuffle(all_files)
val_files = all_files[int(len(all_files)*0.9):]

print(f"\nValidation files: {len(val_files)}")

# Test each model on 10 files (fast!)
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

for model_name in ['bc_pid_checkpoint.pth', 'ppo_parallel_best.pth']:
    if not files_exist[model_name]:
        continue
    
    print(f"\nTesting {model_name}...")
    network = ActorCritic()
    
    # Load weights
    checkpoint = torch.load(model_name, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load with strict=False (BC doesn't have critic weights)
    network.load_state_dict(state_dict, strict=False)
    network.eval()
    
    # Test on 10 files
    costs = []
    for i, data_file in enumerate(val_files[:10]):
        controller = TestController(network)
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()['total_cost']
        costs.append(cost)
        print(f"  File {i+1}/10: cost = {cost:.1f}")
    
    mean_cost = np.mean(costs)
    print(f"  → Mean: {mean_cost:.1f}")

print("\n" + "="*60)
print("DONE")
print("="*60)

