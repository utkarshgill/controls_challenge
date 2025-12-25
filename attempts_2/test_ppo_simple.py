#!/usr/bin/env python3
"""
Simple sequential test of PPO model (no Gym, no multiprocessing).
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE

# Architecture
state_dim, action_dim = 56, 1
hidden_dim = 128
trunk_layers, head_layers = 1, 3

OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0] * 50,
    dtype=np.float32
)

device = torch.device('cpu')

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
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
    
    @torch.no_grad()
    def act(self, state):
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, _, _ = self(state_tensor)
        raw_action = action_mean
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return float(action.cpu().numpy()[0, 0])

class PPOController:
    def __init__(self, model_path):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers).to(device)
        self.actor_critic.load_state_dict(torch.load(model_path, map_location=device))
        self.actor_critic.eval()
        
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        action = self.actor_critic.act(obs)
        
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.prev_error = error
        
        return action

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing PPO Model (Sequential)")
    print("="*60)
    
    # Load files
    all_files = sorted(glob.glob("./data/*.csv"))
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    val_files = all_files[split_idx:]
    
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    
    # Test PPO
    print(f"\nTesting ppo_parallel_best.pth on {len(val_files[:100])} files...")
    costs = []
    for data_file in tqdm(val_files[:100]):
        controller = PPOController('ppo_parallel_best.pth')
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()['total_cost']
        costs.append(cost)
    
    print("\n" + "="*60)
    print(f"PPO: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
    print(f"Min: {np.min(costs):.2f}, Max: {np.max(costs):.2f}")
    print(f"Median: {np.median(costs):.2f}")
    print("="*60)
    
    # Show breakdown by percentile
    print("\nCost distribution:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th percentile: {np.percentile(costs, p):.2f}")

