#!/usr/bin/env python3
"""
TEST: Curvature calculation fix for low speeds

Hypothesis: When v_ego ≈ 0, curvature → ±∞, breaking BC
Fix: Set curvature = 0 when v_ego < 1.0 m/s
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

def build_state_FIXED(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """Fixed version with low-speed protection"""
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    
    # LOW-SPEED PROTECTION: curvature is meaningless below 1 m/s
    if state.v_ego < 1.0:
        curv_now = 0.0
        future_curvs = [0.0] * 50
    else:
        eps = 1e-6
        curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
        
        future_curvs = []
        for t in range(min(50, len(future_plan.lataccel))):
            lat = future_plan.lataccel[t]
            roll = future_plan.roll_lataccel[t]
            v = future_plan.v_ego[t]
            
            # Protect each future step too
            if v < 1.0:
                future_curvs.append(0.0)
            else:
                curv = (lat - roll) / (v ** 2 + eps)
                future_curvs.append(curv)
        
        while len(future_curvs) < 50:
            future_curvs.append(0.0)
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class BCController_Fixed:
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state_FIXED(target_lataccel, current_lataccel, state, future_plan,
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
print("TESTING CURVATURE FIX")
print("="*60)

# Load
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

# Test on easy file (should stay ~same)
print("\n[1] Easy file (00000): v_ego ∈ [33, 34] m/s")
controller = BCController_Fixed(network_bc)
sim = TinyPhysicsSimulator(model, "./data/00000.csv", controller=controller, debug=False)
sim.rollout()
cost_easy = sim.compute_cost()['total_cost']
print(f"  BC (old):  85.9")
print(f"  BC (fixed): {cost_easy:.1f}")
print(f"  PID:        84.4")

# Test on hard file (should improve dramatically)
print("\n[2] Hard file (00069): v_ego ∈ [0, 5] m/s")
controller = BCController_Fixed(network_bc)
sim = TinyPhysicsSimulator(model, "./data/00069.csv", controller=controller, debug=False)
sim.rollout()
cost_hard = sim.compute_cost()['total_cost']
print(f"  BC (old):   1401")
print(f"  BC (fixed): {cost_hard:.1f}")
print(f"  PID:        375")

print("\n" + "="*60)
print("RESULT")
print("="*60)

if cost_hard < 500:
    print("\n✅✅✅ SUCCESS!")
    print(f"   File 00069: {cost_hard:.1f} (was 1401)")
    print("   Low-speed curvature bug FIXED!")
elif cost_hard < 1000:
    print("\n✅ MAJOR IMPROVEMENT!")
    print(f"   File 00069: {cost_hard:.1f} (was 1401)")
    print("   But still not matching PID (375)")
else:
    print("\n❌ No improvement")
    print("   Curvature was not the main issue")

print("="*60)

