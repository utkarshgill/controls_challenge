#!/usr/bin/env python3
"""
EVALUATE BC QUALITY

Question: On the TRAINING DATA itself (without rollout), how well does BC clone PID?

If BC prediction error is high even on training data → network quality issue
If BC prediction error is low on training data but high on rollout → compounding error issue
"""

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
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

class DataCollector:
    """Collect (state, PID_action) pairs WITHOUT rollout (use ground truth states)"""
    def __init__(self):
        self.pid = pid.Controller()
        self.states = []
        self.actions = []
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build state using PID's internal state
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                               self.pid.prev_error, self.pid.error_integral)
        
        # Get PID action
        pid_action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        
        self.states.append(state_vec)
        self.actions.append(pid_action)
        
        return pid_action

print("\n" + "="*60)
print("BC QUALITY EVALUATION (Non-rollout)")
print("="*60)

# Load BC
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

# Load model
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

# Test on file 00069 (hard file) - collect ground truth (state, action) pairs
print("\nCollecting ground truth PID data from file 00069...")
collector = DataCollector()
sim = TinyPhysicsSimulator(model, "./data/00069.csv", controller=collector, debug=False)
sim.rollout()

states = np.array(collector.states)
pid_actions = np.array(collector.actions)

print(f"  Collected {len(states)} (state, action) pairs")

# Evaluate BC predictions on these states
print("\nEvaluating BC predictions on ground truth states...")
with torch.no_grad():
    states_tensor = torch.from_numpy(states * OBS_SCALE).float()
    action_means, _, _ = network_bc(states_tensor)
    bc_raw_actions = action_means.squeeze().numpy()
    bc_actions = np.tanh(bc_raw_actions) * 2.0  # STEER_RANGE[1] = 2.0

prediction_error = bc_actions - pid_actions
mae = np.mean(np.abs(prediction_error))
mse = np.mean(prediction_error ** 2)
max_error = np.max(np.abs(prediction_error))

print(f"\nBC prediction error (on ground truth states):")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.6f}")
print(f"  Max: {max_error:.4f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"\nFile 00069:")
print(f"  BC prediction MAE (ground truth states): {mae:.4f}")
print(f"  BC action MAE (during rollout):         0.3127")
print(f"  Amplification factor:                    {0.3127/mae:.2f}x")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if mae < 0.05:
    print("\n✅ BC network quality is GOOD")
    print("   Prediction error on ground truth is low")
    print(f"   But rollout error is {0.3127/mae:.1f}× worse!")
    print("\n🎯 ROOT CAUSE: COMPOUNDING ERRORS")
    print("   Small errors at step t → wrong states at t+1 → bigger errors")
    print("\n💡 SOLUTIONS:")
    print("   1. Retrain BC on ROLLOUT data (DAgger)")
    print("   2. Use PPO to learn from actual rollouts")
    print("   3. Add noise/augmentation to training for robustness")
elif mae < 0.15:
    print("\n⚠️  BC network quality is MODERATE")
    print("   Some prediction error even on ground truth")
    print("\n💡 SOLUTIONS:")
    print("   1. Bigger network / more training")
    print("   2. Better hyperparameters")
    print("   3. More training data")
else:
    print("\n❌ BC network quality is POOR")
    print("   High prediction error even on ground truth!")
    print("\n💡 BC training failed - need to debug training process")

print("="*60)

