"""
Debug: Check the actual noise magnitude during training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Load checkpoint
bc_path = Path('./experiments/exp020_normalized/model.pth')
bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)

OBS_SCALE = bc_checkpoint['obs_scale']

# Create network with low noise
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(53, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1) - 4.0)  # σ = 0.018
        
    def forward(self, state):
        return self.actor(state), self.log_std.exp()

actor_critic = ActorCritic()
actor_critic.actor.load_state_dict(bc_checkpoint['model_state_dict'])

print(f"Exploration noise σ: {actor_critic.log_std.exp().item():.6f}")

# Test on one route
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_file = all_files[17500]

model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

class TestController:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.actions_det = []
        self.actions_stoch = []
        
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        v_ego = state.v_ego
        
        future_curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
                future_curvs.append(curv)
            else:
                future_curvs.append(0.0)
        
        raw_state = np.array([error, self.error_integral, error_diff, v_ego] + future_curvs, dtype=np.float32)
        x = torch.FloatTensor(raw_state / OBS_SCALE).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_std = actor_critic(x)
            action_det = action_mean.item()
            action_stoch = torch.distributions.Normal(action_mean, action_std).sample().item()
        
        self.actions_det.append(action_det)
        self.actions_stoch.append(action_stoch)
        
        self.prev_error = error
        
        if self.stochastic:
            return float(np.clip(action_stoch, -2.0, 2.0))
        else:
            return float(np.clip(action_det, -2.0, 2.0))

# Test deterministic
print("\nTesting deterministic (no noise)...")
ctrl_det = TestController(stochastic=False)
sim_det = TinyPhysicsSimulator(model, str(test_file), controller=ctrl_det)
cost_det = sim_det.rollout()
print(f"Deterministic cost: {cost_det['total_cost']:.2f}")

# Test stochastic
print("\nTesting stochastic (with σ=0.018 noise)...")
ctrl_stoch = TestController(stochastic=True)
sim_stoch = TinyPhysicsSimulator(model, str(test_file), controller=ctrl_stoch)
cost_stoch = sim_stoch.rollout()
print(f"Stochastic cost: {cost_stoch['total_cost']:.2f}")

# Analyze action differences
action_diffs = np.array(ctrl_stoch.actions_stoch) - np.array(ctrl_det.actions_det)
print(f"\nAction noise stats:")
print(f"  Mean absolute diff: {np.mean(np.abs(action_diffs)):.6f}")
print(f"  Std of diff: {np.std(action_diffs):.6f}")
print(f"  Max absolute diff: {np.max(np.abs(action_diffs)):.6f}")
print(f"\nCost ratio: {cost_stoch['total_cost'] / cost_det['total_cost']:.1f}x")



