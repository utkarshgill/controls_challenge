"""Quick test: one epoch of PPO with BC reg"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

OBS_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333] + [0.1573] * 49, dtype=np.float32)
model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:20000]

# Load BC
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(1) - 5.0)
    def act(self, state, det=False):
        with torch.no_grad():
            mean = self.actor(torch.FloatTensor(state / OBS_SCALE))
            if det: return mean.item()
            return torch.distributions.Normal(mean, self.log_std.exp()).sample().item()

ac = ActorCritic()
bc_ckpt = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac.actor.load_state_dict(bc_ckpt['model_state_dict'])

class Ctrl:
    def __init__(self, ac, stochastic=False):
        self.ac, self.stochastic = ac, stochastic
        self.error_integral, self.prev_error = 0.0, 0.0
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        e_diff = error - self.prev_error
        self.prev_error = error
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([error, self.error_integral, e_diff, state.v_ego] + curvs, dtype=np.float32)
        action = self.ac.act(raw_state, det=not self.stochastic)
        return float(np.clip(action, -2.0, 2.0))

print("Testing 10 routes...")
print(f"σ = {ac.log_std.exp().item():.6f}\n")

# Deterministic
costs_det = []
for f in test_files[:10]:
    ctrl = Ctrl(ac, stochastic=False)
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    cost = sim.rollout()['total_cost']
    costs_det.append(cost)
print(f"Deterministic:  {np.mean(costs_det):.2f} ± {np.std(costs_det):.2f}")

# Stochastic
costs_stoch = []
for f in test_files[:10]:
    ctrl = Ctrl(ac, stochastic=True)
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    cost = sim.rollout()['total_cost']
    costs_stoch.append(cost)
print(f"Stochastic:     {np.mean(costs_stoch):.2f} ± {np.std(costs_stoch):.2f}")
print(f"\nRatio: {np.mean(costs_stoch) / np.mean(costs_det):.2f}x")



