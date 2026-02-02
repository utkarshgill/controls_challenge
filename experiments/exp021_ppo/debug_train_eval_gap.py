"""
Debug: Why is train cost 3x worse than eval cost?
Measure the ACTUAL cost during training rollouts
"""
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

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(1) - 5.0)

ac = ActorCritic()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac.actor.load_state_dict(bc['model_state_dict'])

print(f"σ = {ac.log_std.exp().item():.6f}\n")

# Test same route 10 times
test_route = test_files[0]

class Ctrl:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.ei, self.pe = 0.0, 0.0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        
        with torch.no_grad():
            mean = ac.actor(torch.FloatTensor(raw_state / OBS_SCALE))
            if self.stochastic:
                action = torch.distributions.Normal(mean, ac.log_std.exp()).sample().item()
            else:
                action = mean.item()
        
        return float(np.clip(action, -2.0, 2.0))

# 10 deterministic runs (should be identical)
print("Deterministic (10 runs):")
det_costs = []
for _ in range(10):
    ctrl = Ctrl(stochastic=False)
    sim = TinyPhysicsSimulator(model_onnx, str(test_route), controller=ctrl)
    cost = sim.rollout()['total_cost']
    det_costs.append(cost)
print(f"  Mean: {np.mean(det_costs):.2f}")
print(f"  Std:  {np.std(det_costs):.2f}")
print(f"  Range: [{min(det_costs):.2f}, {max(det_costs):.2f}]")

# 10 stochastic runs
print("\nStochastic (10 runs with σ=0.0067):")
stoch_costs = []
for _ in range(10):
    ctrl = Ctrl(stochastic=True)
    sim = TinyPhysicsSimulator(model_onnx, str(test_route), controller=ctrl)
    cost = sim.rollout()['total_cost']
    stoch_costs.append(cost)
print(f"  Mean: {np.mean(stoch_costs):.2f}")
print(f"  Std:  {np.std(stoch_costs):.2f}")
print(f"  Range: [{min(stoch_costs):.2f}, {max(stoch_costs):.2f}]")

print(f"\nNoise penalty: {np.mean(stoch_costs) / np.mean(det_costs):.2f}x")

# Now test on 10 different routes
print("\n" + "="*60)
print("Testing diversity across 10 random routes:")
print("="*60)

routes = random.sample(test_files, 10)
det_costs_multi = []
stoch_costs_multi = []

for route in routes:
    ctrl = Ctrl(stochastic=False)
    sim = TinyPhysicsSimulator(model_onnx, str(route), controller=ctrl)
    det_costs_multi.append(sim.rollout()['total_cost'])
    
    ctrl = Ctrl(stochastic=True)
    sim = TinyPhysicsSimulator(model_onnx, str(route), controller=ctrl)
    stoch_costs_multi.append(sim.rollout()['total_cost'])

print(f"\nDeterministic across routes:")
print(f"  Mean: {np.mean(det_costs_multi):.2f}")
print(f"  Std:  {np.std(det_costs_multi):.2f}")

print(f"\nStochastic across routes:")
print(f"  Mean: {np.mean(stoch_costs_multi):.2f}")
print(f"  Std:  {np.std(stoch_costs_multi):.2f}")

print(f"\nWorst stochastic route: {max(stoch_costs_multi):.2f}")
print(f"Worst deterministic route: {max(det_costs_multi):.2f}")

# Check: are some routes just hard?
print("\nPer-route comparison:")
for i, (d, s) in enumerate(zip(det_costs_multi, stoch_costs_multi)):
    print(f"  Route {i}: det={d:6.2f}  stoch={s:6.2f}  ratio={s/d:.2f}x")



