"""Check which routes are actually being evaluated"""
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

ac = ActorCritic()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac.actor.load_state_dict(bc['model_state_dict'])

class Ctrl:
    def __init__(self):
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
            action = ac.actor(torch.FloatTensor(raw_state / OBS_SCALE)).item()
        
        return float(np.clip(action, -2.0, 2.0))

# Eval uses test_files[:10] - the FIRST 10
print("Evaluating FIRST 10 test routes (what training script uses):")
costs_first10 = []
for i, f in enumerate(test_files[:10]):
    ctrl = Ctrl()
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    cost = sim.rollout()['total_cost']
    costs_first10.append(cost)
    print(f"  Route {i}: {cost:6.2f}")

print(f"\nMean of first 10: {np.mean(costs_first10):.2f}")
print(f"This matches reported eval cost: 74.88? {abs(np.mean(costs_first10) - 74.88) < 1}")

print("\n" + "="*60)
print("Now check 10 RANDOM test routes:")
costs_random10 = []
for i, f in enumerate(random.sample(test_files, 10)):
    ctrl = Ctrl()
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    cost = sim.rollout()['total_cost']
    costs_random10.append(cost)
    print(f"  Route {i}: {cost:6.2f}")

print(f"\nMean of random 10: {np.mean(costs_random10):.2f}")

print("\n" + "="*60)
print(f"First 10 mean: {np.mean(costs_first10):.2f}")
print(f"Random 10 mean: {np.mean(costs_random10):.2f}")
print(f"Difference: {np.mean(costs_random10) - np.mean(costs_first10):.2f}")
print("\nConclusion: First 10 are representative? {abs(np.mean(costs_first10) - np.mean(costs_random10)) < 10}")



