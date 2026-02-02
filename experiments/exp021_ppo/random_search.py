"""
Random search: Proof of concept that better policies exist
Try random small perturbations of BC weights, evaluate
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

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), 
                                nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    
    def forward(self, x):
        return self.net(x)

class Ctrl:
    def __init__(self, actor):
        self.actor = actor
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
            action = self.actor(torch.FloatTensor(raw_state / OBS_SCALE)).item()
        
        return float(np.clip(action, -2.0, 2.0))

def evaluate(actor, files, n=20):
    costs = []
    for f in files[:n]:
        ctrl = Ctrl(actor)
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    return np.mean(costs)

# Load BC
actor_bc = Actor()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
actor_bc.net.load_state_dict(bc['model_state_dict'])

bc_cost = evaluate(actor_bc, test_files)
print(f"BC baseline: {bc_cost:.2f}\n")

# Random search
print("Random search: perturb BC weights, test if better...")
best_cost = bc_cost
best_actor = actor_bc

np.random.seed(42)
for i in range(100):
    # Clone BC
    actor_perturbed = Actor()
    actor_perturbed.load_state_dict(actor_bc.state_dict())
    
    # Add small Gaussian noise to weights
    with torch.no_grad():
        for param in actor_perturbed.parameters():
            noise = torch.randn_like(param) * 0.01  # 1% noise
            param.add_(noise)
    
    # Evaluate
    cost = evaluate(actor_perturbed, test_files, n=10)
    
    if cost < best_cost:
        best_cost = cost
        best_actor = actor_perturbed
        print(f"  Trial {i:3d}: cost={cost:.2f} ✓ NEW BEST (Δ={bc_cost-cost:+.2f})")
    elif i % 10 == 0:
        print(f"  Trial {i:3d}: cost={cost:.2f}")

print(f"\nResults:")
print(f"  BC baseline: {bc_cost:.2f}")
print(f"  Best random: {best_cost:.2f}")
print(f"  Improvement: {bc_cost - best_cost:.2f}")

if best_cost < bc_cost - 0.5:
    print("\n✓ Random perturbations CAN improve! PPO should be able to find this.")
else:
    print("\n✗ Random perturbations don't help. The BC optimum is tight.")

