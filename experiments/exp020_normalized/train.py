"""
exp020: Proper normalization + all 49 curvatures
State: [error, ei, ed, v_ego, future_curv[0:49]] (53D)
Network: 2 hidden layers (128 units), Tanh
Hypothesis: Proper scaling + full future preview = breakthrough
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# OBS_SCALE from statistics
OBS_SCALE = np.array([
    0.3664,  # error
    7.1769,  # error_integral
    0.1396,  # error_diff
    38.7333,  # v_ego
] + [0.1573] * 49,  # future_curv[0:49]
dtype=np.float32)

# Split routes
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:15000]
val_files = all_files[15000:17500]
test_files = all_files[17500:20000]

print(f"Routes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

def collect_data(files, max_samples=5000):
    X, y = [], []
    for f in files:
        pid = PIDController()
        sim = TinyPhysicsSimulator(model, str(f), controller=pid)
        
        orig = pid.update
        def capture(target, current, state=None, future_plan=None):
            error = target - current
            v_ego = state.v_ego
            
            # Future curvatures (all 49 points)
            future_curvs = []
            for i in range(49):
                if i < len(future_plan.lataccel):
                    lat = future_plan.lataccel[i]
                    curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
                    future_curvs.append(curv)
                else:
                    future_curvs.append(0.0)
            
            raw_state = [error, pid.error_integral + error, error - pid.prev_error, v_ego] + future_curvs
            x = np.array(raw_state, dtype=np.float32) / OBS_SCALE
            
            action = orig(target, current, state, future_plan)
            X.append(x)
            y.append(action)
            return action
        
        pid.update = capture
        sim.rollout()
        if len(X) >= max_samples:
            break
    return torch.FloatTensor(X[:max_samples]), torch.FloatTensor(y[:max_samples]).unsqueeze(1)

# Collect training data
X_train, y_train = collect_data(train_files, max_samples=5000)
print(f"Collected {len(X_train)} training samples, dim={X_train.shape[1]}")

# Network: 53D -> 128 -> 128 -> 1D
net = nn.Sequential(
    nn.Linear(53, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 1)
)

opt = torch.optim.Adam(net.parameters(), lr=0.001)

print("Training...")
for epoch in range(10000):
    pred = net(X_train)
    loss = nn.MSELoss()(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}")

print(f"Final loss: {loss.item():.6f}")

# Save model
torch.save({'model_state_dict': net.state_dict(), 'obs_scale': OBS_SCALE}, 
           'experiments/exp020_normalized/model.pth')
print("✅ Model saved")

# Evaluate
class LearnedController:
    def __init__(self, net, obs_scale):
        self.net = net
        self.obs_scale = obs_scale
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        v_ego = state.v_ego
        
        # Future curvatures
        future_curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
                future_curvs.append(curv)
            else:
                future_curvs.append(0.0)
        
        raw_state = np.array([error, self.error_integral, error_diff, v_ego] + future_curvs, dtype=np.float32)
        x = torch.FloatTensor(raw_state / self.obs_scale).unsqueeze(0)
        
        with torch.no_grad():
            action = self.net(x).item()
        
        self.prev_error = error
        return float(max(-2.0, min(2.0, action)))

def evaluate_cost(files, num_eval=10):
    costs = []
    for f in files[:num_eval]:
        controller = LearnedController(net, OBS_SCALE)
        sim = TinyPhysicsSimulator(model, str(f), controller=controller)
        cost_dict = sim.rollout()
        costs.append(cost_dict['total_cost'])
    return sum(costs) / len(costs)

print("\nEvaluating costs on routes:")
train_cost = evaluate_cost(train_files, num_eval=10)
val_cost = evaluate_cost(val_files, num_eval=10)
test_cost = evaluate_cost(test_files, num_eval=10)

print(f"\nResults:")
print(f"exp017 baseline:        Train=163.72, Val=77.40, Test=75.61")
print(f"exp020 (normalized):    Train={train_cost:.2f}, Val={val_cost:.2f}, Test={test_cost:.2f}")

if test_cost < 75.61:
    improvement = ((75.61 - test_cost) / 75.61) * 100
    print(f"✅ Improvement: {improvement:.1f}%")
else:
    print(f"❌ No improvement (still hitting BC ceiling)")



