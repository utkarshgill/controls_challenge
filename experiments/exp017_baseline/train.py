"""
Baseline: 1 neuron learns PID perfectly
Proof that MLP can learn control from demonstrations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX

# Split routes: train/val/test
import random
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:15000]
val_files = all_files[15000:17500]
test_files = all_files[17500:20000]

print(f"Routes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

def collect_data(files, max_samples=1000):
    X, y = [], []
    for f in files:
        pid = PIDController()
        sim = TinyPhysicsSimulator(model, str(f), controller=pid)
        
        orig = pid.update
        def capture(target, current, state=None, future_plan=None):
            error = target - current
            x = [error, pid.error_integral + error, error - pid.prev_error]
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
X_train, y_train = collect_data(train_files, max_samples=1000)
print(f"Collected {len(X_train)} training samples")

# Train
net = nn.Linear(3, 1, bias=False)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5000):
    loss = nn.MSELoss()(net(X_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

w = net.weight.data.numpy()[0]
print(f"Learned: P={w[0]:.3f} I={w[1]:.3f} D={w[2]:.3f}")

# Save model
torch.save({'model_state_dict': net.state_dict()}, 'experiments/exp017_baseline/model.pth')
print("✅ Model saved")

# Evaluate on actual routes
class LearnedController:
    def __init__(self, net):
        self.net = net
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        x = torch.FloatTensor([[error, self.error_integral, error_diff]])
        with torch.no_grad():
            action = self.net(x).item()
        self.prev_error = error
        return float(max(-2.0, min(2.0, action)))

def evaluate_cost(files, num_eval=10):
    costs = []
    for f in files[:num_eval]:
        controller = LearnedController(net)
        sim = TinyPhysicsSimulator(model, str(f), controller=controller)
        cost_dict = sim.rollout()
        costs.append(cost_dict['total_cost'])
    return sum(costs) / len(costs)

print("\nEvaluating costs on routes:")
train_cost = evaluate_cost(train_files, num_eval=10)
val_cost = evaluate_cost(val_files, num_eval=10)
test_cost = evaluate_cost(test_files, num_eval=10)

print(f"Train cost: {train_cost:.2f}")
print(f"Val cost: {val_cost:.2f}")
print(f"Test cost: {test_cost:.2f}")
