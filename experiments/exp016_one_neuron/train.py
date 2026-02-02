"""
Simple: 1 neuron, 3 inputs [error, error_integral, error_diff], train on PID
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Collect 100 samples
model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)
data_files = sorted(Path('./data').glob('*.csv'))

X, y = [], []

for f in data_files:
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
    
    if len(X) >= 100:
        break

X = torch.FloatTensor(X[:100])
y = torch.FloatTensor(y[:100]).unsqueeze(1)

print(f"Collected {len(X)} samples")

# Train
net = nn.Linear(3, 1, bias=False)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5000):
    pred = net(X)
    loss = nn.MSELoss()(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if (epoch + 1) % 1000 == 0:
        w = net.weight.data.numpy()[0]
        print(f"Epoch {epoch+1}: P={w[0]:.3f} I={w[1]:.3f} D={w[2]:.3f}")

torch.save({'model_state_dict': net.state_dict()}, 'experiments/exp016_one_neuron/model.pth')
print("✅ Saved")
