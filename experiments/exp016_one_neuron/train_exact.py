"""
EXACT PID Recovery: 1 neuron, 3 weights, MSE, SGD

Input: [error, error_integral, error_diff]
Output: PID action
NO normalization. Just learn it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Collect PID data
print("Collecting PID data...")
data_files = sorted(Path('./data').glob('*.csv'))[:1000]

states, actions = [], []

for f in tqdm(data_files):
    controller = PIDController()
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=controller)
    
    orig_update = controller.update
    def capture(target_lataccel, current_lataccel, state, future_plan):
        # BEFORE PID updates
        error = target_lataccel - current_lataccel
        old_ei = controller.error_integral
        old_pe = controller.prev_error
        
        # Call PID
        action = orig_update(target_lataccel, current_lataccel, state, future_plan)
        
        # State PID used: [error, NEW error_integral, error_diff]
        new_ei = controller.error_integral  # After += error
        error_diff = error - old_pe
        
        states.append([error, new_ei, error_diff])
        actions.append(action)
        return action
    
    controller.update = capture
    sim.rollout()

states = torch.FloatTensor(states)
actions = torch.FloatTensor(actions).unsqueeze(1)

print(f"Collected {len(states)} samples")

# Train: 1 neuron, NO BIAS, NO NORMALIZATION
net = nn.Linear(3, 1, bias=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # SGD not Adam!

print("Training with SGD...")
for epoch in range(1000):
    pred = net(states)
    loss = nn.MSELoss()(pred, actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        w = net.weight.data.numpy()[0]
        print(f"Epoch {epoch+1}: loss={loss.item():.8f} | P={w[0]:.6f} I={w[1]:.6f} D={w[2]:.6f}")

w = net.weight.data.numpy()[0]
print(f"\n{'='*60}")
print(f"Final weights:")
print(f"  P: {w[0]:.6f}  (PID: 0.195000)")
print(f"  I: {w[1]:.6f}  (PID: 0.100000)")
print(f"  D: {w[2]:.6f}  (PID: -0.053000)")
print(f"{'='*60}")

torch.save({'model_state_dict': net.state_dict()}, Path(__file__).parent / 'exact.pth')
print("✅ Saved")



