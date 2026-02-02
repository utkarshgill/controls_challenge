"""
SIMPLE: 1 neuron, 3 weights, no bias, behavioral cloning from PID

Input: [error, error * v_ego, v_ego] (3D)
Output: steering

That's it.
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
from data_split import get_data_split

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Collect PID data
print("Collecting PID demonstrations...")
data_split = get_data_split()
train_files = data_split['train'][:1000]  # Just 1000 files

states, actions = [], []

for f in tqdm(train_files):
    controller = PIDController()
    sim = TinyPhysicsSimulator(model_onnx, f, controller=controller)
    
    orig_update = controller.update
    def capture(target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        # Simple 3D state
        s = [error, error * state.v_ego, state.v_ego / 30.0]  # Normalize v_ego
        action = orig_update(target_lataccel, current_lataccel, state, future_plan)
        states.append(s)
        actions.append(action)
        return action
    
    controller.update = capture
    sim.rollout()

states = torch.FloatTensor(states)
actions = torch.FloatTensor(actions).unsqueeze(1)

print(f"Collected {len(states)} samples")

# Train: 1 neuron, 3 weights, no bias
net = nn.Linear(3, 1, bias=False)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

print("Training...")
for epoch in range(100):
    pred = net(states)
    loss = nn.MSELoss()(pred, actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}")

print(f"\nWeights: {net.weight.data.numpy()}")

# Save
torch.save({'model_state_dict': net.state_dict()}, 
           Path(__file__).parent / 'simple.pth')
print("✅ Saved")



