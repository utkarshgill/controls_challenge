"""
Try with Adam optimizer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import pickle

# Load
cache_path = Path(__file__).parent / 'pid_demonstrations_1000.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
    X_np = data['states']
    y_np = data['actions']

X = torch.FloatTensor(X_np)
y = torch.FloatTensor(y_np)

print(f"Data: {X.shape}")

# Model
model = nn.Linear(3, 1, bias=False)

# Try ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("\nTraining with ADAM (lr=0.01)...")
for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        w = model.weight.data.numpy()[0]
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.8f} | P={w[0]:+.6f} I={w[1]:+.6f} D={w[2]:+.6f}")

w = model.weight.data.numpy()[0]
print(f"\n{'='*60}")
print(f"ADAM RESULT:")
print(f"  P: {w[0]:+.9f}  (PID: +0.195000) | Δ={abs(w[0]-0.195):.9f}")
print(f"  I: {w[1]:+.9f}  (PID: +0.100000) | Δ={abs(w[1]-0.100):.9f}")
print(f"  D: {w[2]:+.9f}  (PID: -0.053000) | Δ={abs(w[2]+0.053):.9f}")
total_error = abs(w[0]-0.195) + abs(w[1]-0.100) + abs(w[2]+0.053)
print(f"\nTotal Δ: {total_error:.9f}")
if total_error < 0.001:
    print("✅ SUCCESS")
else:
    print("❌ FAILED")
print(f"{'='*60}")

