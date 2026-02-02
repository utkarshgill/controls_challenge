"""
DEAD SIMPLE: Single neuron learns PID with SGD
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import pickle

# Load collected PID states
cache_path = Path(__file__).parent / 'pid_demonstrations_1000.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
    X_np = data['states']  # [error, error_integral, error_diff]
    y_np = data['actions']  # PID actions

print(f"Data loaded: {X_np.shape}")
print(f"X std: {X_np.std(axis=0)}")

# Convert to tensors
X = torch.FloatTensor(X_np)
y = torch.FloatTensor(y_np)

# Model: 1 linear layer, 3 weights, no bias
model = nn.Linear(3, 1, bias=False)

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
print("\nTraining with SGD (lr=0.001)...")
for epoch in range(1000):
    # Forward
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        w = model.weight.data.numpy()[0]
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.8f} | P={w[0]:+.6f} I={w[1]:+.6f} D={w[2]:+.6f}")

# Results from SGD
w_sgd = model.weight.data.numpy()[0]
print(f"\n{'='*60}")
print(f"SGD RESULT:")
print(f"  P: {w_sgd[0]:+.6f}  (PID: +0.195000) | Error: {abs(w_sgd[0]-0.195):.6f}")
print(f"  I: {w_sgd[1]:+.6f}  (PID: +0.100000) | Error: {abs(w_sgd[1]-0.100):.6f}")
print(f"  D: {w_sgd[2]:+.6f}  (PID: -0.053000) | Error: {abs(w_sgd[2]+0.053):.6f}")
print(f"{'='*60}")

# Compare with analytical solution
print(f"\nANALYTICAL SOLUTION (np.linalg.lstsq):")
w_analytical = np.linalg.lstsq(X_np, y_np.flatten(), rcond=None)[0]
print(f"  P: {w_analytical[0]:+.9f}  (PID: +0.195000) | Error: {abs(w_analytical[0]-0.195):.9f}")
print(f"  I: {w_analytical[1]:+.9f}  (PID: +0.100000) | Error: {abs(w_analytical[1]-0.100):.9f}")
print(f"  D: {w_analytical[2]:+.9f}  (PID: -0.053000) | Error: {abs(w_analytical[2]+0.053):.9f}")
print(f"{'='*60}")

