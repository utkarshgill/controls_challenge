"""
SGD with feature normalization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

torch.manual_seed(42)
np.random.seed(42)

class OneNeuronNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)

print("="*80)
print("SGD with Feature Normalization")
print("="*80)

# Load
cache_path = Path(__file__).parent / 'pid_demonstrations_1000.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
    X_raw = data['states']
    y_raw = data['actions']

print(f"\nRaw data: {X_raw.shape}")
print(f"Raw X std: {X_raw.std(axis=0)}")

# NORMALIZE features
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_norm = (X_raw - X_mean) / X_std

print(f"Normalized X std: {X_norm.std(axis=0)}")

# Convert to tensors
X = torch.FloatTensor(X_norm)
y = torch.FloatTensor(y_raw)

# Train
model = OneNeuronNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"\nTraining with SGD (lr=0.01, 500 epochs)...")

for epoch in range(500):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        weights_norm = model.linear.weight.data.numpy()[0]
        # Unnormalize weights
        weights = weights_norm / X_std
        print(f"Epoch {epoch+1:3d}/500 | Loss: {loss.item():.8f} | "
              f"P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")

# Final
weights_norm = model.linear.weight.data.numpy()[0]
weights = weights_norm / X_std

print(f"\n{'='*80}")
print(f"RESULTS:")
print(f"{'='*80}")
print(f"P: {weights[0]:+.9f}  (PID: +0.195000) | Δ={abs(weights[0] - 0.195):.9f}")
print(f"I: {weights[1]:+.9f}  (PID: +0.100000) | Δ={abs(weights[1] - 0.100):.9f}")
print(f"D: {weights[2]:+.9f}  (PID: -0.053000) | Δ={abs(weights[2] + 0.053):.9f}")

total_error = abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053)
print(f"\nTotal Δ: {total_error:.9f}")

if total_error < 0.001:
    print("✅ SUCCESS!")
else:
    print("❌ FAILED")

