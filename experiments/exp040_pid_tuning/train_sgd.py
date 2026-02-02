"""
SGD training - EXACT copy of exp016 approach
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class OneNeuronNetwork(nn.Module):
    """1 neuron: 3 weights, no bias"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)


print("="*80)
print("Training Single Neuron with SGD - MUST recover PID")
print("="*80)

# Load cache
cache_path = Path(__file__).parent / 'pid_demonstrations_1000.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
    states = data['states']
    actions = data['actions']

print(f"\nLoaded: {states.shape}, {actions.shape}")

# Convert to tensors
X = torch.FloatTensor(states)
y = torch.FloatTensor(actions)

# Create model
model = OneNeuronNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"\nTraining with SGD (lr=0.01, 100 epochs)...")
print(f"Expected: P=0.195, I=0.100, D=-0.053\n")

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        weights = model.linear.weight.data.numpy()[0]
        print(f"Epoch {epoch+1:3d}/100 | Loss: {loss.item():.8f} | "
              f"P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")

# Final results
weights = model.linear.weight.data.numpy()[0]
print(f"\n{'='*80}")
print(f"FINAL RESULTS:")
print(f"{'='*80}")
print(f"P: {weights[0]:+.9f}  (PID: +0.195000) | Δ={abs(weights[0] - 0.195):.9f}")
print(f"I: {weights[1]:+.9f}  (PID: +0.100000) | Δ={abs(weights[1] - 0.100):.9f}")
print(f"D: {weights[2]:+.9f}  (PID: -0.053000) | Δ={abs(weights[2] + 0.053):.9f}")

total_error = abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053)
print(f"\nTotal Δ: {total_error:.9f}")

if total_error < 0.001:
    print("✅ SUCCESS - SGD recovered PID!")
else:
    print("❌ FAILED")
    print(f"\nDEBUG INFO:")
    print(f"X stats: mean={X.mean(dim=0).numpy()}, std={X.std(dim=0).numpy()}")
    print(f"y stats: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
    print(f"X condition number: {np.linalg.cond(states.T @ states):.2e}")

