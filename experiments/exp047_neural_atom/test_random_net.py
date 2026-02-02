"""
Test what a random network outputs
"""

import numpy as np
import torch
import torch.nn as nn

# Create random network
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

net = TinyNet()

# Sample some random observations
np.random.seed(42)
obs = np.random.randn(1000, 4).astype(np.float32)

# Get actions
with torch.no_grad():
    actions = net(torch.from_numpy(obs)).numpy().flatten()

print("Random network statistics:")
print(f"  Mean action: {np.mean(actions):.4f}")
print(f"  Std action: {np.std(actions):.4f}")
print(f"  Min action: {np.min(actions):.4f}")
print(f"  Max action: {np.max(actions):.4f}")
print()
print(f"Action distribution is centered near 0, which is reasonable for straight sections!")
print(f"This explains why deterministic eval (using mean) works well,")
print(f"but stochastic training (with σ≈1.0 noise) performs poorly.")
