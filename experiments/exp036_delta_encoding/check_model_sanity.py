#!/usr/bin/env python3
"""Check if model weights are sane"""
import torch
import numpy as np

checkpoint = torch.load('ppo_best.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print("Checking model sanity...")
print("=" * 80)

has_nan = False
has_inf = False

for key, tensor in state_dict.items():
    if torch.isnan(tensor).any():
        print(f"❌ {key} contains NaN!")
        has_nan = True
    if torch.isinf(tensor).any():
        print(f"❌ {key} contains Inf!")
        has_inf = True
    print(f"✓ {key}: shape={tuple(tensor.shape)}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")

print("=" * 80)
if has_nan or has_inf:
    print("MODEL HAS NaN OR INF - TRAINING DIVERGED")
else:
    print("Model weights are finite and reasonable")

print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Cost: {checkpoint['cost']:.2f}")

