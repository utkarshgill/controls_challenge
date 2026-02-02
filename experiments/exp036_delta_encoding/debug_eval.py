#!/usr/bin/env python3
"""Debug evaluation - print features and residuals"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load checkpoint
checkpoint = torch.load('ppo_best.pth', map_location='cpu', weights_only=False)
weights = checkpoint['model_state_dict']['actor.weight'].numpy()[0]
bias = checkpoint['model_state_dict']['actor.bias'].numpy()[0]

print(f"Weights: w1={weights[0]:.4f}, w2={weights[1]:.4f}, w3={weights[2]:.4f}, bias={bias:.4f}")
print()

# Now let's manually test on one route
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

model_path = "../../models/tinyphysics.onnx"
data_path = "../../data/00000.csv"

import pandas as pd

# Read data
df = pd.read_csv(data_path)
print(f"Data length: {len(df)} steps")

# Compute features at a few timesteps
for step_idx in [100, 150, 200]:
    # Get future lataccel
    future_lat = []
    for i in range(50):
        future_idx = step_idx + i
        if future_idx < len(df):
            future_lat.append(df['targetLateralAcceleration'].values[future_idx])
        else:
            future_lat.append(future_lat[-1] if future_lat else 0.0)
    
    # Compute differential features
    baseline = future_lat[0]
    delta = np.array([lat - baseline for lat in future_lat])
    
    f1 = np.mean(delta[1:6])
    f2 = np.mean(delta[10:25])
    f3 = np.mean(delta)
    
    # Normalize (LATACCEL_SCALE = 5.0 in training script)
    f1_norm = f1 / 5.0
    f2_norm = f2 / 5.0
    f3_norm = f3 / 5.0
    
    # Compute residual
    residual = weights[0] * f1_norm + weights[1] * f2_norm + weights[2] * f3_norm + bias
    
    print(f"Step {step_idx}:")
    print(f"  Raw features: f1={f1:.4f}, f2={f2:.4f}, f3={f3:.4f}")
    print(f"  Normalized:   f1={f1_norm:.4f}, f2={f2_norm:.4f}, f3={f3_norm:.4f}")
    print(f"  Residual (before tanh/clip/filter): {residual:.4f}")
    print(f"  After tanh: {np.tanh(residual):.4f}")
    print()

