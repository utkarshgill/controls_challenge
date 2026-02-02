#!/usr/bin/env python3
"""Quick evaluation of the trained linear controller"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

# Load the trained model
checkpoint = torch.load('ppo_best.pth', map_location='cpu', weights_only=False)
print("=" * 80)
print("Experiment 034: Linear Preview Controller Evaluation")
print("=" * 80)

# Print epoch and cost
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Best eval cost: {checkpoint['cost']:.2f}")

# Print the learned weights
state_dict = checkpoint['model_state_dict']
print(f"\nModel keys: {list(state_dict.keys())}")

if 'actor.weight' in state_dict:
    weights = state_dict['actor.weight'].numpy()[0]  # [w1, w2, w3]
    bias = state_dict['actor.bias'].numpy()[0] if 'actor.bias' in state_dict else 0.0
    print(f"\nLearned Linear Weights:")
    print(f"  w1 (near term, 0.1-0.6s):  {weights[0]:8.4f}")
    print(f"  w2 (mid horizon, 1-2.5s):  {weights[1]:8.4f}")
    print(f"  w3 (long term mass):       {weights[2]:8.4f}")
    print(f"  bias:                      {bias:8.4f}")
    
    # Print magnitude
    mag = np.linalg.norm(weights)
    print(f"\n  Weight magnitude: {mag:.4f}")
    print(f"  Dominant feature: f{np.argmax(np.abs(weights)) + 1}")
    
    print(f"\nInterpretation:")
    if checkpoint['cost'] < 75:
        print(f"  ✅ PREVIEW HELPS! Beat PID baseline ({checkpoint['cost']:.1f} < 75)")
    else:
        print(f"  ❌ Preview doesn't help ({checkpoint['cost']:.1f} >= 75)")
elif 'actor.0.weight' in state_dict:
    # MLP case
    print("\nWARNING: Model has hidden layers (not pure linear)")
    
# Compare to PID baseline
print("\nPID baseline: ~75-100 cost")
print("Target (winners): <45 cost")
print("=" * 80)

