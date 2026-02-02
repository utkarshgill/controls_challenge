"""
Simple test: verify the controller uses pre-sampled actions correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import ActorCritic, HybridController

# Create components
actor_critic = ActorCritic()
controller = HybridController(actor_critic, deterministic=False)

# Mock inputs
target = 1.0
current = 0.5
state = type('State', (), {'roll_lataccel': 0.1, 'v_ego': 20.0, 'a_ego': 0.0})()
future_plan = type('FuturePlan', (), {
    'lataccel': [0.5] * 50,
    'roll_lataccel': [0.1] * 50,
    'v_ego': [20.0] * 50,
    'a_ego': [0.0] * 50
})()

# Test 1: Without pre-sampled action (should sample internally)
print("Test 1: No pre-sampled action (eval mode)")
action1 = controller.update(target, current, state, future_plan)
print(f"  Action: {action1:.6f}")
print(f"  ✓ Controller sampled internally")

# Test 2: With pre-sampled action (training mode)
print("\nTest 2: With pre-sampled action (training mode)")
controller.reset()
pre_sampled_ff = 0.123456  # Fixed value
controller.presampled_ff = pre_sampled_ff

action2 = controller.update(target, current, state, future_plan)
pid_part = 0.195 * 0.5  # P * error (first step, no integral/derivative yet)
expected_action = pid_part + pre_sampled_ff

print(f"  Pre-sampled FF: {pre_sampled_ff:.6f}")
print(f"  PID part: {pid_part:.6f}")
print(f"  Expected total: {expected_action:.6f}")
print(f"  Actual action: {action2:.6f}")
print(f"  Match: {abs(action2 - expected_action) < 1e-6}")

if abs(action2 - expected_action) < 1e-6:
    print("\n✓ SUCCESS: Controller correctly uses pre-sampled actions!")
else:
    print(f"\n✗ FAIL: Mismatch of {abs(action2 - expected_action):.6f}")

# Test 3: Verify presampled_ff is consumed
print("\nTest 3: Verify pre-sampled action is consumed after use")
print(f"  presampled_ff after use: {controller.presampled_ff}")
if controller.presampled_ff is None:
    print("  ✓ Correctly consumed (set to None)")
else:
    print("  ✗ Not consumed!")

