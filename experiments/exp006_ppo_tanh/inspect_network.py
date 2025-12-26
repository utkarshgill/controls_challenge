#!/usr/bin/env python3
"""Inspect what the network learned"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
from controllers.ppo import ActorCritic, build_state, STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS
from tinyphysics import State, FuturePlan, STEER_RANGE

# Load model
actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS)
actor_critic.load_state_dict(torch.load('results/checkpoints/ppo_best.pth', map_location='cpu'))
actor_critic.eval()

print("="*80)
print("NETWORK INSPECTION")
print("="*80)

# Check learned std
log_std = actor_critic.log_std.item()
std = np.exp(log_std)
print(f"\nLearned exploration:")
print(f"  log_std: {log_std:.4f}")
print(f"  std:     {std:.4f}")
print(f"  ± 2σ range: [{-2*std:.4f}, {2*std:.4f}] before tanh")

if std < 0.1:
    print("  ⚠️  Very low exploration! Network is nearly deterministic.")

# Test on various states
print("\n" + "="*80)
print("NETWORK SENSITIVITY TEST")
print("="*80)

def test_state(target, current, v_ego, a_ego, future_values, desc):
    """Test network output on a state"""
    state = State(roll_lataccel=0.0, v_ego=v_ego, a_ego=a_ego)
    future = FuturePlan(
        lataccel=future_values,
        roll_lataccel=[0]*50,
        v_ego=[v_ego]*50,
        a_ego=[a_ego]*50
    )
    
    state_vec = build_state(target, current, state, future)
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
    
    with torch.no_grad():
        action_mean, action_std, value = actor_critic(state_tensor)
        action = torch.tanh(action_mean) * STEER_RANGE[1]
    
    print(f"\n{desc}:")
    print(f"  error: {target - current:.3f}, v_ego: {v_ego:.1f}, a_ego: {a_ego:.2f}")
    print(f"  → action_mean (raw): {action_mean.item():.4f}")
    print(f"  → action (tanh): {action.item():.4f}")
    print(f"  → value estimate: {value.item():.4f}")

# Test scenarios
test_state(
    target=1.0, current=0.0, v_ego=30.0, a_ego=0.0,
    future_values=[1.0]*50,
    desc="Large positive error (need to turn)"
)

test_state(
    target=0.0, current=1.0, v_ego=30.0, a_ego=0.0,
    future_values=[0.0]*50,
    desc="Large negative error (need to straighten)"
)

test_state(
    target=0.0, current=0.0, v_ego=30.0, a_ego=0.0,
    future_values=[0.0]*50,
    desc="No error (straight road)"
)

test_state(
    target=2.0, current=0.0, v_ego=30.0, a_ego=0.0,
    future_values=[2.0]*50,
    desc="Sharp turn coming"
)

test_state(
    target=0.0, current=0.0, v_ego=10.0, a_ego=0.0,
    future_values=[0.0]*50,
    desc="Slow speed"
)

test_state(
    target=0.0, current=0.0, v_ego=40.0, a_ego=0.0,
    future_values=[0.0]*50,
    desc="High speed"
)

test_state(
    target=0.0, current=0.0, v_ego=30.0, a_ego=3.0,
    future_values=[0.0]*50,
    desc="Hard acceleration"
)

test_state(
    target=0.0, current=0.0, v_ego=30.0, a_ego=-3.0,
    future_values=[0.0]*50,
    desc="Hard braking"
)

# Test sensitivity to future plan
print("\n" + "="*80)
print("FUTURE PLAN SENSITIVITY")
print("="*80)

future_tests = [
    ([0.0]*50, "Straight ahead"),
    ([1.0]*50, "Constant curve"),
    ([0.0, 0.5, 1.0, 1.5, 2.0] + [2.0]*45, "Increasing curve"),
    ([2.0, 1.5, 1.0, 0.5, 0.0] + [0.0]*45, "Decreasing curve"),
    ([1.0, -1.0]*25, "S-curve"),
]

for future_vals, desc in future_tests:
    state = State(roll_lataccel=0.0, v_ego=30.0, a_ego=0.0)
    future = FuturePlan(lataccel=future_vals, roll_lataccel=[0]*50, v_ego=[30]*50, a_ego=[0]*50)
    state_vec = build_state(0.0, 0.0, state, future)
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
    
    with torch.no_grad():
        action_mean, _, _ = actor_critic(state_tensor)
        action = torch.tanh(action_mean) * STEER_RANGE[1]
    
    print(f"{desc:<20} → action: {action.item():7.4f}")

# Check if network is outputting similar values for everything
print("\n" + "="*80)
print("OUTPUT VARIANCE TEST")
print("="*80)

outputs = []
for _ in range(1000):
    # Random states
    target = np.random.uniform(-2, 2)
    current = np.random.uniform(-2, 2)
    v_ego = np.random.uniform(10, 40)
    a_ego = np.random.uniform(-3, 3)
    future_vals = np.random.uniform(-2, 2, 50).tolist()
    
    state = State(roll_lataccel=0.0, v_ego=v_ego, a_ego=a_ego)
    future = FuturePlan(lataccel=future_vals, roll_lataccel=[0]*50, v_ego=[v_ego]*50, a_ego=[a_ego]*50)
    state_vec = build_state(target, current, state, future)
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
    
    with torch.no_grad():
        action_mean, _, _ = actor_critic(state_tensor)
        action = torch.tanh(action_mean) * STEER_RANGE[1]
    
    outputs.append(action.item())

outputs = np.array(outputs)
print(f"\nOn 1000 random states:")
print(f"  Mean action:  {outputs.mean():.4f}")
print(f"  Std action:   {outputs.std():.4f}")
print(f"  Min action:   {outputs.min():.4f}")
print(f"  Max action:   {outputs.max():.4f}")
print(f"  Range:        {outputs.max() - outputs.min():.4f}")

if outputs.std() < 0.1:
    print("\n⚠️  WARNING: Output variance is very low!")
    print("    Network is outputting nearly constant values.")
    print("    This suggests it hasn't learned useful representations.")

print("\n" + "="*80)

