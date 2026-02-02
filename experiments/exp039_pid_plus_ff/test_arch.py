"""
Test architecture before full training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import ActorCritic, HybridController, prepare_future_plan, rollout_episode, tinyphysics_model, FuturePlan, State
from tinyphysics import TinyPhysicsSimulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

print("Testing architecture...")
print("="*60)

# Test 1: Network forward pass
print("\n1. Testing network forward pass...")
actor_critic = ActorCritic()
dummy_future_plan = torch.randn(1, 4, 50)
mean, std, value = actor_critic(dummy_future_plan)
print(f"   Input shape: {dummy_future_plan.shape}")
print(f"   Output mean: {mean.shape}, std: {std.shape}, value: {value.shape}")
print(f"   Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
print(f"   ✓ Network forward pass works")

# Test 2: Action sampling
print("\n2. Testing action sampling...")
ff_action, raw_action = actor_critic.act(dummy_future_plan.numpy()[0])
print(f"   FF action: {ff_action:.4f}, Raw: {raw_action:.4f}")
print(f"   ✓ Action sampling works")

# Test 3: Controller integration
print("\n3. Testing controller integration...")
test_file = sorted(DATA_PATH.glob('*.csv'))[0]

controller = HybridController(actor_critic)
sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)

# Take a few steps
for _ in range(5):
    sim.step()

print(f"   Took 5 steps successfully")
print(f"   Current lataccel: {sim.current_lataccel:.3f}")
print(f"   ✓ Controller integration works")

# Test 4: Full episode
print("\n4. Testing full episode rollout...")
controller.reset()
cost_dict = rollout_episode(test_file, controller)
print(f"   Episode completed")
print(f"   Total cost: {cost_dict['total_cost']:.2f}")
print(f"   Lataccel cost: {cost_dict['lataccel_cost']:.2f}")
print(f"   Jerk cost: {cost_dict['jerk_cost']:.2f}")
print(f"   ✓ Full episode works")

# Test 5: Compare to pure PID
print("\n5. Comparing to pure PID baseline...")
from controllers.pid import Controller as PIDController
pid_controller = PIDController()
pid_sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=pid_controller, debug=False)
pid_cost = pid_sim.rollout()
print(f"   PID cost: {pid_cost['total_cost']:.2f}")
print(f"   Ours cost (untrained): {cost_dict['total_cost']:.2f}")
print(f"   Difference: {cost_dict['total_cost'] - pid_cost['total_cost']:.2f}")
print(f"   (Should be near 0 since FF network initialized to output ~0)")

print("\n" + "="*60)
print("✓ All architecture tests passed!")
print("\nReady to train:")
print("  python train.py")
