"""Quick test that setup works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from train import OneNeuronActor, OneNeuronController, tinyphysics_model
from tinyphysics import TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
test_file = sorted(DATA_PATH.glob('*.csv'))[0]

print("Testing setup...")
print("="*60)

# Test 1: OneNeuronActor
print("\n1. Testing OneNeuronActor...")
actor = OneNeuronActor()
state = torch.randn(1, 3)
mean, std = actor(state)
action = actor.act(state.numpy()[0], deterministic=True)
print(f"   Actor output: mean={mean.item():.4f}, std={std.item():.4f}, action={action:.4f}")
print(f"   ✓ Actor works")

# Test 2: OneNeuronController
print("\n2. Testing OneNeuronController...")
controller = OneNeuronController(actor, deterministic=True)
sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)
for _ in range(10):
    sim.step()
print(f"   Took 10 steps successfully")
print(f"   ✓ Controller works")

# Test 3: Full episode
print("\n3. Testing full episode...")
controller.reset()
sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)
cost_dict = sim.rollout()
print(f"   Cost: {cost_dict['total_cost']:.2f}")
print(f"   ✓ Full episode works")

# Test 4: Compare to PID
print("\n4. Comparing to PID baseline...")
pid = PIDController()
sim_pid = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=pid, debug=False)
cost_pid = sim_pid.rollout()
print(f"   PID cost: {cost_pid['total_cost']:.2f}")
print(f"   One-neuron (random): {cost_dict['total_cost']:.2f}")
print(f"   (After BC training, should match PID)")

print("\n" + "="*60)
print("✓ All tests passed! Ready to train.")
print("\nRun: python train.py")

