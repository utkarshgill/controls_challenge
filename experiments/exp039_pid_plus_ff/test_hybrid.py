"""Test hybrid controller vs PID"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from train import ActorCritic, HybridController, tinyphysics_model, rollout_episode
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsSimulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'

# Test on first file
test_file = sorted(DATA_PATH.glob('*.csv'))[0]

print("Testing on:", test_file.name)
print("="*60)

# Test 1: Pure PID
print("\n1. Pure PID:")
pid_controller = PIDController()
cost_dict = rollout_episode(test_file, pid_controller)
print(f"   Cost: {cost_dict['total_cost']:.2f}")
print(f"   (lataccel: {cost_dict['lataccel_cost']:.2f}, jerk: {cost_dict['jerk_cost']:.2f})")

# Test 2: Hybrid (untrained)
print("\n2. Hybrid (PID + untrained FF):")
actor_critic = ActorCritic()
hybrid_controller = HybridController(actor_critic, deterministic=True)
cost_dict = rollout_episode(test_file, hybrid_controller)
print(f"   Cost: {cost_dict['total_cost']:.2f}")
print(f"   (lataccel: {cost_dict['lataccel_cost']:.2f}, jerk: {cost_dict['jerk_cost']:.2f})")

print("\n" + "="*60)
print("If hybrid cost >> PID cost, something is wrong with FF network")
print("If hybrid cost ≈ PID cost, FF network outputs ≈ 0 (good initialization)")

