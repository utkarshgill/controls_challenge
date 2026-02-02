"""
Test that the action we store matches the action the controller uses.
This verifies the fix for the action mismatch bug.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from train import ActorCritic, HybridController, prepare_future_plan, tinyphysics_model
from tinyphysics import TinyPhysicsSimulator

# Create actor-critic
actor_critic = ActorCritic()

# Create controller
controller = HybridController(actor_critic, deterministic=False)

# Get a test file
data_files = sorted(Path(__file__).parent.parent.parent.glob('data/*.csv'))
test_file = data_files[0]

# Track actions
stored_actions = []
executed_actions = []

# Monkey patch to capture executed actions
original_update = controller.update
def capture_update(target, current, state, future_plan):
    # Capture the FF action that will be used
    if controller.presampled_ff is not None:
        executed_actions.append(controller.presampled_ff)
    result = original_update(target, current, state, future_plan)
    return result
controller.update = capture_update

# Run episode
sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)

while sim.step_idx < len(sim.data) - 1:
    state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
    future_plan_array = prepare_future_plan(futureplan, state)
    
    # Sample action
    ff_action, raw_ff = actor_critic.act(future_plan_array, deterministic=False)
    stored_actions.append(ff_action)
    
    # Give to controller
    controller.presampled_ff = ff_action
    
    # Step
    sim.step()

# Compare
stored_actions = np.array(stored_actions)
executed_actions = np.array(executed_actions)

print(f"Stored actions:  {len(stored_actions)}")
print(f"Executed actions: {len(executed_actions)}")
print(f"Match: {np.allclose(stored_actions, executed_actions)}")
print(f"Max diff: {np.max(np.abs(stored_actions - executed_actions)):.10f}")

if np.allclose(stored_actions, executed_actions):
    print("\n✓ SUCCESS: Actions match perfectly!")
else:
    print("\n✗ FAIL: Actions don't match!")
    print(f"First 5 stored:  {stored_actions[:5]}")
    print(f"First 5 executed: {executed_actions[:5]}")

