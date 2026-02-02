"""Debug what FF network outputs"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import ActorCritic, prepare_future_plan
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, FuturePlan, State
from controllers.pid import Controller as PIDController

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)
test_file = sorted(DATA_PATH.glob('*.csv'))[0]

# Create untrained network
actor_critic = ActorCritic()

# Run a few steps and collect FF outputs
controller = PIDController()
sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)

ff_outputs = []
for _ in range(100):
    if sim.step_idx >= len(sim.data) - 1:
        break
    
    state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
    future_plan_array = prepare_future_plan(futureplan, state)
    
    ff_action, raw_ff = actor_critic.act(future_plan_array, deterministic=True)
    ff_outputs.append(ff_action)
    
    sim.step()

ff_outputs = np.array(ff_outputs)
print(f"FF outputs over 100 steps:")
print(f"  Mean: {ff_outputs.mean():.4f}")
print(f"  Std: {ff_outputs.std():.4f}")
print(f"  Range: [{ff_outputs.min():.4f}, {ff_outputs.max():.4f}]")
print(f"\nFor reference:")
print(f"  PID actions are typically in range [-2, 2]")
print(f"  FF should start near 0 for good initialization")
print(f"  Current FF is {'GOOD (near zero)' if abs(ff_outputs.mean()) < 0.01 else 'BAD (too large)'}")

