"""
Inspect available inputs to controller
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Run one simulation and capture inputs
model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)
data_file = sorted(Path('./data').glob('*.csv'))[0]

print(f"Analyzing: {data_file.name}\n")
print("="*80)

class InspectorController:
    def __init__(self):
        self.step = 0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.step < 3:  # Print first 3 steps
            print(f"\nStep {self.step}:")
            print(f"  target_lataccel: {target_lataccel:.4f} (scalar)")
            print(f"  current_lataccel: {current_lataccel:.4f} (scalar)")
            print(f"  state.v_ego: {state.v_ego:.4f}")
            print(f"  state.a_ego: {state.a_ego:.4f}")
            print(f"  state.roll_lataccel: {state.roll_lataccel:.4f}")
            print(f"  future_plan.lataccel: length {len(future_plan.lataccel)}, type {type(future_plan.lataccel)}")
            print(f"    first 5: {future_plan.lataccel[:5]}")
            print(f"    last 5: {future_plan.lataccel[-5:]}")
            print(f"    min/max: {min(future_plan.lataccel):.4f} / {max(future_plan.lataccel):.4f}")
        
        self.step += 1
        return 0.0  # dummy action

controller = InspectorController()
sim = TinyPhysicsSimulator(model, str(data_file), controller=controller)
sim.rollout()

print(f"\n{'='*80}")
print(f"Total steps: {controller.step}")
print(f"{'='*80}\n")

# Summary statistics
print("AVAILABLE INPUTS:")
print("1. Scalars:")
print("   - target_lataccel (what we want)")
print("   - current_lataccel (what we have)")
print("   - state.v_ego (velocity)")
print("   - state.a_ego (forward acceleration)")
print("   - state.roll_lataccel (road banking)")
print()
print("2. Future plan:")
print(f"   - future_plan.lataccel: array of length {len(sim.futureplan.lataccel)}")
print("   - This is the TARGET trajectory ahead")
print()
print("3. What we can derive:")
print("   - error = target - current")
print("   - error_integral (accumulate)")
print("   - error_diff (derivative)")
print("   - future curvature = target / v_ego²")
print()
print("STATE REPRESENTATION OPTIONS:")
print("Option 1 (PID): [error, error_integral, error_diff] → 3D")
print("Option 2 (+ velocity): [error, error_integral, error_diff, v_ego/30] → 4D")
print("Option 3 (+ future): [error, ei, ed, v_ego/30, future[0:5]/5] → 9D")
print("Option 4 (rich): [error, ei, ed, v_ego/30, a_ego/5, roll, future[0:10]/5] → 16D")

