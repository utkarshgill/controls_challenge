"""Debug script to check radius values"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

# Load simulator
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
DATA_PATH = Path(__file__).parent.parent.parent / 'data'
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# Pick a file
data_file = list(DATA_PATH.glob('*.csv'))[0]
print(f"Analyzing: {data_file.name}\n")

controller = PIDController()
sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)

radius_list = []
step_count = 0

original_update = controller.update

def capture(target_lataccel, current_lataccel, state, future_plan):
    global step_count
    
    # Compute radius WITH per-timestep velocities
    future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
    future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
    future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
    
    # Radius = v² / (lataccel - roll)
    # Handle division by zero: cap at large value
    denominator = future_lataccel - future_roll
    # Avoid division by zero - use sign-preserving clipping
    denominator_safe = np.where(
        np.abs(denominator) < 0.01,  # threshold
        np.sign(denominator) * 0.01,  # replace with min value
        denominator
    )
    
    v_squared = future_v_ego ** 2
    radius = v_squared / denominator_safe
    
    # Clip to reasonable range
    radius_clipped = np.clip(radius, -10000, 10000)
    
    radius_list.append(radius_clipped)
    
    if step_count < 5:
        print(f"Step {step_count}:")
        print(f"  lataccel - roll: {denominator[:5]}")
        print(f"  v²: {v_squared[:5]}")
        print(f"  radius (raw): {radius[:5]}")
        print(f"  radius (clipped): {radius_clipped[:5]}")
        print(f"  radius range: [{radius_clipped.min():.1f}, {radius_clipped.max():.1f}]")
        print(f"  radius std: {radius_clipped.std():.1f}\n")
    
    step_count += 1
    return original_update(target_lataccel, current_lataccel, state, future_plan)

controller.update = capture
sim.rollout()

all_radius = np.array(radius_list)
print(f"\n{'='*60}")
print(f"Overall Statistics ({len(radius_list)} steps):")
print(f"  Radius mean: {all_radius.mean():.1f}")
print(f"  Radius std: {all_radius.std():.1f}")
print(f"  Radius min: {all_radius.min():.1f}")
print(f"  Radius max: {all_radius.max():.1f}")
print(f"  Per-step variation: {all_radius.std(axis=1).mean():.1f}")
print(f"{'='*60}")

print(f"\nComparison:")
print(f"  Curvature magnitude: ~0.0003")
print(f"  Radius magnitude: ~{abs(all_radius.mean()):.1f}")
print(f"  Improvement: {abs(all_radius.mean()) / 0.0003:.0f}x better scale!")
