"""Debug script to check curvature values"""
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

# Pick a random file
data_file = list(DATA_PATH.glob('*.csv'))[0]
print(f"Analyzing: {data_file.name}\n")

controller = PIDController()
sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)

curvatures_list = []
step_count = 0

original_update = controller.update

def capture(target_lataccel, current_lataccel, state, future_plan):
    global step_count
    
    # Compute curvatures WITH per-timestep velocities
    future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
    future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
    future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
    
    v_squared = np.maximum(future_v_ego ** 2, 1.0)
    curvatures = (future_lataccel - future_roll) / v_squared
    
    curvatures_list.append(curvatures)
    
    if step_count < 5:
        print(f"Step {step_count}:")
        print(f"  current v_ego: {state.v_ego:.2f} m/s")
        print(f"  future_v_ego[0:5]: {future_v_ego[:5]}")
        print(f"  future_lataccel[0:5]: {future_lataccel[:5]}")
        print(f"  future_roll[0:5]: {future_roll[:5]}")
        print(f"  curvatures[0:5]: {curvatures[:5]}")
        print(f"  curvature range: [{curvatures.min():.6f}, {curvatures.max():.6f}]")
        print(f"  curvature std: {curvatures.std():.6f}\n")
    
    step_count += 1
    return original_update(target_lataccel, current_lataccel, state, future_plan)

controller.update = capture
sim.rollout()

all_curvatures = np.array(curvatures_list)
print(f"\n{'='*60}")
print(f"Overall Statistics ({len(curvatures_list)} steps):")
print(f"  Curvature mean: {all_curvatures.mean():.6f}")
print(f"  Curvature std: {all_curvatures.std():.6f}")
print(f"  Curvature min: {all_curvatures.min():.6f}")
print(f"  Curvature max: {all_curvatures.max():.6f}")
print(f"  Per-step variation (std across time): {all_curvatures.std(axis=1).mean():.6f}")
print(f"{'='*60}")
