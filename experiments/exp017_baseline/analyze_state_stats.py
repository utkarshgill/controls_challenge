"""
Analyze actual state statistics from training data
Measure: error, error_integral, error_diff, v_ego, curvatures
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import random
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Split routes
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:15000]

model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Collect all states
errors = []
error_integrals = []
error_diffs = []
v_egos = []
curvatures = []  # All future curvatures

print("Collecting state statistics from 100 routes...")

for f in train_files[:100]:
    pid = PIDController()
    sim = TinyPhysicsSimulator(model, str(f), controller=pid)
    
    orig = pid.update
    def capture(target, current, state=None, future_plan=None):
        error = target - current
        v_ego = state.v_ego
        
        # Curvatures (future) - ALL 49 points, with safe division
        for i in range(len(future_plan.lataccel)):
            lat = future_plan.lataccel[i]
            curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
            curvatures.append(curv)
        
        errors.append(error)
        error_integrals.append(pid.error_integral + error)
        error_diffs.append(error - pid.prev_error)
        v_egos.append(v_ego)
        
        return orig(target, current, state, future_plan)
    
    pid.update = capture
    sim.rollout()

errors = np.array(errors)
error_integrals = np.array(error_integrals)
error_diffs = np.array(error_diffs)
v_egos = np.array(v_egos)
curvatures = np.array(curvatures)

print("\n" + "="*80)
print("STATE STATISTICS")
print("="*80)

def print_stats(name, data):
    print(f"\n{name}:")
    print(f"  mean:  {np.mean(data):>8.4f}")
    print(f"  std:   {np.std(data):>8.4f}")
    print(f"  min:   {np.min(data):>8.4f}")
    print(f"  max:   {np.max(data):>8.4f}")
    print(f"  p01:   {np.percentile(data, 1):>8.4f}")
    print(f"  p99:   {np.percentile(data, 99):>8.4f}")

print_stats("error", errors)
print_stats("error_integral", error_integrals)
print_stats("error_diff", error_diffs)
print_stats("v_ego", v_egos)
print_stats("curvature", curvatures)

print("\n" + "="*80)
print("RECOMMENDED OBS_SCALE:")
print("="*80)

# Use p99 for scaling (handles outliers)
scale_error = max(abs(np.percentile(errors, 1)), abs(np.percentile(errors, 99)))
scale_ei = max(abs(np.percentile(error_integrals, 1)), abs(np.percentile(error_integrals, 99)))
scale_ed = max(abs(np.percentile(error_diffs, 1)), abs(np.percentile(error_diffs, 99)))
scale_v = np.percentile(v_egos, 99)  # Use p99 as max
scale_curv = max(abs(np.percentile(curvatures, 1)), abs(np.percentile(curvatures, 99)))

print(f"\nOBS_SCALE = np.array([")
print(f"    {scale_error:.4f},  # error")
print(f"    {scale_ei:.4f},  # error_integral")
print(f"    {scale_ed:.4f},  # error_diff")
print(f"    {scale_v:.4f},  # v_ego")
for i in range(49):
    print(f"    {scale_curv:.4f},  # future_curv[{i}]")
print(f"], dtype=np.float32)")
print(f"\nState dimension: 4 (PID + v_ego) + 49 (curvatures) = 53D")
print(f"\nstate_normalized = raw_state / OBS_SCALE")

