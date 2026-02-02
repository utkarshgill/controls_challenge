"""
Find optimal observation scaling based on PID demonstrations.

Insight: Scale features so their EFFECT on action is comparable.
This makes the control law more linear and easier to learn.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Collect from PID
class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.data = []
        self.prev_action = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        
        # Capture state BEFORE action
        state_vec = [
            error,
            self.pid.error_integral + error,  # What it will be
            error - self.pid.prev_error,
            state.v_ego,
            self.prev_action
        ]
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.data.append(state_vec + [action])
        self.prev_action = action
        
        return action

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)

print("Collecting PID demonstrations...")
all_data = []
for f in all_files[:500]:
    collector = PIDDataCollector()
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=collector)
    sim.rollout()
    all_data.extend(collector.data)

data = np.array(all_data)
print(f"Collected {len(data)} samples\n")

X = data[:, :5]  # [error, error_i, error_d, v_ego, prev_action]
y = data[:, 5]   # action

feature_names = ['error', 'error_i', 'error_d', 'v_ego', 'prev_action']

print("FEATURE STATISTICS:")
print("="*70)
for i, name in enumerate(feature_names):
    print(f"{name:15s}: mean={X[:, i].mean():7.3f}, std={X[:, i].std():7.3f}, "
          f"range=[{X[:, i].min():7.3f}, {X[:, i].max():7.3f}]")

print(f"\nAction:          mean={y.mean():7.3f}, std={y.std():7.3f}, "
      f"range=[{y.min():7.3f}, {y.max():7.3f}]")

print("\n" + "="*70)
print("CORRELATION with action:")
print("="*70)
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    print(f"{name:15s}: {corr:6.3f}")

print("\n" + "="*70)
print("LINEAR COEFFICIENTS (action ≈ Σ coef_i * feature_i):")
print("="*70)

# Multiple linear regression
from numpy.linalg import lstsq
coeffs, _, _, _ = lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)
intercept = coeffs[0]
coefs = coeffs[1:]

for i, name in enumerate(feature_names):
    print(f"{name:15s}: {coefs[i]:8.4f}")
print(f"{'intercept':15s}: {intercept:8.4f}")

# Predict and check R²
y_pred = intercept + X @ coefs
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
print(f"\nR² score: {r2:.4f}")

print("\n" + "="*70)
print("OPTIMAL SCALING (to make all coefficients ≈ 1.0):")
print("="*70)
print("Goal: After scaling, linear control law has balanced contributions\n")

# Target: all coefficients around 1.0 after scaling
# If X_scaled = X / scale, then coef_scaled = coef * scale
# We want coef_scaled ≈ 1.0, so scale = coef

optimal_scales = np.abs(coefs)
optimal_scales = optimal_scales / optimal_scales.max()  # Normalize to [0, 1]

for i, name in enumerate(feature_names):
    print(f"{name:15s}: current_coef={coefs[i]:8.4f} → scale={optimal_scales[i]:.4f}")

print(f"\nRecommended OBS_SCALE:")
print("="*70)
print(f"OBS_SCALE = np.array([")
for i, name in enumerate(feature_names):
    print(f"    {optimal_scales[i]:.4f},  # {name}")
print(f"], dtype=np.float32)")

print(f"\nCompare to current (data-based) scaling:")
print(f"Current: [0.3664, 7.1769, 0.1396, 38.7333, 0.5000]")
print(f"Optimal: [{', '.join([f'{s:.4f}' for s in optimal_scales])}]")



