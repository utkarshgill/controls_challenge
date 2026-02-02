"""
Analyze what observation scaling makes the control problem most learnable.

Key insight from beautiful_lander: Different features need different scales
based on their RELATIONSHIP to the output, not just their statistical range.

For controls:
- Error → direct relationship to action (PID P term)
- Error_integral → accumulated, needs large scale
- Error_derivative → change signal, can be noisy
- v_ego → affects "time to maneuver"
- prev_action → jerk minimization

We want features that make: action = f(features) as LINEAR as possible
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:100]  # Sample 100 routes

# Collect PID behavior
data = {
    'error': [],
    'error_i': [],
    'error_d': [],
    'v_ego': [],
    'prev_action': [],
    'action': [],
    'future_curv_mean': [],  # Mean of next 10 curvatures
    'future_curv_max': [],   # Max of next 10 curvatures
}

for f in train_files[:100]:
    pid = PIDController()
    prev_action = 0.0
    
    sim_data = pd.read_csv(f)
    state_obj = type('State', (), {})()
    
    for idx in range(100, min(500, len(sim_data))):
        row = sim_data.iloc[idx]
        state_obj.v_ego = row['vEgo']
        state_obj.a_ego = row['aEgo']
        state_obj.roll_lataccel = row['roll']
        
        target = row['targetLateralAcceleration']
        current = row['targetLateralAcceleration']  # First step approximation
        
        # Future curvatures (next 10 steps)
        future_lats = sim_data['targetLateralAcceleration'].values[idx+1:idx+11]
        future_rolls = sim_data['roll'].values[idx+1:idx+11]
        future_vs = sim_data['vEgo'].values[idx+1:idx+11]
        
        if len(future_lats) > 0:
            future_curvs = [(lat - roll) / max(v**2, 1.0) 
                           for lat, roll, v in zip(future_lats, future_rolls, future_vs)]
            curv_mean = np.mean(np.abs(future_curvs))
            curv_max = np.max(np.abs(future_curvs))
        else:
            curv_mean, curv_max = 0.0, 0.0
        
        error = target - current
        action = pid.update(target, current, state_obj, type('FuturePlan', (), {'lataccel': []})())
        
        data['error'].append(error)
        data['error_i'].append(pid.error_integral)
        data['error_d'].append(error - pid.prev_error)
        data['v_ego'].append(state_obj.v_ego)
        data['prev_action'].append(prev_action)
        data['action'].append(action)
        data['future_curv_mean'].append(curv_mean)
        data['future_curv_max'].append(curv_max)
        
        prev_action = action

# Convert to arrays
for k in data:
    data[k] = np.array(data[k])

print("CORRELATION with action (what features predict output best?):")
print("="*60)
features = ['error', 'error_i', 'error_d', 'v_ego', 'prev_action', 'future_curv_mean', 'future_curv_max']
correlations = []
for feat in features:
    corr = np.corrcoef(data[feat], data['action'])[0, 1]
    correlations.append((feat, corr))
    print(f"{feat:20s}: {corr:6.3f}")

print("\nLINEARITY analysis (how linear is action = f(feature)?):")
print("="*60)
for feat in features:
    # Fit linear model: action = a * feature + b
    X = data[feat]
    y = data['action']
    
    # Remove NaNs
    mask = ~(np.isnan(X) | np.isnan(y))
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        continue
    
    # Linear fit
    a, b = np.polyfit(X, y, 1)
    y_pred = a * X + b
    
    # R² score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Residual stats
    residuals = y - y_pred
    res_std = np.std(residuals)
    
    print(f"{feat:20s}: R²={r2:.3f}, residual_std={res_std:.4f}, coef={a:.4f}")

print("\nOPTIMAL SCALING (to make features comparable magnitude):")
print("="*60)
print("Goal: Scale each feature so coefficient ≈ 1.0 for linear control law")
print()

# For each feature, compute scale that would make coef = 1.0
for feat in features:
    X = data[feat]
    y = data['action']
    mask = ~(np.isnan(X) | np.isnan(y))
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        continue
    
    a, b = np.polyfit(X, y, 1)
    
    # If we scale X by `scale`, new coef = a / scale
    # We want: a / scale ≈ 0.2 (reasonable control gain)
    target_gain = 0.2
    optimal_scale = abs(a / target_gain) if a != 0 else 1.0
    
    print(f"{feat:20s}: current_coef={a:7.4f} → optimal_scale={optimal_scale:7.4f}")

print("\nRECOMMENDED OBS_SCALE:")
print("="*60)
# Compute final recommendations
scales = {}
for feat in features:
    X = data[feat]
    y = data['action']
    mask = ~(np.isnan(X) | np.isnan(y))
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        scales[feat] = 1.0
        continue
    
    a, _ = np.polyfit(X, y, 1)
    optimal_scale = abs(a / 0.2) if abs(a) > 1e-6 else 1.0
    scales[feat] = optimal_scale

print(f"error:            {scales['error']:.4f}")
print(f"error_integral:   {scales['error_i']:.4f}")
print(f"error_derivative: {scales['error_d']:.4f}")
print(f"v_ego:            {scales['v_ego']:.4f}")
print(f"prev_action:      {scales['prev_action']:.4f}")
print(f"\nAs array:")
print(f"OBS_SCALE = np.array([{scales['error']:.4f}, {scales['error_i']:.4f}, {scales['error_d']:.4f}, {scales['v_ego']:.4f}, {scales['prev_action']:.4f}], dtype=np.float32)")



