"""
Analyze: Should we feed CURVATURE or RADIUS to the network?

Key insight: The control law should be as LINEAR as possible in the input features.

Question: Is steering action more linear with curvature or radius?

Physical intuition:
- Tight turn (small R, large curv) → large steering
- Gentle turn (large R, small curv) → small steering

Hypothesis: RADIUS might be more linear because:
- Steering angle ∝ 1/R (direct relationship)
- With curvature: steering ∝ curvature (also direct)

But there's a critical issue with radius:
- When curvature → 0 (straight road), radius → ∞
- This creates unbounded values!

Test: Which representation has more linear relationship to control output?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)

# Collect data
curvatures = []
radii = []
actions = []

for f in all_files[:100]:
    pid = PIDController()
    
    sim_data = pd.read_csv(f)
    state_obj = type('State', (), {})()
    future_obj = type('FuturePlan', (), {'lataccel': []})()
    
    for idx in range(100, min(500, len(sim_data))):
        row = sim_data.iloc[idx]
        state_obj.v_ego = row['vEgo']
        state_obj.a_ego = row['aEgo']
        state_obj.roll_lataccel = row['roll']
        
        target = row['targetLateralAcceleration']
        current = row['targetLateralAcceleration']
        
        # Future curvature at t+5 (0.5s ahead)
        if idx + 5 < len(sim_data):
            future_row = sim_data.iloc[idx + 5]
            future_lat = future_row['targetLateralAcceleration']
            future_roll = future_row['roll']
            future_v = future_row['vEgo']
            
            v2 = max(future_v ** 2, 1.0)
            curv = (future_lat - future_roll) / v2
            
            # Radius (avoid division by zero)
            if abs(curv) > 1e-6:
                radius = 1.0 / curv
            else:
                radius = 999.0  # Straight (cap at 999m)
            
            # Cap radius to reasonable range
            radius = np.clip(radius, -999, 999)
            
            action = pid.update(target, current, state_obj, future_obj)
            
            curvatures.append(curv)
            radii.append(radius)
            actions.append(action)

curvatures = np.array(curvatures)
radii = np.array(radii)
actions = np.array(actions)

print("="*70)
print("CURVATURE vs RADIUS Analysis")
print("="*70)

print(f"\nCurvature stats:")
print(f"  Mean: {curvatures.mean():.6f}")
print(f"  Std: {curvatures.std():.6f}")
print(f"  Range: [{curvatures.min():.6f}, {curvatures.max():.6f}]")
print(f"  |Curv| > 0.001: {(np.abs(curvatures) > 0.001).sum()} / {len(curvatures)}")

print(f"\nRadius stats:")
print(f"  Mean: {radii.mean():.2f}")
print(f"  Std: {radii.std():.2f}")
print(f"  Range: [{radii.min():.2f}, {radii.max():.2f}]")
print(f"  |Radius| > 100m: {(np.abs(radii) > 100).sum()} / {len(radii)}")
print(f"  |Radius| = 999m (straight): {(np.abs(radii) >= 998).sum()} / {len(radii)}")

print(f"\nAction stats:")
print(f"  Mean: {actions.mean():.6f}")
print(f"  Std: {actions.std():.6f}")
print(f"  Range: [{actions.min():.6f}, {actions.max():.6f}]")

# Correlation analysis
print("\n" + "="*70)
print("LINEARITY Analysis")
print("="*70)

# Remove straight-line cases for fair comparison
mask = np.abs(curvatures) > 1e-4
curv_filtered = curvatures[mask]
rad_filtered = radii[mask]
act_filtered = actions[mask]

print(f"\nFiltered to {len(curv_filtered)} samples with significant curvature")

from numpy.linalg import lstsq

# Curvature → action
a_curv, _ = lstsq(np.c_[np.ones(len(curv_filtered)), curv_filtered], act_filtered, rcond=None)[0]
pred_curv = a_curv[0] + a_curv[1] * curv_filtered
r2_curv = 1 - np.sum((act_filtered - pred_curv)**2) / np.sum((act_filtered - act_filtered.mean())**2)

print(f"\nCurvature → Action:")
print(f"  Linear fit: action = {a_curv[0]:.4f} + {a_curv[1]:.4f} × curvature")
print(f"  R² = {r2_curv:.4f}")

# Radius → action  
a_rad, _ = lstsq(np.c_[np.ones(len(rad_filtered)), rad_filtered], act_filtered, rcond=None)[0]
pred_rad = a_rad[0] + a_rad[1] * rad_filtered
r2_rad = 1 - np.sum((act_filtered - pred_rad)**2) / np.sum((act_filtered - act_filtered.mean())**2)

print(f"\nRadius → Action:")
print(f"  Linear fit: action = {a_rad[0]:.4f} + {a_rad[1]:.4f} × (1/radius)")
print(f"  R² = {r2_rad:.4f}")

# 1/Radius → action
inv_rad = 1.0 / rad_filtered
a_invrad, _ = lstsq(np.c_[np.ones(len(inv_rad)), inv_rad], act_filtered, rcond=None)[0]
pred_invrad = a_invrad[0] + a_invrad[1] * inv_rad
r2_invrad = 1 - np.sum((act_filtered - pred_invrad)**2) / np.sum((act_filtered - act_filtered.mean())**2)

print(f"\n1/Radius (=curvature) → Action:")
print(f"  Linear fit: action = {a_invrad[0]:.4f} + {a_invrad[1]:.4f} × (1/radius)")
print(f"  R² = {r2_invrad:.4f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if r2_curv > r2_rad:
    print(f"✓ CURVATURE is more linear (R²={r2_curv:.4f} vs {r2_rad:.4f})")
else:
    print(f"✓ RADIUS is more linear (R²={r2_rad:.4f} vs {r2_curv:.4f})")

print("\nPractical considerations:")
print("- Curvature: Bounded, no singularities, direct physical meaning")
print("- Radius: Can be very large (→∞), needs clipping, less intuitive scaling")
print("\nRecommendation: Use CURVATURE (current approach is correct)")

