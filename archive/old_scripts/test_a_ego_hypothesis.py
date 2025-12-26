#!/usr/bin/env python3
"""
TEST: Does a_ego matter?

Hypothesis: On files with high |a_ego|, the friction circle coupling matters.
File 00069 has lots of braking/acceleration → BC needs a_ego to predict correctly.
"""

import numpy as np
import pandas as pd

print("\n" + "="*60)
print("TESTING a_ego HYPOTHESIS")
print("="*60)

# Check what a_ego looks like on easy vs hard files
print("\n[1] Easy file (00000):")
df_easy = pd.read_csv("./data/00000.csv")
print(f"  v_ego: [{df_easy['vEgo'].min():.2f}, {df_easy['vEgo'].max():.2f}]")
print(f"  a_ego: [{df_easy['aEgo'].min():.2f}, {df_easy['aEgo'].max():.2f}]")
print(f"  |a_ego| mean: {np.mean(np.abs(df_easy['aEgo'])):.2f}")
print(f"  |a_ego| max:  {np.max(np.abs(df_easy['aEgo'])):.2f}")

print("\n[2] Hard file (00069):")
df_hard = pd.read_csv("./data/00069.csv")
print(f"  v_ego: [{df_hard['vEgo'].min():.2f}, {df_hard['vEgo'].max():.2f}]")
print(f"  a_ego: [{df_hard['aEgo'].min():.2f}, {df_hard['aEgo'].max():.2f}]")
print(f"  |a_ego| mean: {np.mean(np.abs(df_hard['aEgo'])):.2f}")
print(f"  |a_ego| max:  {np.max(np.abs(df_hard['aEgo'])):.2f}")

# Compute friction circle usage
def friction_usage(lat_accel, long_accel, mu_g=9.8):
    """How much of friction circle is used (0=none, 1=at limit)"""
    return np.sqrt(lat_accel**2 + long_accel**2) / mu_g

easy_friction = friction_usage(df_easy['targetLateralAcceleration'].values, 
                               df_easy['aEgo'].values)
hard_friction = friction_usage(df_hard['targetLateralAcceleration'].values,
                               df_hard['aEgo'].values)

print("\n[3] Friction circle usage:")
print(f"  Easy file: mean={np.mean(easy_friction):.3f}, max={np.max(easy_friction):.3f}")
print(f"  Hard file: mean={np.mean(hard_friction):.3f}, max={np.max(hard_friction):.3f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if np.mean(np.abs(df_hard['aEgo'])) > 2 * np.mean(np.abs(df_easy['aEgo'])):
    print("\n✅ HYPOTHESIS SUPPORTED!")
    print("   Hard file has 2× more longitudinal acceleration")
    print("   Friction coupling matters significantly")
    print("\n   Without a_ego in state:")
    print("   → BC sees same lateral demand")
    print("   → But doesn't know friction is limited by braking")
    print("   → Commands too-aggressive steering")
    print("   → Cost explodes")
else:
    print("\n❌ HYPOTHESIS WEAK")
    print("   a_ego doesn't differ much between files")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("\n1. Check if BC was trained WITH a_ego:")
print("   → Look at train_bc_pid.py line 77")
print("   → Current: a_ego was REMOVED")
print("\n2. Retrain BC WITH a_ego:")
print("   → Add a_ego back to state (dim 56 → 57)")
print("   → Collect new expert data")
print("   → Train new BC")
print("\n3. Test if it fixes file 00069")
print("="*60)

