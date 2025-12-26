#!/usr/bin/env python3
"""
Check training data distribution

Question: Does training data contain low-speed scenarios?
"""

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

print("\n" + "="*60)
print("TRAINING DATA DISTRIBUTION ANALYSIS")
print("="*60)

# Sample training files
all_files = sorted(glob.glob("./data/*.csv"))
np.random.seed(42)
np.random.shuffle(all_files)
train_files = all_files[:int(0.9 * len(all_files))]
sample_files = train_files[:500]  # Sample 500 for speed

print(f"\nAnalyzing {len(sample_files)} training files...")

all_v_ego = []
all_target_lat = []
low_speed_files = []

for f in tqdm(sample_files, desc="Loading"):
    df = pd.read_csv(f)
    v_ego_vals = df['vEgo'].values
    all_v_ego.extend(v_ego_vals)
    all_target_lat.extend(df['targetLateralAcceleration'].values)
    
    # Check if this file has low speeds
    if v_ego_vals.min() < 1.0:
        low_speed_files.append((f, v_ego_vals.min(), v_ego_vals.max()))

all_v_ego = np.array(all_v_ego)
all_target_lat = np.array(all_target_lat)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nv_ego distribution (training data):")
print(f"  Mean: {np.mean(all_v_ego):.2f} m/s")
print(f"  Min:  {np.min(all_v_ego):.2f} m/s")
print(f"  Max:  {np.max(all_v_ego):.2f} m/s")
print(f"  Percentiles:")
print(f"    1%:  {np.percentile(all_v_ego, 1):.2f} m/s")
print(f"    5%:  {np.percentile(all_v_ego, 5):.2f} m/s")
print(f"    95%: {np.percentile(all_v_ego, 95):.2f} m/s")
print(f"    99%: {np.percentile(all_v_ego, 99):.2f} m/s")

print(f"\nLow-speed files (v_ego < 1.0 m/s):")
print(f"  Count: {len(low_speed_files)}/{len(sample_files)} ({100*len(low_speed_files)/len(sample_files):.1f}%)")

if len(low_speed_files) > 0:
    print(f"  Examples:")
    for f, vmin, vmax in low_speed_files[:5]:
        print(f"    {f}: v_ego ∈ [{vmin:.2f}, {vmax:.2f}]")

print(f"\ntarget_lataccel distribution:")
print(f"  Mean: {np.mean(all_target_lat):.2f}")
print(f"  Min:  {np.min(all_target_lat):.2f}")
print(f"  Max:  {np.max(all_target_lat):.2f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if len(low_speed_files) > 10:
    print("\n✅ Training data INCLUDES low-speed scenarios")
    print("   BC was trained on curvature explosions → learned to handle them")
    print("   Setting curv=0 at low speeds creates NEW distribution mismatch!")
else:
    print("\n❌ Training data has VERY FEW low-speed scenarios")
    print("   BC never learned to handle v_ego ≈ 0")
    print("   File 00069 is out-of-distribution!")

print("\n" + "="*60)
print("THE REAL QUESTION")
print("="*60)
print("\nIf BC clones PID perfectly on easy files (MAE=0.008)")
print("but fails on file 00069 (MAE=0.31), the issue is:")
print("\n  BC ACTIONS DIVERGE FROM PID DUE TO COMPOUNDING ERRORS")
print("\nPossible root causes:")
print("  1. BC network too small → can't capture PID perfectly")
print("  2. Training used MSE loss → small errors accumulate")
print("  3. BC sees slightly different states during rollout vs training")
print("  4. File 00069 is fundamentally OOD (hard even for PID: cost=375)")
print("="*60)

