#!/usr/bin/env python3
"""Check if training data has directional bias"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.pid import Controller as PIDController

def main():
    model_path = "../../models/tinyphysics.onnx"
    files = sorted(glob.glob("../../data/*.csv"))[:100]  # First 100 files
    
    print("="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(files)} files with PID controller...")
    
    all_actions = []
    all_targets = []
    all_errors = []
    
    for file_path in tqdm(files, desc="Collecting"):
        model = TinyPhysicsModel(model_path, debug=False)
        pid = PIDController()
        sim = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
        sim.rollout()
        
        # Get control phase only (steps 100-500)
        actions = sim.action_history[100:min(500, len(sim.action_history))]
        targets = sim.target_lataccel_history[100:min(500, len(sim.target_lataccel_history))]
        currents = sim.current_lataccel_history[100:min(500, len(sim.current_lataccel_history))]
        errors = [t - c for t, c in zip(targets, currents)]
        
        all_actions.extend(actions)
        all_targets.extend(targets)
        all_errors.extend(errors)
    
    all_actions = np.array(all_actions)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nPID Actions (n={len(all_actions):,}):")
    print(f"  Mean:      {all_actions.mean():8.4f}")
    print(f"  Median:    {np.median(all_actions):8.4f}")
    print(f"  Std:       {all_actions.std():8.4f}")
    print(f"  Min:       {all_actions.min():8.4f}")
    print(f"  Max:       {all_actions.max():8.4f}")
    print(f"  % positive: {100 * (all_actions > 0).mean():.1f}%")
    print(f"  % negative: {100 * (all_actions < 0).mean():.1f}%")
    
    print(f"\nTarget Lateral Accel:")
    print(f"  Mean:      {all_targets.mean():8.4f}")
    print(f"  Median:    {np.median(all_targets):8.4f}")
    print(f"  Std:       {all_targets.std():8.4f}")
    print(f"  Min:       {all_targets.min():8.4f}")
    print(f"  Max:       {all_targets.max():8.4f}")
    print(f"  % positive: {100 * (all_targets > 0).mean():.1f}%")
    print(f"  % negative: {100 * (all_targets < 0).mean():.1f}%")
    
    print(f"\nTracking Errors:")
    print(f"  Mean:      {all_errors.mean():8.4f}")
    print(f"  Median:    {np.median(all_errors):8.4f}")
    print(f"  Std:       {all_errors.std():8.4f}")
    print(f"  % positive: {100 * (np.array(all_errors) > 0).mean():.1f}%")
    print(f"  % negative: {100 * (np.array(all_errors) < 0).mean():.1f}%")
    
    # Check for systematic bias
    print("\n" + "="*80)
    print("BIAS DETECTION")
    print("="*80)
    
    if abs(all_actions.mean()) > 0.01:
        print(f"\n⚠️  BIAS DETECTED in actions!")
        print(f"   Mean action: {all_actions.mean():.4f}")
        if all_actions.mean() > 0:
            print("   → Systematic RIGHT bias in data")
        else:
            print("   → Systematic LEFT bias in data")
    else:
        print("\n✅ No systematic bias in actions (mean ≈ 0)")
    
    if abs(all_targets.mean()) > 0.01:
        print(f"\n⚠️  BIAS DETECTED in targets!")
        print(f"   Mean target: {all_targets.mean():.4f}")
    else:
        print("\n✅ No systematic bias in targets (mean ≈ 0)")
    
    # Distribution plot info
    print("\n" + "="*80)
    print("DISTRIBUTION SUMMARY")
    print("="*80)
    
    action_hist, bins = np.histogram(all_actions, bins=20, range=(-2, 2))
    print("\nAction distribution (binned):")
    for i in range(len(action_hist)):
        bar = '█' * int(50 * action_hist[i] / action_hist.max())
        print(f"  [{bins[i]:5.2f}, {bins[i+1]:5.2f}): {bar} {action_hist[i]:,}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

