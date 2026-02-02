#!/usr/bin/env python3
"""Check if logged steerCommand matches PID or something better"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load first test file
data_file = Path("data/00000.csv")
df = pd.read_csv(data_file)

# Compute what PID would command
from controllers.pid import Controller as PIDController

pid = PIDController()

# Process data (same as simulator)
roll_lataccel = np.sin(df['roll'].values) * 9.81
target_lataccel = df['targetLateralAcceleration'].values

# Assume current_lataccel tracks target perfectly initially
current_lataccel = target_lataccel.copy()

pid_commands = []
for i in range(len(df)):
    target = target_lataccel[i]
    current = current_lataccel[i]
    
    # PID update
    pid_action = pid.update(target, current, None, None)
    pid_commands.append(pid_action)

pid_commands = np.array(pid_commands)
logged_commands = -df['steerCommand'].values  # Note the sign flip

# Compare
mse = np.mean((pid_commands - logged_commands)**2)
correlation = np.corrcoef(pid_commands, logged_commands)[0, 1]

print("Comparing logged steerCommand vs PID")
print("=" * 80)
print(f"MSE: {mse:.6f}")
print(f"Correlation: {correlation:.4f}")
print(f"\nLogged command stats:")
print(f"  Mean: {np.mean(logged_commands):.4f}")
print(f"  Std:  {np.std(logged_commands):.4f}")
print(f"  Range: [{np.min(logged_commands):.4f}, {np.max(logged_commands):.4f}]")
print(f"\nPID command stats:")
print(f"  Mean: {np.mean(pid_commands):.4f}")
print(f"  Std:  {np.std(pid_commands):.4f}")
print(f"  Range: [{np.min(pid_commands):.4f}, {np.max(pid_commands):.4f}]")

# Check a few timesteps
print(f"\nFirst 10 steps comparison:")
print(f"{'Step':<6} {'Logged':<10} {'PID':<10} {'Diff':<10}")
for i in range(10):
    diff = logged_commands[i] - pid_commands[i]
    print(f"{i:<6} {logged_commands[i]:<10.4f} {pid_commands[i]:<10.4f} {diff:<10.4f}")

if correlation > 0.95 and mse < 0.01:
    print("\n" + "=" * 80)
    print("✅ Logged commands are VERY SIMILAR to PID")
    print("BC on these logs = learning PID (explains why we hit ~101)")
elif correlation > 0.8:
    print("\n" + "=" * 80)
    print("⚠️  Logged commands are SIMILAR to PID but with differences")
    print("Might be PID with tuning or slight modifications")
else:
    print("\n" + "=" * 80)
    print("🎯 Logged commands are DIFFERENT from PID!")
    print("These logs might be from a better controller!")
    print("BC on these = learning from better teacher")

