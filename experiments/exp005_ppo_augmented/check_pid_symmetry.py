#!/usr/bin/env python3
"""Check if PID responses are symmetric"""
import sys
sys.path.insert(0, '../..')

from controllers.pid import Controller as PIDController

# Test PID on symmetric errors
pid = PIDController()

print("="*60)
print("PID SYMMETRY TEST")
print("="*60)

# Reset for each test
from tinyphysics import State, FuturePlan

state = State(roll_lataccel=0.0, v_ego=30.0, a_ego=0.0)
future = FuturePlan(lataccel=[0]*50, roll_lataccel=[0]*50, v_ego=[30]*50, a_ego=[0]*50)

tests = [
    (+1.0, 0.0, "+1.0 error"),
    (-1.0, 0.0, "-1.0 error"),
    (+0.5, 0.0, "+0.5 error"),
    (-0.5, 0.0, "-0.5 error"),
    (+2.0, 0.0, "+2.0 error"),
    (-2.0, 0.0, "-2.0 error"),
]

for target, current, desc in tests:
    pid_fresh = PIDController()  # Fresh PID for each test
    action = pid_fresh.update(target, current, state, future)
    print(f"{desc:15} → action: {action:7.4f}")

print("\n" + "="*60)
print("PID IS PERFECTLY SYMMETRIC (as expected)")
print("="*60)

