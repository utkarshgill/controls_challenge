"""Debug: Check if collected data is correct"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from controllers.pid import Controller as PIDController

# Test: For simple inputs, verify our data matches PID equation
controller = PIDController()

# Simulate a few steps manually
test_cases = [
    # (target, current) -> error
    (1.0, 0.0),  # error = 1.0
    (1.0, 0.5),  # error = 0.5
    (0.5, 0.5),  # error = 0.0
]

print("Manual verification:")
print("PID: P=0.195, I=0.100, D=-0.053")
print()

for i, (target, current) in enumerate(test_cases):
    error = target - current
    controller.error_integral += error
    error_diff = error - controller.prev_error
    controller.prev_error = error
    
    # Compute expected action
    expected_action = (controller.p * error + 
                      controller.i * controller.error_integral + 
                      controller.d * error_diff)
    
    # State vector we'd store
    state_vec = [error, controller.error_integral, error_diff]
    
    # If we do linear regression: action = w[0]*error + w[1]*integral + w[2]*diff
    # We should recover: w[0]=0.195, w[1]=0.100, w[2]=-0.053
    
    print(f"Step {i}:")
    print(f"  error={error:.3f}, integral={controller.error_integral:.3f}, diff={error_diff:.3f}")
    print(f"  expected action={expected_action:.6f}")
    print(f"  verify: {0.195*error + 0.100*controller.error_integral + -0.053*error_diff:.6f}")
    print()

