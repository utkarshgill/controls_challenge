"""
v1: PID + Roll Compensation

Idea: Road roll (banking) contributes to lateral acceleration.
Account for this in our control.

Changes from baseline:
- Add feedforward term based on state.roll_lataccel
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import BaseController, STEER_RANGE


class Controller(BaseController):
    """PID + Roll Compensation"""
    
    def __init__(self):
        # Standard PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # NEW: Roll compensation gain
        # Hypothesis: If road is banked right (+roll), we need less right steering
        self.roll_gain = 0.15  # Start conservative
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # NEW: Roll feedforward
        # If roll is providing +0.3 lat accel, we can reduce our steering slightly
        roll_compensation = -self.roll_gain * state.roll_lataccel
        
        action = pid_action + roll_compensation
        return action
