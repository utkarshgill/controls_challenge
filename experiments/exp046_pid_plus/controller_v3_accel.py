"""
v3: Acceleration Feedforward

Idea: a_ego tells us if we're speeding up/slowing down.
This affects how steering translates to lateral acceleration.

Changes from baseline:
- Add small correction based on a_ego
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import BaseController, STEER_RANGE


class Controller(BaseController):
    """PID + Accel Feedforward"""
    
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Small gain for acceleration effect
        self.accel_gain = 0.05
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Accel correction (if accelerating, steering is slightly less effective)
        accel_correction = -self.accel_gain * state.a_ego
        
        action = pid_action + accel_correction
        return action
