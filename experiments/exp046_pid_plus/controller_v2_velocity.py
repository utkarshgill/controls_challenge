"""
v2: Velocity-Adaptive PID

Idea: At higher speeds, steering has more effect (higher lateral acceleration per degree).
We should scale our gains based on velocity.

Physics: lat_accel = v² / R (for circular motion)
So sensitivity to steering increases with v²

Changes from baseline:
- Scale P gain based on velocity
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import BaseController, STEER_RANGE


class Controller(BaseController):
    """Velocity-Adaptive PID"""
    
    def __init__(self):
        # Base PID params (tuned for ~33.6 m/s)
        self.p_base = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Reference velocity (mean from diagnostics)
        self.v_ref = 33.6
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Adapt P gain based on velocity
        # If v is higher than reference, we need less steering for same lat accel
        # P should scale as 1/v (roughly)
        v_ratio = self.v_ref / max(state.v_ego, 1.0)  # Avoid div by zero
        p = self.p_base * v_ratio
        
        # Standard PID with adaptive P
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        action = p * error + self.i * self.error_integral + self.d * error_diff
        return action
