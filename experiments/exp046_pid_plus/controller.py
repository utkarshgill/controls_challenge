"""
Experiment 046: PID Plus
Starting point: Pure PID that works (cost ~103)
Strategy: Diagnose, then grow incrementally
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import BaseController, STEER_RANGE


class Controller(BaseController):
    """Pure PID baseline - copied from controllers/pid.py"""
    
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff
