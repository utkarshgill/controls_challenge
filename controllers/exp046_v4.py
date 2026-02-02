"""
v4: Tuned gains

v3 worked, let's try slightly stronger derivative term
"""

import numpy as np
from . import BaseController


class Controller(BaseController):
    """PID + Tuned turn anticipation"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward params (increased derivative weight)
        self.ff_immediate = 0.35  # Slightly higher
        self.ff_derivative = 0.20  # Increased from 0.15
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward
        ff_action = 0.0
        
        if len(future_plan.lataccel) >= 5:
            # Immediate: next step error
            immediate_error = future_plan.lataccel[0] - target_lataccel
            ff_action += self.ff_immediate * immediate_error
            
            # Derivative: rate of change over next 5 steps
            future_slice = future_plan.lataccel[:5]
            target_derivative = (future_slice[-1] - future_slice[0]) / 5.0
            ff_action += self.ff_derivative * target_derivative
        
        return pid_action + ff_action
