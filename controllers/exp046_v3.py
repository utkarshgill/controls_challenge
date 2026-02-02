"""
v3: PID + Turn Anticipation

Look at derivative of future targets to detect turns coming
If target is increasing → turn coming → act early
"""

import numpy as np
from . import BaseController


class Controller(BaseController):
    """PID + Turn derivative feedforward"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward params
        self.ff_immediate = 0.3  # Weight for next step (like v1)
        self.ff_derivative = 0.15  # Weight for rate of change
        
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
            # Immediate: next step error (like v1)
            immediate_error = future_plan.lataccel[0] - target_lataccel
            ff_action += self.ff_immediate * immediate_error
            
            # Derivative: rate of change over next 5 steps
            # If targets are [0.5, 0.6, 0.7, 0.8, 0.9], derivative ≈ 0.1
            # This tells us a turn is coming
            future_slice = future_plan.lataccel[:5]
            target_derivative = (future_slice[-1] - future_slice[0]) / 5.0
            ff_action += self.ff_derivative * target_derivative
        
        return pid_action + ff_action
