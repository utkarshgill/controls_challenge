"""
Simplest possible improvement: PID + Feedforward

Just use future_plan[0] to anticipate next target
"""

import numpy as np
from . import BaseController


class Controller(BaseController):
    """PID + 1-step feedforward"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward gain
        self.ff_gain = 0.3
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward: anticipate next target
        if len(future_plan.lataccel) > 0:
            future_error = future_plan.lataccel[0] - target_lataccel
            ff_action = self.ff_gain * future_error
        else:
            ff_action = 0.0
        
        return pid_action + ff_action
