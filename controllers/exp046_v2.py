"""
v2: PID + Multi-step Feedforward

Instead of just future[0], look at next 5-10 steps
Compute weighted average (closer steps matter more)
"""

import numpy as np
from . import BaseController


class Controller(BaseController):
    """PID + Weighted future feedforward"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward params
        self.ff_gain = 0.4
        self.lookahead = 10  # Look 10 steps (1 second) ahead
        
        # Exponential decay weights (closer = more important)
        # weights = [1.0, 0.8, 0.64, 0.51, 0.41, ...]
        self.decay = 0.8
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward: weighted average of future targets
        if len(future_plan.lataccel) > 0:
            # How many steps to look ahead
            n = min(self.lookahead, len(future_plan.lataccel))
            
            # Compute weights (exponential decay)
            weights = np.array([self.decay ** i for i in range(n)])
            weights /= weights.sum()  # Normalize
            
            # Weighted average of future targets
            future_targets = np.array(future_plan.lataccel[:n])
            predicted_trend = np.dot(weights, future_targets)
            
            # Feedforward based on difference from current target
            future_error = predicted_trend - target_lataccel
            ff_action = self.ff_gain * future_error
        else:
            ff_action = 0.0
        
        return pid_action + ff_action
