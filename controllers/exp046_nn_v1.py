"""
Neural Controller v1: Single Linear Neuron

Learned weights from data (should match v3 performance)
"""

import numpy as np
from . import BaseController
from pathlib import Path


class Controller(BaseController):
    """PID + Learned Linear Feedforward"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.prev_action = 0.0
        
        # Load learned weights
        weights_path = Path(__file__).parent.parent / 'experiments' / 'exp046_pid_plus' / 'linear_weights.npy'
        self.w = np.load(weights_path)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Neural feedforward
        ff_action = 0.0
        
        if len(future_plan.lataccel) >= 5:
            # Extract features (same as training)
            immediate_error = future_plan.lataccel[0] - target_lataccel
            future_slice = future_plan.lataccel[:5]
            target_derivative = (future_slice[-1] - future_slice[0]) / 5.0
            
            features = np.array([
                immediate_error,
                target_derivative,
                error,
                self.prev_action,
                state.v_ego / 34.0,
                state.roll_lataccel
            ])
            
            # Linear prediction
            ff_action = self.w @ features
        
        action = pid_action + ff_action
        self.prev_action = action
        return action
