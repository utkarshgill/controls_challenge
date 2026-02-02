"""
Neural Controller v2: Single Neuron + Tanh

(Should match v1 since data is linear)
"""

import numpy as np
from . import BaseController
from pathlib import Path


class Controller(BaseController):
    """PID + Nonlinear Feedforward"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.prev_action = 0.0
        
        # Load learned params
        weights_path = Path(__file__).parent.parent / 'experiments' / 'exp046_pid_plus' / 'nonlinear_weights.npy'
        params = np.load(weights_path)
        self.w = params[:-1]
        self.b = params[-1]
        
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
            
            # Nonlinear prediction
            ff_action = np.tanh(self.w @ features + self.b)
        
        action = pid_action + ff_action
        self.prev_action = action
        return action
