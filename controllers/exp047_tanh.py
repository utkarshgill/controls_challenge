"""
Experiment 47: Single neuron with tanh activation
Learned to mimic PID behavior
"""

import numpy as np
from pathlib import Path
from . import BaseController


class Controller(BaseController):
    """Single neuron: tanh(w · [error, error_integral, error_diff, roll_lataccel] + b)"""
    
    def __init__(self):
        # Load learned parameters (trained on 2000 routes)
        params_path = Path(__file__).parent.parent / 'experiments' / 'exp047_neural_atom' / 'tanh_params_proper.npz'
        params = np.load(params_path)
        self.w = params['w']
        self.b = params['b']
        
        # State
        self.error_integral = 0
        self.prev_error = 0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute PID-like features
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Neural network forward pass
        features = np.array([error, self.error_integral, error_diff, state.roll_lataccel])
        action = np.tanh(self.w @ features + self.b)
        
        return float(action)
