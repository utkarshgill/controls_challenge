"""
Experiment 47: PID + FF with exponentially decaying weights
Testing hypothesis: w[i] = 0.3 * exp(-i/10) should beat single-step FF
"""

import numpy as np
from pathlib import Path
from . import BaseController


class Controller(BaseController):
    """PID + FF with decaying weights"""
    
    def __init__(self):
        # Fixed PID
        self.p, self.i, self.d = 0.195, 0.100, -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Load decay weights
        weights_path = Path(__file__).parent.parent / 'experiments' / 'exp047_neural_atom' / 'decay_weights.npy'
        self.weights = np.load(weights_path)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # FF: exponentially decaying weights on future errors
        future_padded = list(future_plan.lataccel[:50]) + [target_lataccel] * (50 - len(future_plan.lataccel))
        future_errors = np.array([(f - target_lataccel) for f in future_padded[:50]], dtype=np.float32)
        ff_action = np.dot(self.weights, future_errors)
        
        return float(pid_action + ff_action)
