"""
Experiment 47: PID + learned feedforward using ALL 50 future lataccels
Single linear neuron with 50 weights learned from exp046_v3's FF actions
"""

import numpy as np
from pathlib import Path
from . import BaseController


class Controller(BaseController):
    """PID + linear neuron trained on all 50 future lataccels"""
    
    def __init__(self):
        # Fixed PID parameters
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Load learned weights
        weights_path = Path(__file__).parent.parent / 'experiments' / 'exp047_neural_atom' / 'full_future_weights.npy'
        self.weights = np.load(weights_path)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID (fixed)
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward: linear combination of (future_lataccel[i] - target_lataccel)
        future_padded = list(future_plan.lataccel[:50]) + [target_lataccel] * (50 - len(future_plan.lataccel))
        features = np.array([(f - target_lataccel) for f in future_padded[:50]], dtype=np.float32)
        ff_action = np.dot(self.weights, features)
        
        return float(pid_action + ff_action)
