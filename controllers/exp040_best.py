"""
Exp040: PPO-tuned single neuron controller
Learned PID gains: P=0.194, I=0.109, D=-0.058
"""
import numpy as np
from . import BaseController


class Controller(BaseController):
    """Single neuron with 3 weights learned by PPO"""
    
    def __init__(self):
        # Best weights from PPO training (epoch 10)
        self.P = 0.194
        self.I = 0.109
        self.D = -0.058
        
        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute error
        error = target_lataccel - current_lataccel
        
        # Update integral
        self.error_integral += error
        
        # Compute derivative
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Linear combination (PID)
        action = self.P * error + self.I * self.error_integral + self.D * error_diff
        
        return action

