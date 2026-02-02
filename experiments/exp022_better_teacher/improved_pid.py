"""
Improved PID with velocity-dependent gains
Physics: At high speed, steering more sensitive → lower gain needed
"""
import numpy as np

class Controller:
    def __init__(self):
        # Base PID gains (from original)
        self.P_base = 0.195
        self.I_base = 0.100
        self.D_base = -0.053
        
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Velocity-dependent scaling
        # At low speed (v=0): gain = 1.0
        # At high speed (v=40 m/s): gain = 0.5
        # Physics: steering effectiveness ∝ v²
        v = state.v_ego
        velocity_scale = 1.0 / (1.0 + 0.0125 * v)  # Decreases with speed
        
        # Apply scaled gains
        P = self.P_base * velocity_scale
        I = self.I_base * velocity_scale  
        D = self.D_base * velocity_scale
        
        action = P * error + I * self.error_integral + D * error_diff
        
        return float(np.clip(action, -2.0, 2.0))



