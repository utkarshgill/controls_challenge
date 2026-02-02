"""
Experiment 034: 3-Parameter Linear Preview Controller

Learned weights from PPO training:
- w1 = -0.0967  (near term, 0.1-0.6s)
- w2 = -0.1624  (mid horizon, 1-2.5s)  
- w3 = -0.2173  (long term mass)
- bias = 0.2911
"""
import numpy as np
from . import BaseController

class Controller(BaseController):
    def __init__(self):
        # PID parameters (frozen baseline)
        self.pid_p = 0.195
        self.pid_i = 0.100
        self.pid_d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Learned linear weights
        self.w1 = -0.0967
        self.w2 = -0.1624
        self.w3 = -0.2173
        self.bias = 0.2911
        
        # Residual processing
        self.prev_residual = 0.0
        self.lowpass_alpha = 0.05
        self.residual_scale = 0.1
        self.residual_clip = 0.5
        self.lataccel_scale = 3.0
        
        self.step_count = 0
        self.warmup_steps = 50
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID component
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        pid_action = (self.pid_p * error + 
                     self.pid_i * self.error_integral + 
                     self.pid_d * error_diff)
        pid_action = np.clip(pid_action, -2.0, 2.0)
        
        # Residual component (disabled during warmup)
        if self.step_count < self.warmup_steps or future_plan is None:
            scaled_residual = 0.0
        else:
            # Compute 3 preview features
            future_lat = future_plan.lataccel
            if len(future_lat) > 0:
                baseline = future_lat[0]
                delta = np.array([lat - baseline for lat in future_lat])
                
                # f1: near term (0.1-0.6s = steps 1-6)
                f1 = np.mean(delta[1:6]) / self.lataccel_scale if len(delta) > 6 else 0.0
                
                # f2: mid horizon (1.0-2.5s = steps 10-25)
                f2 = np.mean(delta[10:25]) / self.lataccel_scale if len(delta) > 25 else 0.0
                
                # f3: long term signed mass
                f3 = np.mean(delta) / self.lataccel_scale
            else:
                f1, f2, f3 = 0.0, 0.0, 0.0
            
            # Linear combination
            raw_residual = self.w1 * f1 + self.w2 * f2 + self.w3 * f3 + self.bias
            
            # Process: tanh → clip → lowpass → scale
            residual = np.tanh(raw_residual) * self.residual_clip
            filtered = self.lowpass_alpha * residual + (1 - self.lowpass_alpha) * self.prev_residual
            self.prev_residual = filtered
            scaled_residual = filtered * self.residual_scale
        
        self.step_count += 1
        
        # Combined action
        combined = pid_action + scaled_residual
        return float(np.clip(combined, -2.0, 2.0))

