"""
Experiment 037: NNFF-Style Temporal Probes Controller

7 temporal probes (learned from PPO):
- Δlat @ 0.3s, 0.6s, 1.0s, 1.5s
- Δroll @ 0.3s, 1.0s
- v_ego

From checkpoint epoch 20.
"""
import numpy as np
from . import BaseController

class Controller(BaseController):
    def __init__(self):
        # PID baseline (frozen)
        self.pid_p = 0.195
        self.pid_i = 0.100
        self.pid_d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Learned linear weights (from checkpoint epoch 20)
        self.weights = np.array([0.076908, -0.087718, -0.018120, -0.012752, -0.056945, -0.237352, 0.360987])
        self.bias = 0.131716
        
        # Residual processing
        self.prev_residual = 0.0
        self.lowpass_alpha = 0.05
        self.residual_scale = 0.1
        self.residual_clip = 0.5
        
        # Scales (from training)
        self.lataccel_scale = 3.0
        self.v_scale = 40.0
        
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
        
        # Residual component (disabled during warmup or if no future_plan)
        if self.step_count < self.warmup_steps or future_plan is None:
            scaled_residual = 0.0
        else:
            # Extract 7 temporal probes
            lat = future_plan.lataccel
            roll = future_plan.roll_lataccel
            
            if len(lat) >= 16 and len(roll) >= 11:
                delta_lat_03 = (lat[3] - target_lataccel) / self.lataccel_scale
                delta_lat_06 = (lat[6] - target_lataccel) / self.lataccel_scale
                delta_lat_10 = (lat[10] - target_lataccel) / self.lataccel_scale
                delta_lat_15 = (lat[15] - target_lataccel) / self.lataccel_scale
                
                delta_roll_03 = (roll[3] - state.roll_lataccel) / self.lataccel_scale
                delta_roll_10 = (roll[10] - state.roll_lataccel) / self.lataccel_scale
                
                v_ego_norm = state.v_ego / self.v_scale
                
                # Linear combination
                raw_residual = (self.weights[0] * delta_lat_03 +
                               self.weights[1] * delta_lat_06 +
                               self.weights[2] * delta_lat_10 +
                               self.weights[3] * delta_lat_15 +
                               self.weights[4] * delta_roll_03 +
                               self.weights[5] * delta_roll_10 +
                               self.weights[6] * v_ego_norm +
                               self.bias)
                
                # Process: tanh → clip → lowpass → scale
                residual = np.tanh(raw_residual) * self.residual_clip
                filtered = self.lowpass_alpha * residual + (1 - self.lowpass_alpha) * self.prev_residual
                self.prev_residual = filtered
                scaled_residual = filtered * self.residual_scale
            else:
                scaled_residual = 0.0
        
        self.step_count += 1
        
        # Combined action
        combined = pid_action + scaled_residual
        return float(np.clip(combined, -2.0, 2.0))

