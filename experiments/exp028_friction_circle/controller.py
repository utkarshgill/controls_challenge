"""Controller with friction circle awareness"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

V_NOMINAL = 30.0
MU_G = 9.8

class ConvBCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.mlp = nn.Sequential(
            nn.Linear(7 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, curv_sequence):
        curv_input = curv_sequence.unsqueeze(1)
        conv_out = self.conv(curv_input)
        conv_flat = conv_out.reshape(conv_out.size(0), -1)
        combined = torch.cat([base_features, conv_flat], dim=1)
        return self.mlp(combined)

class Controller:
    def __init__(self):
        checkpoint_path = Path(__file__).parent / "model.pth"
        
        self.network = ConvBCNetwork()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.network.eval()
        
        self.curv_scale = ckpt['curv_scale']
        
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        
        v = max(state.v_ego, 1.0)
        
        # Friction circle
        a_long = state.a_ego
        friction_used_long = min(abs(a_long) / MU_G, 0.99)
        available_lat_fraction = np.sqrt(1.0 - friction_used_long ** 2)
        available_lat_accel = available_lat_fraction * MU_G
        
        error_friction_norm = error / max(available_lat_accel, 1.0)
        
        total_accel = np.sqrt(current_lataccel ** 2 + a_long ** 2)
        friction_utilization = total_accel / MU_G
        
        base = np.array([
            error_friction_norm,
            self.error_integral / v,
            error_diff,
            v / V_NOMINAL,
            friction_utilization,
            available_lat_fraction,
            self.prev_action
        ], dtype=np.float32)
        base_tensor = torch.FloatTensor(base).unsqueeze(0)
        
        # Future curvatures
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_v2 = max(future_v ** 2, 1.0)
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                lat = future_plan.lataccel[i]
                curv = (lat - future_roll) / future_v2
                curvs.append(curv)
            else:
                curvs.append(0.0)
        
        curv_tensor = torch.FloatTensor(curvs).unsqueeze(0) / self.curv_scale
        
        with torch.no_grad():
            action = self.network(base_tensor, curv_tensor).item()
        
        self.prev_error = error
        self.prev_action = action
        
        return float(np.clip(action, -2.0, 2.0))



