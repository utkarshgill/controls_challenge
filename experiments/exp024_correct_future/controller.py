"""Controller with CORRECT future curvature calculation"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class ConvBCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, future_sequences):
        conv_out = self.conv(future_sequences)
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
        
        self.base_scale = ckpt['base_scale']
        self.curv_scale = ckpt['curv_scale']
        self.accel_scale = ckpt['accel_scale']
        
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        
        base = np.array([error, self.error_integral, error_diff, state.v_ego], dtype=np.float32)
        base_norm = torch.FloatTensor(base / self.base_scale).unsqueeze(0)
        
        # CORRECT future calculation
        curvs = []
        accels = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                future_lat = future_plan.lataccel[i]
                future_a = future_plan.a_ego[i] if i < len(future_plan.a_ego) else state.a_ego
                
                curv = (future_lat - future_roll) / max(future_v ** 2, 1.0)
                curvs.append(curv)
                accels.append(future_a)
            else:
                curvs.append(0.0)
                accels.append(0.0)
        
        future_seq = np.stack([
            np.array(curvs) / self.curv_scale,
            np.array(accels) / self.accel_scale
        ], axis=0).astype(np.float32)
        future_tensor = torch.FloatTensor(future_seq).unsqueeze(0)
        
        with torch.no_grad():
            action = self.network(base_norm, future_tensor).item()
        
        self.prev_error = error
        
        return float(np.clip(action, -2.0, 2.0))



