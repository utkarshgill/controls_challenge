"""Controller with action history for jerk minimization"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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
            nn.Linear(5 + 16*8, 128),
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
        
        self.base_scale = ckpt['base_scale']
        self.curv_scale = ckpt['curv_scale']
        
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_action = 0.0  # Track for jerk minimization
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        
        # Include previous action in state
        base = np.array([error, self.error_integral, error_diff, state.v_ego, self.prev_action], dtype=np.float32)
        base_norm = torch.FloatTensor(base / self.base_scale).unsqueeze(0)
        
        # Future curvatures
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                lat = future_plan.lataccel[i]
                curv = (lat - future_roll) / max(future_v ** 2, 1.0)
                curvs.append(curv)
            else:
                curvs.append(0.0)
        curv_seq = torch.FloatTensor(curvs).unsqueeze(0) / self.curv_scale
        
        with torch.no_grad():
            action = self.network(base_norm, curv_seq).item()
        
        self.prev_error = error
        self.prev_action = action  # Update for next step
        
        return float(np.clip(action, -2.0, 2.0))



