"""Controller with vehicle-centric state (like LunarLander)"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class VehicleStateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.future_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.net = nn.Sequential(
            nn.Linear(6 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, vehicle_state, future_trajectory):
        future_features = self.future_conv(future_trajectory)
        future_flat = future_features.reshape(future_features.size(0), -1)
        combined = torch.cat([vehicle_state, future_flat], dim=1)
        return self.net(combined)

class Controller:
    def __init__(self):
        checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp030_vehicle_state" / "model.pth"
        
        self.network = VehicleStateNetwork()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.network.eval()
        
        self.lataccel_scale = ckpt['lataccel_scale']
        self.v_scale = ckpt['v_scale']
        self.a_scale = ckpt['a_scale']
        self.action_scale = ckpt['action_scale']
        
        self.prev_action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # VEHICLE STATE (not PID constructs!)
        vehicle_state = np.array([
            current_lataccel,
            target_lataccel,
            state.v_ego,
            state.a_ego,
            state.roll_lataccel,
            self.prev_action
        ], dtype=np.float32)
        
        # Normalize
        scales = np.array([self.lataccel_scale, self.lataccel_scale, self.v_scale, 
                          self.a_scale, self.lataccel_scale, self.action_scale])
        vehicle_state = vehicle_state / scales
        vehicle_tensor = torch.FloatTensor(vehicle_state).unsqueeze(0)
        
        # Future trajectory
        future_lataccels = []
        future_v_egos = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_lataccels.append(future_plan.lataccel[i])
                future_v_egos.append(future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego)
            else:
                future_lataccels.append(target_lataccel)
                future_v_egos.append(state.v_ego)
        
        future_traj = np.stack([future_lataccels, future_v_egos], axis=0).astype(np.float32)
        future_traj[0, :] = future_traj[0, :] / self.lataccel_scale
        future_traj[1, :] = future_traj[1, :] / self.v_scale
        future_tensor = torch.FloatTensor(future_traj).unsqueeze(0)
        
        with torch.no_grad():
            action = self.network(vehicle_tensor, future_tensor).item()
        
        self.prev_action = action
        
        return float(np.clip(action, -2.0, 2.0))



