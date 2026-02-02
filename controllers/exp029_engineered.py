"""Controller with engineered temporal features (12D state)"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

V_NOMINAL = 30.0

class SimpleMLPNetwork(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class Controller:
    def __init__(self):
        checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp029_engineered_features" / "model.pth"
        
        self.network = SimpleMLPNetwork()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.network.eval()
        
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        
        v = max(state.v_ego, 1.0)
        v2 = max(v ** 2, 1.0)
        
        # Extract future curvatures
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
        curvs = np.array(curvs)
        
        # Engineer features
        immediate_curv = curvs[0:5].mean()
        immediate_peak = np.abs(curvs[0:5]).max()
        tactical_curv = curvs[5:20].mean()
        tactical_peak = np.abs(curvs[5:20]).max()
        strategic_curv = curvs[20:49].mean()
        curv_accel = (curvs[5] - curvs[0]) / 0.5 if len(curvs) > 5 else 0.0
        curv_smoothness = curvs[0:10].std()
        
        state_vec = np.array([
            error / v2,
            self.error_integral / v,
            error_diff,
            v / V_NOMINAL,
            self.prev_action,
            immediate_curv,
            immediate_peak,
            tactical_curv,
            tactical_peak,
            strategic_curv,
            curv_accel,
            curv_smoothness
        ], dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            action = self.network(state_tensor).item()
        
        self.prev_error = error
        self.prev_action = action
        
        return float(np.clip(action, -2.0, 2.0))



