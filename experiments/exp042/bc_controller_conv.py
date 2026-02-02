"""
Behavioral Cloning Controller with Conv1D
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from controllers import BaseController
from train_bc_conv import MLPConv


class BCControllerConv(BaseController):
    """Controller using Conv1D BC model"""
    def __init__(self, model_path):
        super().__init__()
        
        self.mlp = MLPConv()
        self.mlp.load_state_dict(torch.load(model_path))
        self.mlp.eval()
        
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def compute_features(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        v_ego_squared = state.v_ego ** 2
        if v_ego_squared > 0.01:
            current_curvature = (current_lataccel - state.roll_lataccel) / v_ego_squared
            target_curvature = (target_lataccel - state.roll_lataccel) / v_ego_squared
        else:
            current_curvature = 0.0
            target_curvature = 0.0
        
        future_curvatures = []
        for i in range(len(future_plan.lataccel)):
            v_future_squared = future_plan.v_ego[i] ** 2
            if v_future_squared > 0.01:
                future_curv = (future_plan.lataccel[i] - future_plan.roll_lataccel[i]) / v_future_squared
            else:
                future_curv = 0.0
            future_curvatures.append(future_curv)
        
        while len(future_curvatures) < 50:
            future_curvatures.append(0.0)
        future_curvatures = future_curvatures[:50]
        
        features = [error, self.error_integral, error_diff, current_curvature, target_curvature] + future_curvatures
        return np.array(features, dtype=np.float32)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = self.compute_features(target_lataccel, current_lataccel, state, future_plan)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.mlp(obs_tensor).item()
        
        return action
