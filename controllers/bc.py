from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from tinyphysics import STEER_RANGE

STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3

class BCNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x) * STEER_RANGE[1]

# Normalization constants (must match training!)
NORM_SCALE = np.array([
    2.0,   # error (-2 to 2)
    5.0,   # roll_lataccel (-5 to 5)
    0.05,  # v_ego (0 to 40 m/s)
    5.0,   # a_ego (-5 to 5)
    2.0,   # current_lataccel (-2 to 2)
    *[2.0]*50  # future_lataccel (-2 to 2 each)
], dtype=np.float32)

def build_state(target_lataccel, current_lataccel, state, future_plan):
    """Build 55-dim state vector with normalization"""
    error = target_lataccel - current_lataccel
    
    # Pad future plan if needed
    future_lataccel = np.array(future_plan.lataccel)
    if len(future_lataccel) == 0:
        future_lataccel = np.zeros(50, dtype=np.float32)
    elif len(future_lataccel) < 50:
        future_lataccel = np.pad(future_lataccel, (0, 50 - len(future_lataccel)), 'edge')
    else:
        future_lataccel = future_lataccel[:50]
    
    state_vec = np.array([
        error,
        state.roll_lataccel,
        state.v_ego,
        state.a_ego,
        current_lataccel,
        *future_lataccel
    ], dtype=np.float32)
    
    # Normalize (must match training!)
    state_vec = state_vec * NORM_SCALE
    
    return state_vec

class Controller(BaseController):
    """
    BC controller trained on PID data
    State: 55D = [error, roll, v_ego, a_ego, current, future×50]
    """
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            import os
            # Find checkpoint relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, '../experiments/exp003_ppo_bc_init/results/checkpoints/bc_best.pth')
        
        self.network = BCNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS)
        self.network.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.network.eval()
        print(f"✅ Loaded BC weights from {checkpoint_path}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            action = self.network(state_tensor)
        
        return float(action.item())

