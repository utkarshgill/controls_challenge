from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from tinyphysics import STEER_RANGE
import os

STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
TRUNK_LAYERS = 1
HEAD_LAYERS = 3

# Normalization
NORM_SCALE = np.array([
    2.0, 5.0, 0.05, 5.0, 2.0,
    *[2.0]*50
], dtype=np.float32)

def build_state(target_lataccel, current_lataccel, state, future_plan):
    """Build normalized 55D state"""
    error = target_lataccel - current_lataccel
    future_lataccel = np.array(future_plan.lataccel)
    
    if len(future_lataccel) == 0:
        future_lataccel = np.zeros(50, dtype=np.float32)
    elif len(future_lataccel) < 50:
        future_lataccel = np.pad(future_lataccel, (0, 50 - len(future_lataccel)), 'edge')
    else:
        future_lataccel = future_lataccel[:50]
    
    state_vec = np.array([
        error, state.roll_lataccel, state.v_ego, state.a_ego, current_lataccel,
        *future_lataccel
    ], dtype=np.float32)
    
    return state_vec * NORM_SCALE

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
        super().__init__()
        
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.critic_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value

class Controller(BaseController):
    """
    Pure PPO controller
    State: 55D normalized
    """
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, '../experiments/exp004_ppo_pure/results/checkpoints/ppo_best.pth')
        
        self.actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS)
        self.actor_critic.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.actor_critic.eval()
        print(f"✅ Loaded PPO weights from {checkpoint_path}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            action_mean, _, _ = self.actor_critic(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        
        return float(action.item())

