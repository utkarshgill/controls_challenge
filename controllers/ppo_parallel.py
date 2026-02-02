"""
PPO Parallel Controller - For evaluation with evaluate.py
Loads weights from ppo_parallel_best.pth
"""

import numpy as np
import torch
import torch.nn as nn
from controllers import BaseController

# State normalization (must match training)
OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] +  # [error, error_diff, error_integral, lataccel, v_ego, curv]
    [1000.0] * 50,
    dtype=np.float32
)

STEER_RANGE = [-2, 2]

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """56D: PID terms (3) + current state (3) + 50 future curvatures"""
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self, state_dim=56, action_dim=1, hidden_dim=128, trunk_layers=1, head_layers=3):
        super(ActorCritic, self).__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.05))
        
        # Critic head (not used for inference)
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
    
    @torch.no_grad()
    def act(self, state, deterministic=True):
        """Deterministic action for evaluation"""
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).unsqueeze(0)
        action_mean, action_std, _ = self(state_tensor)
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action.squeeze().cpu().numpy()

class Controller(BaseController):
    """
    PPO Parallel Controller
    Loads trained weights and performs inference
    """
    # Class-level cache to avoid reloading model 100x
    _shared_model = None
    _model_loaded = False
    
    def __init__(self):
        # Load model once and share across all instances
        if not Controller._model_loaded:
            Controller._shared_model = ActorCritic()
            try:
                state_dict = torch.load('ppo_parallel_best.pth', map_location='cpu')
                Controller._shared_model.load_state_dict(state_dict)
                print("✅ Loaded PPO weights from ppo_parallel_best.pth")
            except FileNotFoundError:
                print("⚠️  ppo_parallel_best.pth not found, using random weights")
            Controller._shared_model.eval()
            Controller._model_loaded = True
        
        self.actor_critic = Controller._shared_model
        
        # State tracking (per instance)
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Compute steering action"""
        # Build state vector
        error = target_lataccel - current_lataccel
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                               self.prev_error, self.error_integral)
        
        # Get action from policy (deterministic for evaluation)
        action = self.actor_critic.act(state_vec, deterministic=True)
        
        # Update state tracking with anti-windup (±14 = 99.9% training coverage)
        self.error_integral = np.clip(self.error_integral + error, -14, 14)
        self.prev_error = error
        
        return float(action)
