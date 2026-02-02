"""
Controller for Experiment 032: Residual PPO

Architecture: u = u_PID + ε × π_θ(s_preview)

Where PID is frozen and provides feedback stability,
and the learned residual provides anticipatory corrections.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
ACTOR_LAYERS = 3

# Normalization scales
LATACCEL_SCALE = 3.0
V_SCALE = 40.0
A_SCALE = 3.0
CURVATURE_SCALE = 0.01

# Residual parameters
RESIDUAL_SCALE = 0.1
RESIDUAL_CLIP = 0.5
LOWPASS_ALPHA = 0.3


class PIDController:
    """Frozen PID baseline - matches controllers/pid.py exactly"""
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        output = self.p * error + self.i * self.error_integral + self.d * error_diff
        return float(np.clip(output, -2.0, 2.0))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        actor = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        critic = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(critic_layers - 1):
            critic.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        critic.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*critic)

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        value = self.critic(state)
        return action_mean, action_std, value


class Controller:
    def __init__(self):
        checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp032_residual_ppo" / "ppo_best.pth"
        
        if not checkpoint_path.exists():
            checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp032_residual_ppo" / "ppo_final.pth"
        
        # Initialize PID (frozen baseline)
        self.pid = PIDController()
        
        # Initialize learned residual network
        self.network = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, 3)
        
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.network.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Loaded residual PPO from {checkpoint_path.name} (cost: {ckpt.get('cost', 'N/A')})")
        else:
            print(f"⚠️  No checkpoint found, using PID only (ε=0)")
        
        self.network.eval()
        self.error_integral = 0.0
        self.prev_residual = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # === PID baseline (frozen) ===
        pid_action = self.pid.update(target_lataccel, current_lataccel)
        
        # === Learned residual ===
        # Construct state vector
        error = target_lataccel - current_lataccel
        self.error_integral += error * 0.1
        self.error_integral = np.clip(self.error_integral, -10, 10)
        
        # Future curvatures (properly normalized)
        future_curvatures = []
        for i in range(50):
            if i < len(future_plan.lataccel):
                future_lat = future_plan.lataccel[i]
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                curvature = (future_lat - future_roll) / (future_v ** 2 + 1e-6)
                curvature_normalized = curvature / CURVATURE_SCALE
            else:
                curvature_normalized = 0.0
            future_curvatures.append(curvature_normalized)
        
        state_vector = np.array([
            error / LATACCEL_SCALE,
            self.error_integral / LATACCEL_SCALE,
            state.v_ego / V_SCALE,
            state.a_ego / A_SCALE,
            state.roll_lataccel / LATACCEL_SCALE,
            *future_curvatures
        ], dtype=np.float32)
        
        # Get residual from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            raw_residual = action_mean.item()
        
        # Process residual: tanh → clip → filter → scale
        residual = np.tanh(raw_residual) * RESIDUAL_CLIP
        filtered_residual = LOWPASS_ALPHA * residual + (1 - LOWPASS_ALPHA) * self.prev_residual
        self.prev_residual = filtered_residual
        scaled_residual = filtered_residual * RESIDUAL_SCALE
        
        # === Combined action ===
        combined_action = pid_action + scaled_residual
        
        return float(np.clip(combined_action, -2.0, 2.0))

