"""
Controller for Experiment 031: Beautiful PPO

Uses the trained PPO policy from exp031_beautiful_ppo.
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        # Actor network
        actor = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network  
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
        checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp031_beautiful_ppo" / "ppo_best.pth"
        
        if not checkpoint_path.exists():
            checkpoint_path = Path(__file__).parent.parent / "experiments" / "exp031_beautiful_ppo" / "ppo_final.pth"
        
        self.network = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, 3)
        
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.network.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Loaded checkpoint from {checkpoint_path.name} (cost: {ckpt.get('cost', 'N/A')})")
        else:
            print(f"⚠️  No checkpoint found at {checkpoint_path}, using random initialization")
        
        self.network.eval()
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute error
        error = target_lataccel - current_lataccel
        
        # Update integral with anti-windup
        self.error_integral += error * 0.1
        self.error_integral = np.clip(self.error_integral, -10, 10)
        
        # Future curvatures (50 steps)
        future_curvatures = []
        for i in range(50):
            if i < len(future_plan.lataccel):
                future_lat = future_plan.lataccel[i]
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                curvature = (future_lat - future_roll) / (future_v ** 2 + 1e-6)
            else:
                curvature = 0.0
            future_curvatures.append(curvature)
        
        # Construct state vector (55D)
        state_vector = np.array([
            error / LATACCEL_SCALE,
            self.error_integral / LATACCEL_SCALE,
            state.v_ego / V_SCALE,
            state.a_ego / A_SCALE,
            state.roll_lataccel / LATACCEL_SCALE,
            *future_curvatures
        ], dtype=np.float32)
        
        # Get action from network (deterministic)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * 2.0  # Scale to [-2, 2]
            action = action.item()
        
        return float(np.clip(action, -2.0, 2.0))

