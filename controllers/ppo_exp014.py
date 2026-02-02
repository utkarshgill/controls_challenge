"""
PPO Controller from exp014 (fine-tuned from BC)
Can be evaluated using: python tinyphysics.py --controller ppo_exp014 --data_path ./data --num_segs 100
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Hyperparameters (must match training)
STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):
    """Actor-Critic network (same as training)"""
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        
        # Shared trunk
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.trunk = nn.Sequential(*layers)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1], scale to [-2, 2]
        )
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.1))
        
        # Critic head (not used in evaluation, but needed for loading)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Returns action_mean, action_std, value"""
        features = self.trunk(state)
        action_mean = self.actor_head(features) * 2.0  # Scale to [-2, 2]
        action_std = self.log_std.exp()
        value = self.critic_head(features)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            return action


class Controller:
    """
    PPO Controller - official interface for tinyphysics.py
    NOTE: A new instance is created for each rollout
    """
    def __init__(self):
        # Load checkpoint
        checkpoint_path = Path(__file__).parent.parent / "experiments/exp014_ppo_finetune/ppo_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load network
        self.network = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        
        # Load normalization stats
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        
        # Internal state (reset for each new rollout)
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Controller interface (called by tinyphysics simulator)
        
        Args:
            target_lataccel: Target lateral acceleration
            current_lataccel: Current lateral acceleration
            state: State namedtuple (roll_lataccel, v_ego, a_ego)
            future_plan: FuturePlan namedtuple (lataccel, roll_lataccel, v_ego, a_ego)
        
        Returns:
            Steering command in [-2, 2]
        """
        # Compute error
        error = target_lataccel - current_lataccel
        self.error_integral += error
        
        # Get state variables
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel
        
        # Calculate curvatures from future plan
        future_v_egos = np.array(future_plan.v_ego, dtype=np.float32)
        if len(future_v_egos) < 50:
            pad_mode = 'constant' if len(future_v_egos) == 0 else 'edge'
            future_v_egos = np.pad(future_v_egos, (0, 50 - len(future_v_egos)), mode=pad_mode)
        else:
            future_v_egos = future_v_egos[:50]
        
        future_lataccels = np.array(future_plan.lataccel, dtype=np.float32)
        if len(future_lataccels) < 50:
            pad_mode = 'constant' if len(future_lataccels) == 0 else 'edge'
            future_lataccels = np.pad(future_lataccels, (0, 50 - len(future_lataccels)), mode=pad_mode)
        else:
            future_lataccels = future_lataccels[:50]
        
        future_roll = np.array(future_plan.roll_lataccel, dtype=np.float32)
        if len(future_roll) < 50:
            pad_mode = 'constant' if len(future_roll) == 0 else 'edge'
            future_roll = np.pad(future_roll, (0, 50 - len(future_roll)), mode=pad_mode)
        else:
            future_roll = future_roll[:50]
        
        # Curvature calculation
        v_ego_sq = np.maximum(future_v_egos ** 2, 1.0)
        curvatures = (future_lataccels - future_roll) / v_ego_sq
        
        # Build state vector [error, error_integral, v_ego, a_ego, roll_lataccel, curvatures[50]]
        state_vec = np.array([
            error,
            self.error_integral,
            v_ego,
            a_ego,
            roll_lataccel,
            *curvatures
        ], dtype=np.float32)
        
        # Normalize
        state_norm = (state_vec - self.state_mean) / self.state_std
        
        # Get action (deterministic for evaluation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            action = self.network.get_action(state_tensor, deterministic=True)
            action = action.cpu().numpy()[0, 0]
        
        return float(np.clip(action, -2.0, 2.0))



