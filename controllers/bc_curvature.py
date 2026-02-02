"""
BC Controller with CURVATURE-SPACE representation
State: 58D including current/target curvatures, error derivative, friction circle
"""
import numpy as np
import torch
from pathlib import Path

# Hyperparameters (must match training)
STATE_DIM = 58
ACTION_DIM = 1
HIDDEN_DIM = 256  # Increased from 128 for more capacity
NUM_LAYERS = 3

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Constants from tinyphysics
DEL_T = 0.1  # Timestep


class BCNetwork(torch.nn.Module):
    """
    Behavioral cloning network (curvature-space)
    Architecture inspired by beautiful_lander.py with trunk+head design
    """
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        
        # Shared trunk (feature extraction)
        trunk_layers = []
        trunk_layers.append(torch.nn.Linear(state_dim, hidden_dim))
        trunk_layers.append(torch.nn.Tanh())
        
        for _ in range(num_layers - 1):
            trunk_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(torch.nn.Tanh())
        
        self.trunk = torch.nn.Sequential(*trunk_layers)
        
        # Actor head (policy)
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()  # Output in [-1, 1], then scale to [-2, 2]
        )
        
        # Learnable log_std
        self.log_std = torch.nn.Parameter(torch.ones(action_dim) * np.log(0.1))
    
    def forward(self, state):
        """Returns mean and std"""
        features = self.trunk(state)
        mean = self.mean_head(features) * 2.0  # Scale tanh output [-1,1] to [-2,2]
        std = self.log_std.exp()
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        mean, std = self.forward(state)
        if deterministic:
            return mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action


class Controller:
    """
    BC Controller with curvature-space representation
    NOTE: A new instance is created for each rollout
    """
    def __init__(self):
        # Load checkpoint
        checkpoint_path = Path(__file__).parent.parent / "experiments/exp013_bc_from_pid/bc_best_curvature.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load network
        self.network = BCNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        
        # Load normalization stats
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        
        # Internal state
        self.curvature_error_integral = 0.0
        self.prev_curvature_error = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Controller interface (called by tinyphysics simulator)
        
        Returns: Steering command in [-2, 2]
        """
        # Current vehicle state
        v_ego = state.v_ego
        a_ego = state.a_ego
        
        # Convert to curvature space (speed-invariant)
        v_ego_sq = max(v_ego ** 2, 1.0)
        current_curvature = (current_lataccel - state.roll_lataccel) / v_ego_sq  # Subtract roll (consistent with future)
        target_curvature = target_lataccel / v_ego_sq
        
        # Curvature error (PID terms)
        curvature_error = target_curvature - current_curvature
        self.curvature_error_integral += curvature_error
        curvature_error_derivative = (curvature_error - self.prev_curvature_error) / DEL_T
        self.prev_curvature_error = curvature_error
        
        # Friction circle: available lateral capacity
        max_combined_accel = 10.0
        longitudinal_fraction = min(abs(a_ego) / max_combined_accel, 1.0)
        friction_available = np.sqrt(max(0.0, 1.0 - longitudinal_fraction**2))
        
        # Calculate future curvatures (each with own future v_ego)
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
        
        # Each curvature uses its own future speed (physically correct)
        v_ego_sq_future = np.maximum(future_v_egos ** 2, 1.0)
        curvatures = (future_lataccels - future_roll) / v_ego_sq_future
        
        # Build state vector (58D)
        state_vec = np.array([
            current_curvature,
            target_curvature,
            curvature_error,
            self.curvature_error_integral,
            curvature_error_derivative,
            v_ego,
            a_ego,
            friction_available,
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

