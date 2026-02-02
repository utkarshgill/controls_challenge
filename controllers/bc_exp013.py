"""
BC Controller from exp013
Can be evaluated using: python tinyphysics.py --controller bc_exp013 --data_path ./data --num_segs 100
"""
import numpy as np
import torch
from pathlib import Path

# Hyperparameters (must match training)
STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


class BCNetwork(torch.nn.Module):
    """Behavioral cloning network"""
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        
        # Build MLP
        layers = []
        layers.append(torch.nn.Linear(state_dim, hidden_dim))
        layers.append(torch.nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.Tanh())
        
        self.trunk = torch.nn.Sequential(*layers)
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
    BC Controller - official interface for tinyphysics.py
    NOTE: A new instance is created for each rollout, so error_integral resets automatically
    """
    def __init__(self):
        # Load checkpoint
        checkpoint_path = Path(__file__).parent.parent / "experiments/exp013_bc_from_pid/bc_best.pth"
        
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
        # curvature = (lat_accel - roll) / v²
        # Each timestep uses its own future v_ego (physically correct!)
        
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
        v_ego_sq = np.maximum(future_v_egos ** 2, 1.0)  # Prevent division by zero
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

