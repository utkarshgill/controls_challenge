"""
BC training for exp044 (attention architecture).
Network learns residual over PID baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

# PID coefficients (from controllers/pid.py)
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053
pid_residual_weight = 0.1  # Network learns residual over 10% PID baseline

# Network architecture (matching train.py)
state_dim, action_dim = 57, 1
hidden_dim = 128
attention_dim = 32
actor_layers, critic_layers = 4, 4


class ActorCritic(nn.Module):
    """Minimal attention architecture (same as train.py)"""
    def __init__(self, state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers):
        super().__init__()
        self.attention_dim = attention_dim
        
        # Query: encode current state
        self.query_net = nn.Linear(7, attention_dim)
        
        # Keys/Values: encode each future_κ
        self.key_net = nn.Linear(1, attention_dim)
        self.value_net = nn.Linear(1, attention_dim)
        
        # Actor: attended features → action (add small MLP for stability)
        self.actor = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic: attended features → value (add small MLP for stability)
        self.critic = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Clamp inputs for stability
        state = torch.clamp(state, -100, 100)
        
        current = state[:, :7]
        future_κ = state[:, 7:57]
        
        # Query: "Given current state, what future matters?"
        query = self.query_net(current)  # [batch, attention_dim]
        query = torch.clamp(query, -10, 10)
        
        # Keys/Values: encode each future curvature
        keys = self.key_net(future_κ.unsqueeze(-1))  # [batch, 50, attention_dim]
        values = self.value_net(future_κ.unsqueeze(-1))  # [batch, 50, attention_dim]
        keys = torch.clamp(keys, -10, 10)
        values = torch.clamp(values, -10, 10)
        
        # Attention
        scores = torch.bmm(keys, query.unsqueeze(-1)) / np.sqrt(self.attention_dim)
        scores = torch.clamp(scores, -10, 10)  # Prevent extreme values
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        
        # Weighted sum
        attended = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        attended = torch.clamp(attended, -10, 10)  # Prevent extreme attended features
        
        # Output
        action_mean = self.actor(attended)
        action_mean = torch.clamp(action_mean, -10, 10)  # Prevent extreme actions
        action_std = self.log_std.exp().clamp(min=1e-6, max=2.0)  # Bound std
        value = self.critic(attended)
        
        return action_mean, action_std, value


def collect_bc_data_worker(args):
    """Worker function for parallel BC data collection"""
    csv_path, model_path_str = args
    
    model = TinyPhysicsModel(model_path_str, debug=False)
    pid = PIDController()
    
    observations = []
    network_targets = []  # Targets for network (residual over PID baseline)
    
    # PID state for computing feedback component
    error_integral = 0.0
    prev_error = 0.0
    
    # Run PID controller
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=pid, debug=False)
    
    # Hook into controller to capture obs+actions
    original_update = pid.update
    def hooked_update(target_lataccel, current_lataccel, state, future_plan):
        nonlocal error_integral, prev_error
        # Compute features (same as PPOController)
        n_future = len(future_plan.lataccel)
        
        if n_future == 50:
            future_lat = np.asarray(future_plan.lataccel, dtype=np.float32)
            future_roll = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
            future_v = np.asarray(future_plan.v_ego, dtype=np.float32)
        elif n_future == 0:
            future_lat = np.full(50, target_lataccel, dtype=np.float32)
            future_roll = np.full(50, state.roll_lataccel, dtype=np.float32)
            future_v = np.full(50, state.v_ego, dtype=np.float32)
        else:
            future_lat = np.asarray(future_plan.lataccel, dtype=np.float32)
            future_roll = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
            future_v = np.asarray(future_plan.v_ego, dtype=np.float32)
            pad_len = 50 - n_future
            future_lat = np.pad(future_lat, (0, pad_len), mode='edge')
            future_roll = np.pad(future_roll, (0, pad_len), mode='edge')
            future_v = np.pad(future_v, (0, pad_len), mode='edge')
        
        # Pure geometric curvature
        future_v_sq = np.maximum(future_v ** 2, 25.0)
        curvature = (future_lat - future_roll) / future_v_sq
        np.clip(curvature, -1.0, 1.0, out=curvature)
        
        v_sq = max(state.v_ego ** 2, 25.0)
        current_curvature = np.clip((target_lataccel - state.roll_lataccel) / v_sq, -1.0, 1.0)
        
        # Error for feedback
        error = target_lataccel - current_lataccel
        
        # Compute PID feedback component (matching train.py)
        error_integral += error
        error_diff = error - prev_error
        prev_error = error
        pid_feedback = PID_P * error + PID_I * error_integral + PID_D * error_diff
        
        obs = np.empty(57, dtype=np.float32)
        obs[0] = target_lataccel
        obs[1] = current_lataccel
        obs[2] = error
        obs[3] = state.v_ego
        obs[4] = state.a_ego
        obs[5] = state.roll_lataccel
        obs[6] = current_curvature
        obs[7:57] = curvature  # 50 future curvatures
        
        # Get full PID action
        full_pid_action = original_update(target_lataccel, current_lataccel, state, future_plan)
        
        # Network should learn the residual: (full_action - α * pid_baseline)
        # So that at inference: α * pid_baseline + network_output = full_action
        network_target = full_pid_action - pid_residual_weight * pid_feedback
        
        observations.append(obs)
        network_targets.append(network_target)
        
        return full_pid_action
    
    pid.update = hooked_update
    sim.rollout()
    
    return np.array(observations), np.array(network_targets)


def train_bc(actor_critic, obs_batch, action_batch, epochs=10, lr=1e-3, batch_size=1024):
    """Train network to imitate PID"""
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
    
    dataset_size = len(obs_batch)
    
    for epoch in range(1, epochs + 1):
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        obs_shuffled = obs_batch[indices]
        action_shuffled = action_batch[indices]
        
        total_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            batch_obs = torch.FloatTensor(obs_shuffled[i:i+batch_size])
            batch_actions = torch.FloatTensor(action_shuffled[i:i+batch_size]).unsqueeze(1)
            
            # Forward pass
            action_mean, _, _ = actor_critic(batch_obs)
            
            # MSE loss
            loss = F.mse_loss(action_mean, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")


def main():
    print("=== BC Training for Exp044 (Attention) ===\n")
    
    # Setup
    data_dir = Path(__file__).parent.parent.parent / 'data'
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    csv_files = sorted(data_dir.glob('*.csv'))[:1000]  # Use 1000 files for BC
    
    print(f"Collecting demonstrations from {len(csv_files)} CSVs...")
    
    # Parallel data collection
    args = [(f, str(model_path)) for f in csv_files]
    results = process_map(collect_bc_data_worker, args, max_workers=8, chunksize=20)
    
    # Concatenate all data
    all_obs = np.concatenate([r[0] for r in results])
    all_actions = np.concatenate([r[1] for r in results])
    
    print(f"Collected {len(all_obs)} samples")
    print(f"Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
    print(f"Action mean/std: {all_actions.mean():.3f} / {all_actions.std():.3f}\n")
    
    # Initialize network
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers)
    
    print("Training network to imitate PID (learning residual)...")
    train_bc(actor_critic, all_obs, all_actions, epochs=10, lr=1e-3)
    
    # Save checkpoint
    save_path = Path(__file__).parent / 'bc_init.pt'
    torch.save(actor_critic.state_dict(), save_path)
    print(f"\n✓ BC checkpoint saved to {save_path.name}")


if __name__ == '__main__':
    main()
