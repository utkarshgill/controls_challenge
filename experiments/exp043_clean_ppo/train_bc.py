"""
Quick BC training to warm-start PPO.
Uses same architecture as train.py for compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
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

# ActorCritic class (same as train.py)
class ActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic learning residual over PID baseline.
    Input: 7 current + 50 future_κ = 57 (pure problem statement)
    Output: Network learns FF + corrections, combined with PID feedback
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super().__init__()
        
        # Simple MLP with ReLU (matching beautiful_lander.py and train.py)
        actor = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        critic = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(critic_layers - 1):
            critic.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        critic.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*critic)
    
    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        value = self.critic(state)
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


def train_bc():
    """Train BC to imitate PID"""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    
    # Use subset for speed
    train_files = sorted(list(data_dir.glob('*.csv')))[:1000]  # 1K files = fast
    
    # Create model (same as PPO) - MUST match train.py exactly!
    actor_critic = ActorCritic(state_dim=57, action_dim=1, hidden_dim=128, 
                               actor_layers=4, critic_layers=4)
    optimizer = optim.Adam(actor_critic.actor.parameters(), lr=1e-3)
    
    print(f"Collecting BC data from {len(train_files)} files (parallel)...")
    
    # Prepare args for parallel processing
    args = [(csv, str(model_path)) for csv in train_files]
    
    # Parallel collection with 8 workers
    results = process_map(collect_bc_data_worker, args, max_workers=8, chunksize=10, desc="Collecting data")
    
    # Unpack results
    all_obs = []
    all_actions = []
    for obs, actions in results:
        all_obs.append(obs)
        all_actions.append(actions)
    
    # Concatenate all data
    all_obs = np.concatenate(all_obs)
    all_actions = np.concatenate(all_actions)
    
    print(f"\nBC dataset: {len(all_obs)} samples")
    print(f"Training BC for 10 epochs (intentionally undertrained)...\n")
    
    # Train BC
    batch_size = 1024
    for epoch in range(10):  # Intentionally few epochs!
        perm = np.random.permutation(len(all_obs))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(all_obs), batch_size):
            idx = perm[i:i+batch_size]
            obs_batch = torch.FloatTensor(all_obs[idx])
            actions_batch = torch.FloatTensor(all_actions[idx]).unsqueeze(-1)
            
            # Forward
            action_pred = actor_critic.actor(obs_batch)
            loss = nn.MSELoss()(action_pred, actions_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        print(f"Epoch {epoch+1}/10 | Loss: {epoch_loss/n_batches:.6f}")
    
    # Save
    torch.save(actor_critic.state_dict(), 'bc_init.pt')
    print("\n✓ BC checkpoint saved to bc_init.pt")
    print("✓ Ready for PPO fine-tuning!")


if __name__ == '__main__':
    train_bc()
