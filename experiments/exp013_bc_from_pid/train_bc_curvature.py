"""
Train behavioral cloning network on PID trajectories (CURVATURE SPACE)
State: 58D with current/target curvatures, error derivative, friction circle
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import trange

# State: 58D curvature-space representation
STATE_DIM = 58
ACTION_DIM = 1

# Hyperparameters
HIDDEN_DIM = 256  # Increased from 128 for more capacity
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
NUM_EPOCHS = 50

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


class BCDataset(Dataset):
    """Dataset for behavioral cloning"""
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions).unsqueeze(-1)  # [N, 1]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BCNetwork(nn.Module):
    """
    Behavioral cloning network (curvature-space)
    Architecture inspired by beautiful_lander.py:
    - Shared trunk for feature extraction
    - Separate actor head for action prediction
    - Outputs μ (mean action) with learnable log_std
    """
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        
        # Shared trunk (feature extraction)
        trunk_layers = []
        trunk_layers.append(nn.Linear(state_dim, hidden_dim))
        trunk_layers.append(nn.Tanh())  # Tanh for bounded gradients (better for control than ReLU)
        
        for _ in range(num_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.Tanh())
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Actor head (policy)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1], then scale to [-2, 2]
        )
        
        # Learnable log_std (initialized to log(0.1) for small exploration)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.1))
    
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
    
    def compute_loss(self, states, actions):
        """Negative log likelihood loss"""
        mean, std = self.forward(states)
        # Clamp actions to valid range for numerical stability
        actions_clamped = torch.clamp(actions, -2.0, 2.0)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_clamped).sum(-1)
        return -log_prob.mean()


def train():
    print("="*60)
    print("Training BC from PID (CURVATURE SPACE)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    with open('pid_trajectories_curvature.pkl', 'rb') as f:
        data = pickle.load(f)
    
    states = data['states']
    actions = data['actions']
    
    print(f"Loaded {len(states):,} samples")
    print(f"State dim: {states.shape[1]}")
    
    # Normalize states
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std
    
    # Save normalization stats
    np.save('state_mean_curvature.npy', state_mean)
    np.save('state_std_curvature.npy', state_std)
    print(f"✓ Saved normalization stats")
    
    # Create dataset and dataloader
    dataset = BCDataset(states_norm, actions)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Create network
    network = BCNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    print(f"\nNetwork architecture:")
    print(f"  State dim: {STATE_DIM} (curvature-space)")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Output: μ ± σ")
    print(f"  Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {device}")
    print()
    
    # Training loop
    best_loss = float('inf')
    
    pbar = trange(NUM_EPOCHS, desc="Training BC")
    for epoch in range(NUM_EPOCHS):
        network.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            loss = network.compute_loss(batch_states, batch_actions)
            loss.backward()
            # Gradient clipping (from beautiful_lander.py)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'state_mean': state_mean,
                'state_std': state_std,
            }, 'bc_best_curvature.pth')
        
        if epoch % 5 == 0:
            sigma = network.log_std.exp().item()
            pbar.write(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | σ: {sigma:.4f}")
        
        pbar.update(1)
    
    pbar.close()
    
    print(f"\n" + "="*60)
    print(f"Training complete!")
    print(f"="*60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"✅ Saved: bc_best_curvature.pth")
    print("="*60 + "\n")


if __name__ == '__main__':
    train()

