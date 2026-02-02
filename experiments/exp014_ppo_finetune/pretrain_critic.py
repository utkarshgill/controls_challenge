"""
Stage 1: Pre-train the Critic on BC Rollouts

Problem: BC only trained the actor. Critic is random → garbage value estimates → PPO breaks.
Solution: Collect rollouts with BC policy, train critic to predict returns.

Then Stage 2 will use the pre-trained critic for stable PPO updates.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import random

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX, DEL_T
from train_ppo_v5_frozen import ActorCritic, PPOController, STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS

# Hyperparameters
NUM_ROLLOUTS = 100  # Collect 100 rollouts with BC
CRITIC_EPOCHS = 50  # Train critic for 50 epochs
CRITIC_LR = 1e-3
BATCH_SIZE = 256

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def collect_bc_rollouts(actor_critic, model, data_files, state_mean, state_std, num_rollouts):
    """Collect rollouts using BC policy (deterministic)"""
    print(f"\n{'='*80}")
    print(f"Collecting {num_rollouts} rollouts with BC policy...")
    print(f"{'='*80}")
    
    all_states = []
    all_returns = []
    
    for i, data_file in enumerate(tqdm(random.sample(data_files, num_rollouts), desc="Rollouts")):
        controller = PPOController(actor_critic, state_mean, state_std, device, 
                                   collect_data=True, deterministic=True)
        sim = TinyPhysicsSimulator(model, data_file, controller=controller)
        cost_dict = sim.rollout()
        
        traj = controller.get_trajectory()
        if traj is not None and len(traj['rewards']) > 0:
            # Compute returns (sum of future rewards)
            rewards = traj['rewards']
            returns = []
            G = 0
            gamma = 0.99
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            all_states.extend(traj['states'])
            all_returns.extend(returns)
    
    return np.array(all_states), np.array(all_returns)


def train_critic(actor_critic, states, returns, epochs, lr, batch_size):
    """Train critic to predict returns"""
    print(f"\n{'='*80}")
    print(f"Training Critic to Predict Returns")
    print(f"{'='*80}")
    print(f"States: {len(states)}")
    print(f"Epochs: {epochs}")
    print(f"LR: {lr}")
    print(f"{'='*80}\n")
    
    # Only optimize critic
    optimizer = optim.Adam(actor_critic.critic_head.parameters(), lr=lr)
    
    states_tensor = torch.FloatTensor(states).to(device)
    returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(device)
    
    for epoch in trange(epochs, desc="Training Critic"):
        # Shuffle data
        indices = np.random.permutation(len(states))
        
        total_loss = 0
        num_batches = 0
        
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states_tensor[batch_idx]
            batch_returns = returns_tensor[batch_idx]
            
            # Forward pass
            features = actor_critic.trunk(batch_states)
            predicted_values = actor_critic.critic_head(features)
            
            # MSE loss
            loss = nn.MSELoss()(predicted_values, batch_returns)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            tqdm.write(f"Epoch {epoch+1}/{epochs} | Critic Loss: {avg_loss:.4f}")
    
    print(f"\n✅ Critic pre-training complete!")


def main():
    print("="*80)
    print("Stage 1: Pre-train Critic on BC Rollouts")
    print("="*80)
    
    # Load BC checkpoint
    bc_checkpoint_path = Path(__file__).parent.parent / "exp013_bc_from_pid/bc_best.pth"
    if not bc_checkpoint_path.exists():
        print(f"❌ BC checkpoint not found: {bc_checkpoint_path}")
        return
    
    print(f"\n✓ Loading BC checkpoint: {bc_checkpoint_path}")
    bc_checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
    
    state_mean = bc_checkpoint['state_mean']
    state_std = bc_checkpoint['state_std']
    print(f"✓ Loaded normalization stats")
    
    # Create actor-critic
    actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS, 
                               bc_checkpoint=bc_checkpoint).to(device)
    actor_critic.eval()  # Eval mode for BC rollouts
    
    print(f"\n✓ Actor-Critic created")
    print(f"  Actor (from BC): ✓")
    print(f"  Critic (random): ⚠️  ← We'll fix this!")
    
    # Load data
    model_path = Path(__file__).parent.parent.parent / "models/tinyphysics.onnx"
    model = TinyPhysicsModel(str(model_path), debug=False)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_files = sorted([str(f) for f in data_dir.glob("*.csv")])
    
    random.seed(42)
    
    print(f"\n✓ Data loaded: {len(data_files)} files")
    
    # Stage 1: Collect BC rollouts
    states, returns = collect_bc_rollouts(actor_critic, model, data_files, 
                                          state_mean, state_std, NUM_ROLLOUTS)
    
    # Stage 2: Train critic
    actor_critic.train()  # Training mode for critic
    train_critic(actor_critic, states, returns, CRITIC_EPOCHS, CRITIC_LR, BATCH_SIZE)
    
    # Save actor-critic with pre-trained critic
    save_path = Path(__file__).parent / "actor_critic_pretrained.pth"
    torch.save({
        'model_state_dict': actor_critic.state_dict(),
        'state_mean': state_mean,
        'state_std': state_std,
    }, save_path)
    
    print(f"\n{'='*80}")
    print(f"✅ Saved actor-critic with pre-trained critic:")
    print(f"   {save_path}")
    print(f"{'='*80}")
    print(f"\nNow run Stage 2 (PPO fine-tuning) with this checkpoint!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()



