"""
Pre-train critic on BC rollouts.
Clean, simple, surgical.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from bc_controller import BCController
from ppo_models import ActorCritic
from ppo_controller import compute_rewards_from_trajectory


def collect_bc_rollouts(bc_checkpoint_path, model, csv_files, num_files=50):
    """
    Collect rollouts using BC controller.
    Store (obs, returns) for critic training.
    """
    all_obs = []
    all_returns = []
    
    sampled_files = random.sample(csv_files, min(num_files, len(csv_files)))
    
    # Load BC weights once (avoid repeated loading and printing)
    from ppo_controller import PPOController
    from ppo_models import ActorCritic
    from train_bc import MLP
    
    bc_model = MLP(input_dim=55, hidden_sizes=[64, 32])
    bc_model.load_state_dict(torch.load(bc_checkpoint_path))
    
    for csv_path in tqdm(sampled_files, desc="Collecting BC rollouts"):
        # Create temporary ActorCritic with BC weights
        ac_temp = ActorCritic()
        
        # Copy weights from BC model (avoid repeated file I/O)
        ac_temp.actor_fc1.weight.data.copy_(bc_model.fc1.weight.data)
        ac_temp.actor_fc1.bias.data.copy_(bc_model.fc1.bias.data)
        ac_temp.actor_fc2.weight.data.copy_(bc_model.fc2.weight.data)
        ac_temp.actor_fc2.bias.data.copy_(bc_model.fc2.bias.data)
        ac_temp.actor_mean.weight.data.copy_(bc_model.fc3.weight.data)
        ac_temp.actor_mean.bias.data.copy_(bc_model.fc3.bias.data)
        
        ppo_ctrl = PPOController(ac_temp, steer_scale=2.0, collecting=True)
        
        sim = TinyPhysicsSimulator(model, str(csv_path), controller=ppo_ctrl, debug=False)
        sim.rollout()
        
        # Compute rewards
        rewards = compute_rewards_from_trajectory(ppo_ctrl.trajectory)
        
        # Compute returns (discounted sum of future rewards)
        gamma = 0.99
        returns = np.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Store observations and returns
        for step, ret in zip(ppo_ctrl.trajectory, returns):
            all_obs.append(step['obs'])
            all_returns.append(ret)
    
    all_obs = np.array(all_obs)
    all_returns = np.array(all_returns)
    
    # Normalize returns for stable training
    returns_mean = all_returns.mean()
    returns_std = all_returns.std()
    normalized_returns = (all_returns - returns_mean) / (returns_std + 1e-8)
    
    return all_obs, normalized_returns, returns_mean, returns_std


def pretrain_critic(model_path='../../models/tinyphysics.onnx',
                   data_dir='../../data',
                   bc_checkpoint='./outputs/best_model.pt',
                   output_path='./outputs/pretrained_critic.pt',
                   num_rollouts=100,
                   num_epochs=50,
                   batch_size=256,
                   lr=1e-3,
                   seed=42):
    """
    Pre-train critic on BC rollouts.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"{'='*70}")
    print("Pre-training Critic on BC Rollouts")
    print(f"{'='*70}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading TinyPhysics model...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Get data files
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    train_files = all_files[:800]
    
    # Collect BC rollouts
    print(f"Collecting {num_rollouts} BC rollouts...")
    obs, normalized_returns, returns_mean, returns_std = collect_bc_rollouts(
        bc_checkpoint, tinyphysics_model, train_files, num_files=num_rollouts
    )
    
    print(f"\nData collected:")
    print(f"  Observations: {obs.shape}")
    print(f"  Normalized returns: {normalized_returns.shape}")
    print(f"  Return range (normalized): [{normalized_returns.min():.2f}, {normalized_returns.max():.2f}]")
    print(f"  Return mean (normalized): {normalized_returns.mean():.4f}")
    print(f"  Return std (normalized): {normalized_returns.std():.4f}")
    print(f"  Original return mean: {returns_mean:.2f}")
    print(f"  Original return std: {returns_std:.2f}")
    
    # Create ActorCritic (we only train the critic)
    print(f"\nCreating critic...")
    actor_critic = ActorCritic(state_dim=55, action_dim=1)
    
    # Optimizer for critic only
    critic_optimizer = optim.Adam(
        list(actor_critic.critic_fc1.parameters()) +
        list(actor_critic.critic_fc2.parameters()) +
        list(actor_critic.critic_value.parameters()),
        lr=lr
    )
    
    criterion = nn.MSELoss()
    
    # Convert to tensors
    obs_t = torch.FloatTensor(obs)
    returns_t = torch.FloatTensor(normalized_returns)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(obs_t, returns_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"\nTraining critic for {num_epochs} epochs...")
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        
        for batch_obs, batch_returns in dataloader:
            # Forward pass (critic only)
            _, _, values = actor_critic(batch_obs)
            
            # MSE loss
            loss = criterion(values.squeeze(-1), batch_returns)
            
            # Backward pass
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()
            
            epoch_loss += loss.item() * batch_obs.size(0)
        
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.6f}")
    
    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")
    
    # Test critic predictions
    print(f"\nTesting critic predictions (normalized space)...")
    actor_critic.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(obs), size=10)
        sample_obs = obs_t[sample_idx]
        sample_returns_norm = returns_t[sample_idx]
        
        _, _, predicted_values_norm = actor_critic(sample_obs)
        predicted_values_norm = predicted_values_norm.squeeze(-1)
        
        print(f"\nSample predictions:")
        print(f"{'True (norm)':<15} {'Pred (norm)':<15} {'Error (norm)':<15}")
        print(f"{'-'*45}")
        for true_val, pred_val in zip(sample_returns_norm, predicted_values_norm):
            error = abs(true_val.item() - pred_val.item())
            print(f"{true_val.item():<15.4f} {pred_val.item():<15.4f} {error:<15.4f}")
    
    # Save only critic weights
    print(f"\nSaving pre-trained critic...")
    critic_state = {
        'critic_fc1': actor_critic.critic_fc1.state_dict(),
        'critic_fc2': actor_critic.critic_fc2.state_dict(),
        'critic_value': actor_critic.critic_value.state_dict(),
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'final_loss': losses[-1],
        'config': {
            'num_rollouts': num_rollouts,
            'num_epochs': num_epochs,
            'lr': lr
        }
    }
    
    torch.save(critic_state, output_path)
    print(f"Saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("✓ Critic pre-training complete!")
    print(f"{'='*70}")
    print(f"\nNext step: Use this pre-trained critic for PPO training")
    print(f"The critic now understands value estimates from BC rollouts.")
    print(f"{'='*70}")
    
    return actor_critic


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-train critic on BC rollouts')
    parser.add_argument('--model_path', type=str, default='../../models/tinyphysics.onnx')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--bc_checkpoint', type=str, default='./outputs/best_model.pt')
    parser.add_argument('--output', type=str, default='./outputs/pretrained_critic.pt')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of BC rollouts to collect')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    pretrain_critic(
        model_path=args.model_path,
        data_dir=args.data_dir,
        bc_checkpoint=args.bc_checkpoint,
        output_path=args.output,
        num_rollouts=args.num_rollouts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
