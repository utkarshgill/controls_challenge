"""
PPO training for controls challenge.
Structure inspired by beautiful_lander.py's train_one_epoch pattern.
"""

import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from ppo_models import ActorCritic
from ppo_trainer import PPO
from ppo_controller import PPOController, compute_rewards_from_trajectory


def train_one_epoch(epoch, actor_critic, ppo, controller, model, csv_files, 
                    episodes_per_epoch=10):
    """
    Train for one epoch - inspired by beautiful_lander.py structure.
    
    Args:
        epoch: Current epoch number
        actor_critic: ActorCritic model
        ppo: PPO trainer
        controller: PPOController for rollouts
        model: TinyPhysicsModel
        csv_files: List of CSV files to sample from
        episodes_per_epoch: Number of episodes to collect
        
    Returns:
        dict with epoch statistics
    """
    # Sample CSV files for this epoch
    sampled_files = random.sample(csv_files, min(episodes_per_epoch, len(csv_files)))
    
    # Collect trajectories from multiple episodes
    all_trajectories = []
    all_rewards = []
    episode_costs = []
    
    for csv_path in sampled_files:
        # Reset controller for new episode
        controller.reset()
        controller.collecting = True
        
        # Run rollout
        sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
        costs = sim.rollout()
        
        # Compute rewards
        rewards = compute_rewards_from_trajectory(controller.trajectory)
        
        # Store
        all_trajectories.append(controller.trajectory)
        all_rewards.append(rewards)
        episode_costs.append(costs['total_cost'])
    
    # Combine all trajectories for batch update
    combined_trajectory = []
    combined_rewards = []
    for traj, rew in zip(all_trajectories, all_rewards):
        combined_trajectory.extend(traj)
        combined_rewards.extend(rew)
    
    combined_rewards = np.array(combined_rewards)
    
    # PPO update on all collected experience
    update_stats = ppo.update(combined_trajectory, combined_rewards)
    
    # Compute epoch statistics
    mean_cost = np.mean(episode_costs)
    mean_reward = np.mean([r.mean() for r in all_rewards])
    
    return {
        'epoch': epoch,
        'mean_cost': mean_cost,
        'mean_reward': mean_reward,
        'num_episodes': len(episode_costs),
        'std': update_stats['std'],
        'actor_loss': update_stats['actor_loss'],
        'critic_loss': update_stats['critic_loss'],
        'entropy': update_stats['entropy']
    }


def evaluate_policy(actor_critic, model, csv_files, num_files=10):
    """
    Evaluate policy on held-out files.
    
    Args:
        actor_critic: ActorCritic model
        model: TinyPhysicsModel
        csv_files: List of CSV files to evaluate on
        num_files: Number of files to test
        
    Returns:
        mean cost across files
    """
    controller = PPOController(actor_critic, steer_scale=2.0, collecting=False)
    costs = []
    
    eval_files = random.sample(csv_files, min(num_files, len(csv_files)))
    
    for csv_path in eval_files:
        controller.reset()
        sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)


def train_ppo(model_path='../../models/tinyphysics.onnx',
              data_dir='../../data',
              bc_checkpoint='./outputs/best_model.pt',
              critic_checkpoint=None,
              output_dir='./outputs',
              num_epochs=50,
              episodes_per_epoch=10,
              eval_interval=10,
              train_files_range=(0, 800),
              val_files_range=(800, 900),
              pi_lr=3e-4,
              vf_lr=1e-3,
              K_epochs=10,
              seed=42):
    """
    Train PPO from BC warm-start.
    
    Args:
        model_path: Path to TinyPhysics model
        data_dir: Directory with CSV files
        bc_checkpoint: Path to BC model for warm-start
        output_dir: Where to save trained models
        num_epochs: Number of training epochs
        episodes_per_epoch: CSV files per epoch
        eval_interval: Evaluate every N epochs
        train_files_range: (start, end) indices for training files
        val_files_range: (start, end) indices for validation files
        pi_lr: Policy learning rate
        vf_lr: Value learning rate
        K_epochs: PPO update epochs
        seed: Random seed
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get data files
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    
    train_files = all_files[train_files_range[0]:train_files_range[1]]
    val_files = all_files[val_files_range[0]:val_files_range[1]]
    
    print(f"{'='*70}")
    print("PPO Fine-Tuning from BC Warm-Start")
    print(f"{'='*70}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Epochs: {num_epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(f"Policy LR: {pi_lr}")
    print(f"Value LR: {vf_lr}")
    print(f"K epochs: {K_epochs}")
    print(f"{'='*70}\n")
    
    # Load TinyPhysics model
    print("Loading TinyPhysics model...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Create ActorCritic and load BC weights
    print(f"Creating ActorCritic...")
    actor_critic = ActorCritic(state_dim=55, action_dim=1, hidden_sizes=[64, 32])
    actor_critic.load_bc_weights(bc_checkpoint)
    
    # Load pre-trained critic if provided
    returns_mean, returns_std = None, None
    if critic_checkpoint:
        print(f"Loading pre-trained critic from {critic_checkpoint}...")
        critic_state = torch.load(critic_checkpoint, weights_only=False)
        actor_critic.critic_fc1.load_state_dict(critic_state['critic_fc1'])
        actor_critic.critic_fc2.load_state_dict(critic_state['critic_fc2'])
        actor_critic.critic_value.load_state_dict(critic_state['critic_value'])
        returns_mean = critic_state.get('returns_mean', None)
        returns_std = critic_state.get('returns_std', None)
        print(f"✓ Critic loaded! Pre-training loss: {critic_state['final_loss']:.6f}")
        if returns_mean is not None:
            print(f"  Value normalization: mean={returns_mean:.2f}, std={returns_std:.2f}")
    
    # Initialize exploration std (start conservative like our successful training)
    actor_critic.log_std.data.fill_(np.log(0.3))
    print(f"Initial exploration std: {actor_critic.log_std.exp().item():.4f}")
    
    # Create PPO trainer
    print(f"Creating PPO trainer...")
    ppo = PPO(actor_critic, pi_lr=pi_lr, vf_lr=vf_lr, gamma=0.99, lamda=0.95,
              K_epochs=K_epochs, eps_clip=0.2, vf_coef=0.5, entropy_coef=0.001,
              returns_mean=returns_mean, returns_std=returns_std)
    
    # Create controller
    controller = PPOController(actor_critic, steer_scale=2.0, collecting=True)
    
    print(f"\nStarting training...")
    print(f"{'='*70}\n")
    
    # Training loop
    history = []
    best_val_cost = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train one epoch
        stats = train_one_epoch(epoch, actor_critic, ppo, controller, 
                               tinyphysics_model, train_files, episodes_per_epoch)
        
        history.append(stats)
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d} | Cost: {stats['mean_cost']:8.2f} | "
                  f"Reward: {stats['mean_reward']:7.2f} | Std: {stats['std']:.4f} | "
                  f"Critic Loss: {stats['critic_loss']:.2f}")
        
        # Evaluate periodically
        if epoch % eval_interval == 0 and epoch > 0:
            val_cost = evaluate_policy(actor_critic, tinyphysics_model, val_files, num_files=10)
            print(f"  └─ Val cost: {val_cost:.2f}")
            
            # Save best model
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                torch.save(actor_critic.state_dict(), output_path / 'best_ppo_model.pt')
                print(f"     ✓ New best model saved!")
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    print(f"Best validation cost: {best_val_cost:.2f}")
    print(f"Final exploration std: {actor_critic.log_std.exp().item():.4f}")
    
    # Save final model and history
    torch.save(actor_critic.state_dict(), output_path / 'final_ppo_model.pt')
    torch.save({
        'history': history,
        'best_val_cost': best_val_cost,
        'config': {
            'num_epochs': num_epochs,
            'episodes_per_epoch': episodes_per_epoch,
            'pi_lr': pi_lr,
            'vf_lr': vf_lr,
            'K_epochs': K_epochs
        }
    }, output_path / 'ppo_training_history.pt')
    
    print(f"\nModels saved to {output_path}")
    print(f"  - best_ppo_model.pt")
    print(f"  - final_ppo_model.pt")
    print(f"  - ppo_training_history.pt")
    
    return actor_critic, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO from BC warm-start')
    parser.add_argument('--model_path', type=str, default='../../models/tinyphysics.onnx')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--bc_checkpoint', type=str, default='./outputs/best_model.pt')
    parser.add_argument('--critic_checkpoint', type=str, default=None,
                        help='Pre-trained critic checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--episodes_per_epoch', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--K_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_ppo(
        model_path=args.model_path,
        data_dir=args.data_dir,
        bc_checkpoint=args.bc_checkpoint,
        critic_checkpoint=args.critic_checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        eval_interval=args.eval_interval,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        K_epochs=args.K_epochs,
        seed=args.seed
    )
