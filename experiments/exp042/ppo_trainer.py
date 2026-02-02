"""
PPO trainer - handles advantage computation and policy updates.
Adapted from beautiful_lander.py for controls challenge.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PPO:
    """
    Proximal Policy Optimization trainer.
    
    Key components (from beautiful_lander.py):
    - GAE for advantage estimation
    - Clipped surrogate loss
    - Separate optimizers for actor and critic
    - Gradient clipping for stability
    """
    
    def __init__(self, actor_critic, pi_lr=3e-4, vf_lr=1e-3, 
                 gamma=0.99, lamda=0.95, K_epochs=10, eps_clip=0.2, 
                 vf_coef=0.5, entropy_coef=0.001,
                 returns_mean=None, returns_std=None):
        """
        Args:
            actor_critic: ActorCritic model
            pi_lr: Policy learning rate
            vf_lr: Value function learning rate
            gamma: Discount factor
            lamda: GAE lambda parameter
            K_epochs: Number of PPO epochs per update
            eps_clip: PPO clipping parameter
            vf_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            returns_mean: Mean for normalizing returns (from critic pre-training)
            returns_std: Std for normalizing returns (from critic pre-training)
        """
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.returns_mean = returns_mean if returns_mean is not None else 0.0
        self.returns_std = returns_std if returns_std is not None else 1.0
        
        # Separate optimizers like beautiful_lander.py
        self.pi_optimizer = optim.Adam(
            list(actor_critic.actor_fc1.parameters()) +
            list(actor_critic.actor_fc2.parameters()) +
            list(actor_critic.actor_mean.parameters()) +
            [actor_critic.log_std],
            lr=pi_lr
        )
        self.vf_optimizer = optim.Adam(
            list(actor_critic.critic_fc1.parameters()) +
            list(actor_critic.critic_fc2.parameters()) +
            list(actor_critic.critic_value.parameters()),
            lr=vf_lr
        )
    
    def compute_advantages(self, rewards, state_values, is_done=True):
        """
        Compute GAE advantages for a single episode.
        
        Args:
            rewards: numpy array of shape (T,) - unnormalized
            state_values: numpy array of shape (T,) - normalized (from critic)
            is_done: whether episode terminated
            
        Returns:
            advantages: normalized advantages (T,)
            returns: targets for value function (T,) - unnormalized
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        
        # Denormalize state values for GAE computation
        # (critic predicts in normalized space, but rewards are unnormalized)
        denorm_values = state_values * self.returns_std + self.returns_mean
        
        # Terminal value (0 if done, otherwise last state value)
        next_value = 0.0 if is_done else denorm_values[-1]
        
        # Backward pass to compute GAE
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = denorm_values[t + 1]
            
            # TD error: r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_val - denorm_values[t]
            
            # GAE: A_t = δ_t + γλA_{t+1}
            gae = delta + self.gamma * self.lamda * gae
            advantages[t] = gae
        
        # Returns = advantages + baseline (unnormalized for advantages)
        returns = advantages + denorm_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_loss(self, batch_obs, batch_raw_actions, batch_old_logprobs,
                     batch_advantages, batch_returns):
        """
        Compute PPO loss (actor + critic + entropy).
        
        Args:
            batch_obs: States (batch, state_dim)
            batch_raw_actions: Actions before scaling (batch, action_dim)
            batch_old_logprobs: Log probs from data collection (batch,)
            batch_advantages: Normalized advantages (batch,)
            batch_returns: Value targets (batch,)
            
        Returns:
            total_loss: Combined loss
            actor_loss: Policy loss (for logging)
            critic_loss: Value loss (for logging)
            entropy: Policy entropy (for logging)
        """
        # Forward pass
        action_means, action_stds, state_values = self.actor_critic(batch_obs)
        
        # Compute log probabilities of taken actions
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = dist.log_prob(batch_raw_actions).sum(-1)
        
        # PPO clipped surrogate loss
        ratios = torch.exp(action_logprobs - batch_old_logprobs)
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss (MSE)
        # Normalize returns to match critic's pre-training scale
        normalized_returns = (batch_returns - self.returns_mean) / (self.returns_std + 1e-8)
        critic_loss = F.mse_loss(state_values.squeeze(-1), normalized_returns)
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().sum(-1).mean()
        
        # Combined loss
        total_loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
        
        return total_loss, actor_loss.item(), critic_loss.item(), entropy.item()
    
    def update(self, trajectory, rewards):
        """
        Update policy using collected trajectory and rewards.
        
        Args:
            trajectory: list of dicts with keys ['obs', 'raw_action', 'value']
            rewards: numpy array of rewards (T,)
            
        Returns:
            dict with training statistics
        """
        # Extract data from trajectory
        obs = np.array([t['obs'] for t in trajectory], dtype=np.float32)
        raw_actions = np.array([t['raw_action'] for t in trajectory], dtype=np.float32)
        old_values = np.array([t['value'] for t in trajectory], dtype=np.float32)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs)
        raw_actions_t = torch.FloatTensor(raw_actions).unsqueeze(-1)  # (T, 1)
        
        # Compute old log probabilities
        with torch.no_grad():
            action_means, action_stds, _ = self.actor_critic(obs_t)
            dist = torch.distributions.Normal(action_means, action_stds)
            old_logprobs = dist.log_prob(raw_actions_t).sum(-1)
        
        # Compute advantages using GAE
        advantages, returns = self.compute_advantages(rewards, old_values)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)
        
        # PPO update for K epochs
        losses = []
        for epoch in range(self.K_epochs):
            # Compute loss
            loss, actor_loss, critic_loss, entropy = self.compute_loss(
                obs_t, raw_actions_t, old_logprobs, advantages_t, returns_t
            )
            
            # Backprop
            self.pi_optimizer.zero_grad()
            self.vf_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (like beautiful_lander.py)
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_critic.actor_fc1.parameters()) +
                list(self.actor_critic.actor_fc2.parameters()) +
                list(self.actor_critic.actor_mean.parameters()) +
                [self.actor_critic.log_std],
                max_norm=0.5
            )
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_critic.critic_fc1.parameters()) +
                list(self.actor_critic.critic_fc2.parameters()) +
                list(self.actor_critic.critic_value.parameters()),
                max_norm=0.5
            )
            
            self.pi_optimizer.step()
            self.vf_optimizer.step()
            
            losses.append(loss.item())
        
        # Return statistics
        return {
            'loss': np.mean(losses),
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'mean_advantage': advantages.mean(),
            'mean_return': returns.mean(),
            'std': self.actor_critic.log_std.exp().item()
        }


def test_ppo():
    """Test PPO class with dummy data"""
    print("Testing PPO trainer...")
    
    from ppo_models import ActorCritic
    
    # Create ActorCritic
    ac = ActorCritic(state_dim=55, action_dim=1)
    
    # Create PPO trainer
    ppo = PPO(ac, pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lamda=0.95)
    
    # Create dummy trajectory
    T = 100
    trajectory = []
    for t in range(T):
        trajectory.append({
            'obs': np.random.randn(55).astype(np.float32),
            'raw_action': np.random.randn(1).astype(np.float32)[0],
            'value': np.random.randn(1).astype(np.float32)[0]
        })
    
    # Create dummy rewards
    rewards = np.random.randn(T).astype(np.float32)
    
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  Rewards shape: {rewards.shape}")
    
    # Test update
    stats = ppo.update(trajectory, rewards)
    
    print(f"\n  Update statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value:.6f}")
    
    print("\n✓ PPO trainer test passed!")


if __name__ == '__main__':
    test_ppo()
