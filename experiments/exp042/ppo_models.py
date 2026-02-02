"""
PPO models for controls challenge.
Architecture inspired by beautiful_lander.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_bc import MLP


class ActorCritic(nn.Module):
    """
    Actor-Critic for PPO fine-tuning.
    
    Actor: Predicts steering command (continuous action)
    Critic: Predicts state value for advantage estimation
    
    Can be warm-started from BC MLP weights.
    """
    def __init__(self, state_dim=55, action_dim=1, hidden_sizes=[64, 32]):
        super(ActorCritic, self).__init__()
        
        # Actor network (matches BC MLP architecture)
        self.actor_fc1 = nn.Linear(state_dim, hidden_sizes[0], bias=True)
        self.actor_fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.actor_mean = nn.Linear(hidden_sizes[1], action_dim, bias=True)
        
        # Learnable log std for action distribution (like beautiful_lander.py)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (separate from actor)
        self.critic_fc1 = nn.Linear(state_dim, hidden_sizes[0], bias=True)
        self.critic_fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.critic_value = nn.Linear(hidden_sizes[1], 1, bias=True)
        
    def forward(self, state):
        """
        Forward pass for both actor and critic.
        
        Returns:
            action_mean: Mean of action distribution (batch, action_dim)
            action_std: Std of action distribution (batch, action_dim)
            state_value: Estimated value of state (batch, 1)
        """
        # Actor forward
        x = self.actor_fc1(state)
        x = torch.tanh(x)
        x = self.actor_fc2(x)
        x = torch.tanh(x)
        action_mean = self.actor_mean(x)
        action_mean = torch.tanh(action_mean)  # Squash to [-1, 1]
        
        # Action std from learned parameter
        action_std = self.log_std.exp()
        
        # Critic forward
        v = self.critic_fc1(state)
        v = torch.tanh(v)
        v = self.critic_fc2(v)
        v = torch.tanh(v)
        state_value = self.critic_value(v)
        
        return action_mean, action_std, state_value
    
    def load_bc_weights(self, bc_model_path):
        """
        Warm-start actor from pre-trained BC model.
        
        Args:
            bc_model_path: Path to BC model checkpoint (best_model.pt)
        """
        # Load BC model
        bc_model = MLP(input_dim=55, hidden_sizes=[64, 32])
        bc_model.load_state_dict(torch.load(bc_model_path))
        
        # Copy weights to actor
        self.actor_fc1.weight.data.copy_(bc_model.fc1.weight.data)
        self.actor_fc1.bias.data.copy_(bc_model.fc1.bias.data)
        
        self.actor_fc2.weight.data.copy_(bc_model.fc2.weight.data)
        self.actor_fc2.bias.data.copy_(bc_model.fc2.bias.data)
        
        self.actor_mean.weight.data.copy_(bc_model.fc3.weight.data)
        self.actor_mean.bias.data.copy_(bc_model.fc3.bias.data)
        
        print(f"✓ Actor warm-started from BC checkpoint: {bc_model_path}")
        print(f"  Critic initialized randomly")
    
    @torch.inference_mode()
    def act(self, state, deterministic=False):
        """
        Select action given state.
        
        Args:
            state: numpy array of shape (state_dim,) or (batch, state_dim)
            deterministic: if True, use mean action (no sampling)
            
        Returns:
            action: scaled action in valid range
            raw_action: action before scaling (for PPO logging)
        """
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Add batch dim if needed
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        # Get action distribution
        action_mean, action_std, _ = self(state)
        
        if deterministic:
            raw_action = action_mean
        else:
            # Sample from Normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.sample()
        
        # action_mean is already in [-1, 1] from tanh
        # Scale to steering range [-2, 2]
        action = raw_action * 2.0
        action = torch.clamp(action, -2.0, 2.0)
        
        # Remove batch dim if input was single state
        if action.shape[0] == 1:
            action = action.squeeze(0)
            raw_action = raw_action.squeeze(0)
        
        return action, raw_action


def test_actor_critic():
    """Test ActorCritic initialization and forward pass"""
    print("Testing ActorCritic...")
    
    # Create model
    ac = ActorCritic(state_dim=55, action_dim=1, hidden_sizes=[64, 32])
    
    # Test forward pass
    test_state = torch.randn(4, 55)  # Batch of 4 states
    action_mean, action_std, value = ac(test_state)
    
    print(f"  Input shape: {test_state.shape}")
    print(f"  Action mean shape: {action_mean.shape}")
    print(f"  Action std shape: {action_std.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in ac.parameters())}")
    print(f"  Actor parameters: {sum(p.numel() for p in list(ac.actor_fc1.parameters()) + list(ac.actor_fc2.parameters()) + list(ac.actor_mean.parameters()) + [ac.log_std])}")
    print(f"  Critic parameters: {sum(p.numel() for p in list(ac.critic_fc1.parameters()) + list(ac.critic_fc2.parameters()) + list(ac.critic_value.parameters()))}")
    
    # Test action sampling
    action, raw_action = ac.act(test_state[0].numpy())
    print(f"\n  Single state action: {action.item():.4f}")
    print(f"  Raw action: {raw_action.item():.4f}")
    
    print("\n✓ ActorCritic test passed!")


if __name__ == '__main__':
    test_actor_critic()
