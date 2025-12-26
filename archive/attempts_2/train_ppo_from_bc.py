#!/usr/bin/env python3
"""
PPO Training from BC Initialization

Loads BC weights and fine-tunes with PPO on the TinyPhysics control task.
Based on beautiful_lander.py battle-tested PPO implementation.

Architecture:
- State: 56D (same as BC)
- Action: 1D steering in [-2, 2]
- Network: ActorCritic with BC weights loaded into actor head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import glob
import os

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, STEER_RANGE
from controllers import BaseController

# Architecture (must match BC)
state_dim, action_dim = 56, 1
hidden_dim = 128
trunk_layers, head_layers = 1, 3

# State normalization (same as BC)
OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] +  # [error, error_diff, error_integral, lataccel, v_ego, curv]
    [1000.0] * 50,
    dtype=np.float32
)

# PPO hyperparameters (tuned for fine-tuning from BC)
n_epochs = 100
episodes_per_epoch = 20
n_parallel = 8  # Parallel simulators
batch_size = 2048
K_epochs = 4  # Fewer epochs to prevent overfitting to batch
lr = 1e-5  # Much lower for fine-tuning
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.1  # Tighter clipping for conservative updates
entropy_coef = 0.0  # No exploration bonus (BC already explores)

device = torch.device('cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """56D: PID terms (3) + current state (3) + 50 future curvatures"""
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
        super(ActorCritic, self).__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Low initial std for fine-tuning (0.05 instead of 1.0)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.05))
        
        # Critic head
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value
    
    @torch.no_grad()
    def act(self, state, deterministic=False, return_internals=False):
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action_value = torch.tanh(raw_action) * STEER_RANGE[1]
        if return_internals:
            return action_value.squeeze().cpu().numpy(), state_tensor, raw_action
        return action_value.squeeze().cpu().numpy()

class PPOController(BaseController):
    """Wraps PPO agent for use with TinyPhysicsSimulator"""
    def __init__(self, ppo, sim):
        self.ppo = ppo
        self.sim = sim
        self.step_count = 0
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = 0.0
        
    def reset(self):
        self.step_count = 0
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                               self.prev_error, self.error_integral)
        
        # Only record states/actions/rewards after warmup
        if self.step_count >= CONTROL_START_IDX:
            action = self.ppo.select_action(state_vec)
            
            # Compute reward immediately after action
            lat_cost = current_lataccel ** 2
            jerk_cost = (current_lataccel - self.prev_lataccel) ** 2
            reward = -(lat_cost + jerk_cost) / 100.0
            done = (self.step_count >= len(self.sim.data) - 1)
            
            self.ppo.rewards.append(reward)
            self.ppo.dones.append(done)
        else:
            # Use actor but don't record
            action = self.ppo.actor_critic.act(state_vec, deterministic=False)
        
        # Update state for next step
        self.error_integral += error
        self.prev_error = error
        self.prev_lataccel = current_lataccel
        self.step_count += 1
        
        return float(action.item()) if hasattr(action, 'item') else float(action)

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef
        self.states, self.actions, self.rewards, self.dones, self.values = [], [], [], [], []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).unsqueeze(0).to(device)
            action_mean, action_std, value = self.actor_critic(state_tensor)
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.sample()
            action_value = torch.tanh(raw_action) * STEER_RANGE[1]
            
            self.states.append(state_tensor)
            self.actions.append(raw_action)
            self.values.append(value)
            
            return action_value.squeeze(0).cpu().numpy()

    def update(self):
        """PPO update (from beautiful_lander.py)"""
        if len(self.rewards) == 0:
            return 0.0, 0.0, 0.0
        
        # CRITICAL: Wrap everything in no_grad like beautiful_lander.py line 134
        with torch.no_grad():
            rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
            
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            old_values = torch.cat(self.values).squeeze(-1)
            
            # Compute old logprobs
            action_means, action_stds, _ = self.actor_critic(old_states)
            dist = torch.distributions.Normal(action_means, action_stds)
            old_logprobs = tanh_log_prob(old_actions, dist)
            
            # Compute advantages
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = old_values[t + 1]
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - old_values[t]
                advantages[t] = last_gae = delta + self.gamma * self.lamda * (1 - dones[t]) * last_gae
            
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Now old_states, old_actions, old_logprobs, returns, advantages are all detached
        dataset = TensorDataset(old_states, old_actions, old_logprobs, returns, advantages)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        
        total_loss = 0
        for _ in range(self.K_epochs):
            for states_batch, actions_batch, old_logprobs_batch, returns_batch, advantages_batch in loader:
                action_means, action_stds, values = self.actor_critic(states_batch)
                dist = torch.distributions.Normal(action_means, action_stds)
                logprobs = tanh_log_prob(actions_batch, dist)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(logprobs - old_logprobs_batch)
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Clipped value loss (prevents huge value estimates)
                values_pred = values.squeeze(-1)
                critic_loss = F.mse_loss(values_pred, returns_batch)
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1)  # Tighter gradient clipping
                self.optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / (self.K_epochs * len(loader))
        self.states, self.actions, self.rewards, self.dones, self.values = [], [], [], [], []
        return avg_loss, advantages.mean().item(), returns.mean().item()

def rollout_episode(ppo, model, data_file):
    """Run one episode and collect trajectory"""
    sim = TinyPhysicsSimulator(model, data_file, controller=None, debug=False)
    controller = PPOController(ppo, sim)
    sim.controller = controller
    
    # Run full rollout (rewards are computed in controller)
    sim.rollout()
    
    # Get final cost
    cost = sim.compute_cost()
    return cost['total_cost']

def train_ppo(bc_checkpoint_path='bc_pid_checkpoint.pth'):
    """Main PPO training loop"""
    print("\n" + "="*60)
    print("PPO Training from BC Initialization")
    print("="*60)
    
    # Load data files
    all_files = sorted(glob.glob("./data/*.csv"))
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Load model
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    
    # Create ActorCritic and load BC weights
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers).to(device)
    
    if os.path.exists(bc_checkpoint_path):
        print(f"\nLoading BC weights from {bc_checkpoint_path}")
        checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
        
        # Load actor weights from BC network
        bc_state_dict = checkpoint['model_state_dict']
        actor_state_dict = {
            'trunk.0.weight': bc_state_dict['trunk.0.weight'],
            'trunk.0.bias': bc_state_dict['trunk.0.bias'],
            'actor_layers.0.weight': bc_state_dict['actor_layers.0.weight'],
            'actor_layers.0.bias': bc_state_dict['actor_layers.0.bias'],
            'actor_layers.2.weight': bc_state_dict['actor_layers.2.weight'],
            'actor_layers.2.bias': bc_state_dict['actor_layers.2.bias'],
            'actor_layers.4.weight': bc_state_dict['actor_layers.4.weight'],
            'actor_layers.4.bias': bc_state_dict['actor_layers.4.bias'],
            'actor_mean.weight': bc_state_dict['actor_mean.weight'],
            'actor_mean.bias': bc_state_dict['actor_mean.bias'],
        }
        actor_critic.load_state_dict(actor_state_dict, strict=False)
        if 'evaluation' in checkpoint:
            print(f"BC train cost: {checkpoint['evaluation']['bc_train_cost']:.2f}")
            print(f"BC val cost: {checkpoint['evaluation']['bc_val_cost']:.2f}")
        else:
            print("BC weights loaded (evaluation metrics not available in checkpoint)")
    else:
        print(f"\n⚠️  BC checkpoint not found at {bc_checkpoint_path}")
        print("Training from random initialization...")
    
    # Create PPO agent
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print("\n" + "="*60)
    print("Training PPO...")
    print("="*60)
    
    best_cost = float('inf')
    
    for epoch in trange(n_epochs, desc="PPO Training"):
        # Sample files for this epoch
        epoch_files = np.random.choice(train_files, size=episodes_per_epoch, replace=False)
        
        # Collect trajectories
        episode_costs = []
        for data_file in epoch_files:
            cost = rollout_episode(ppo, model, data_file)
            episode_costs.append(cost)
        
        # PPO update
        loss, avg_adv, avg_ret = ppo.update()
        
        # Log
        mean_cost = np.mean(episode_costs)
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save(actor_critic.state_dict(), 'ppo_best.pth')
        
        # Early logging for first few epochs to catch divergence
        if epoch <= 2 or epoch % 5 == 0:
            tqdm.write(f"Epoch {epoch:3d} | Cost: {mean_cost:6.2f} | Best: {best_cost:6.2f} | Loss: {loss:.4f}")
        
        # Early stopping if cost explodes (>10x BC baseline)
        if mean_cost > 1000:
            tqdm.write(f"\n⚠️  Cost exploded to {mean_cost:.2f}. Stopping early.")
            tqdm.write("This suggests the policy is diverging. Check hyperparameters.")
            break
    
    print("\n" + "="*60)
    print(f"Training complete! Best cost: {best_cost:.2f}")
    print("Saved: ppo_best.pth")
    print("="*60)
    
    return actor_critic

if __name__ == '__main__':
    import torch.nn.functional as F
    train_ppo()

