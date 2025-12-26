#!/usr/bin/env python3
"""
PPO Training with Parallel Environments (Gym Wrapper)

Based on beautiful_lander.py's battle-tested parallel rollout pattern.
Uses AsyncVectorEnv for 8x speedup over sequential rollouts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import glob
import os
import gymnasium as gym

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, STEER_RANGE

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

# PPO hyperparameters (FIXED to match beautiful_lander.py)
n_epochs = 100
num_envs = 8  # Parallel environments
steps_per_epoch = 10_000  # Total steps across all envs per epoch (~25 episodes)
batch_size = 10_000  # Match reference
K_epochs = 10  # Match reference
lr = 1e-3  # FIXED: was 1e-5 (100× too slow!)
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2  # Match reference
entropy_coef = 0.001  # FIXED: was 0.0 (no exploration!)

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

class TinyPhysicsGymEnv(gym.Env):
    """
    Gymnasium wrapper for TinyPhysicsSimulator.
    Enables parallel rollouts with AsyncVectorEnv.
    """
    def __init__(self, model_path, data_files):
        super().__init__()
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data_files = data_files
        self.sim = None
        
        # PID-like state tracking
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = 0.0
        self.episode_cost = 0.0
        
        # Gym spaces
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (56,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset to a random file from the dataset"""
        super().reset(seed=seed)
        
        # Pick random file
        data_file = np.random.choice(self.data_files)
        self.sim = TinyPhysicsSimulator(self.model, data_file, controller=None, debug=False)
        
        # Fast-forward through warmup period (no control, just sim steps)
        for _ in range(CONTROL_START_IDX):
            state, target, future = self.sim.get_state_target_futureplan(self.sim.step_idx)
            # No control during warmup - set steer to 0
            self.sim.current_steer = 0.0
            self.sim.sim_step(self.sim.step_idx)
            self.sim.step_idx += 1
        
        # Reset state tracking
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = self.sim.current_lataccel
        self.episode_cost = 0.0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step with the given action"""
        # Get current state
        state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        current_lataccel = self.sim.current_lataccel
        
        # Apply action (convert from normalized range)
        action_value = float(np.clip(action[0], -2.0, 2.0))
        self.sim.current_steer = action_value
        
        # Step simulator (bypass controller callback - we set current_steer directly above)
        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1
        
        # Compute reward - FIXED to match evaluation cost exactly!
        # Eval: total = (50 × lat_cost) + jerk_cost
        lat_cost = (current_lataccel - target_lataccel) ** 2
        jerk_cost = ((current_lataccel - self.prev_lataccel) / 0.1) ** 2  # dt = 0.1
        reward = -(50 * lat_cost + jerk_cost) / 100.0  # FIXED: was equal weight!
        
        # Accumulate episode cost (match eval: 50:1 weighting!)
        self.episode_cost += (50 * lat_cost + jerk_cost)
        
        # Update internal state with anti-windup
        error = target_lataccel - current_lataccel
        self.error_integral = np.clip(self.error_integral + error, -14, 14)  # Anti-windup
        self.prev_error = error
        self.prev_lataccel = current_lataccel
        
        # Check if done
        done = self.sim.step_idx >= len(self.sim.data)
        truncated = False
        
        # Get next observation (zeros if done)
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(56, dtype=np.float32)
        
        # Info includes episode cost when done
        info = {}
        if done:
            info['episode_cost'] = self.episode_cost
        
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        """Build current observation vector"""
        state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        current_lataccel = self.sim.current_lataccel
        return build_state(target_lataccel, current_lataccel, state, future_plan,
                          self.prev_error, self.error_integral)

def make_env(model_path, data_files):
    """Factory function for creating TinyPhysicsGymEnv (needed for AsyncVectorEnv)"""
    def _init():
        return TinyPhysicsGymEnv(model_path, data_files)
    return _init

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
        # FIXED: Match reference (std=1.0, not 0.05!)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
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
    def act(self, state, deterministic=False):
        """Act function for vectorized environments"""
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
        action_mean, action_std, _ = self(state_tensor)
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action.cpu().numpy()

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef
        self.states, self.actions = [], []

    def __call__(self, state):
        """Policy function for rollout (stores states/actions)"""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
            action_mean, action_std, _ = self.actor_critic(state_tensor)
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
            action = torch.tanh(raw_action) * STEER_RANGE[1]
            
            self.states.append(state_tensor)
            self.actions.append(raw_action)
            
            return action.cpu().numpy()

    def compute_advantages(self, rewards, state_values, is_terminals):
        """GAE computation (from beautiful_lander.py)"""
        T, N = rewards.shape
        advantages, gae = torch.zeros_like(rewards), torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        """Compute PPO losses (from beautiful_lander.py)"""
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, 
                                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().sum(-1).mean()
        return actor_loss + critic_loss - self.entropy_coef * entropy
    
    def update(self, rewards, dones):
        """PPO update (from beautiful_lander.py pattern)"""
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(device)
            old_states, old_actions = torch.cat(self.states), torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1).view(-1, rewards.size(1))
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)  # Match reference
                self.optimizer.step()
        
        self.states, self.actions = [], []

def rollout(env, policy, num_steps):
    """
    Parallel rollout (from beautiful_lander.py).
    Collects exactly num_steps across all parallel environments.
    """
    states, _ = env.reset()
    traj_rewards, traj_dones = [], []
    episode_costs = []
    step_count = 0
    
    pbar = tqdm(total=num_steps, desc="Rollout", leave=False)
    while step_count < num_steps:
        actions = policy(states)
        states, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        traj_rewards.append(rewards)
        traj_dones.append(dones)
        step_count += env.num_envs
        pbar.update(env.num_envs)
        
        # Extract episode costs - AsyncVectorEnv puts episode_cost directly in infos dict
        # infos = {'episode_cost': array([...]), '_episode_cost': array([...])}
        if isinstance(infos, dict) and 'episode_cost' in infos:
            # Get costs for all envs that finished this step
            costs_array = np.array(infos['episode_cost'])
            for i, done_flag in enumerate(dones):
                if done_flag and not np.isnan(costs_array[i]):
                    episode_costs.append(float(costs_array[i]))
    
    pbar.close()
    return traj_rewards, traj_dones, episode_costs

def train_ppo(bc_checkpoint_path='bc_pid_checkpoint.pth'):
    """Main PPO training loop with parallel environments"""
    print("\n" + "="*60)
    print("PPO Training with Parallel Environments")
    print("="*60)
    
    # Load data files
    all_files = sorted(glob.glob("./data/*.csv"))
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print(f"Parallel envs: {num_envs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Create parallel environments
    model_path = "./models/tinyphysics.onnx"
    env = gym.vector.AsyncVectorEnv([
        make_env(model_path, train_files) for _ in range(num_envs)
    ])
    
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
        print("✅ BC weights loaded into actor")
    else:
        print(f"\n⚠️  BC checkpoint not found at {bc_checkpoint_path}")
        print("Training from random initialization...")
    
    # Create PPO agent
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print("\n" + "="*60)
    print("Training PPO...")
    print("="*60)
    
    best_cost = float('inf')
    all_episode_costs = []
    
    pbar = trange(n_epochs, desc="PPO Training")
    for epoch in pbar:
        # Collect trajectories
        rewards, dones, episode_costs = rollout(env, ppo, num_steps=steps_per_epoch)
        
        # PPO update
        ppo.update(rewards, dones)
        
        # Track costs
        if episode_costs:
            all_episode_costs.extend(episode_costs)
            mean_cost = np.mean(episode_costs)
            if mean_cost < best_cost:
                best_cost = mean_cost
                torch.save(actor_critic.state_dict(), 'ppo_parallel_best.pth')
            
            # Logging every 5 epochs
            if epoch % 5 == 0:
                pbar.write(f"Epoch {epoch:3d} | Cost: {mean_cost:6.2f} | Best: {best_cost:6.2f} | Episodes: {len(episode_costs)}")
            
            # Early stopping if cost explodes (only after learning starts)
            if epoch > 20 and mean_cost > 10 * best_cost:
                pbar.write(f"\n⚠️  Cost exploded to {mean_cost:.2f} (10× best of {best_cost:.2f}). Stopping early.")
                break
        else:
            # Debug: no episodes completed this epoch
            if epoch % 5 == 0:
                pbar.write(f"Epoch {epoch:3d} | ⚠️  No episodes completed")
    
    env.close()
    pbar.close()
    
    print("\n" + "="*60)
    print(f"Training complete! Best cost: {best_cost:.2f}")
    print("Saved: ppo_parallel_best.pth")
    print("="*60)
    
    return actor_critic

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO')
    parser.add_argument('--init-from', type=str, default='bc_pid_checkpoint.pth',
                       help='Path to BC checkpoint for initialization')
    args = parser.parse_args()
    
    train_ppo(bc_checkpoint_path=args.init_from)

