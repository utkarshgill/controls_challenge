#!/usr/bin/env python3
"""
Pure PPO - Direct Control (No PID, No Residual)

The winner got <45 with PPO. This is that: pure PPO, full future plan, patience.

Strategy:
- 55D state: 5 current + 50 future curvatures
- MLP architecture (simple, like beautiful_lander)
- Direct steering control (no crutches)
- High initial exploration, prevent collapse
- Parallel rollouts (20 episodes simultaneously for decorrelation)
- Long training (1000 epochs)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from hashlib import md5
import glob

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, LAT_ACCEL_COST_MULTIPLIER, CONTROL_START_IDX
from controllers import BaseController

# Hyperparameters
state_dim, action_dim = 57, 1  # Added error_derivative and error_integral
hidden_dim = 128
trunk_layers, head_layers = 1, 3
lr = 3e-4  # Lower LR for stability
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
batch_size, K_epochs = 2048, 10
entropy_coef = 0.01  # Higher entropy bonus to maintain exploration
initial_log_std = -0.5  # σ ≈ 0.6 (high exploration)
min_log_std = -2.0  # Don't let σ go below 0.135
n_epochs = 1000  # Patience!
episodes_per_epoch = 20  # Parallel rollouts (match beautiful_lander scale)
max_episode_steps = 600  # Max steps for padding
STEER_RANGE = (-2.0, 2.0)

# State normalization (PID-informed state with temporal information)
OBS_SCALE = np.array(
    [10.0, 10.0, 100.0, 2.0, 0.03, 20.0, 1000.0] +  # [error, error_deriv, error_integral, lataccel, v_ego, a_ego, curv_now]
    [1000.0] * 50,                                    # 50 future curvatures
    dtype=np.float32
)

device = torch.device('cpu')

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers, initial_log_std=0.0):
        super().__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * initial_log_std)
        self.min_log_std = min_log_std
        
        # Critic head
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        # Clamp log_std to prevent collapse
        action_std = torch.clamp(self.log_std, min=self.min_log_std).exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value
    
    @torch.no_grad()
    def act(self, state, deterministic=False):
        state_normalized = state * OBS_SCALE
        state_tensor = torch.as_tensor(state_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action.squeeze(0).cpu().numpy(), state_tensor, raw_action

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef
        self.states, self.actions = [], []

    def select_action(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        # rewards: [T, N], state_values: [T, N], is_terminals: [T, N]
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
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
        # rewards: [T, N], dones: [T, N] numpy arrays
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(device)
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1).view(-1, rewards.size(1))  # [T, N]
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states, self.actions = [], []

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """57D: PID terms (3) + current state (4) + 50 future curvatures"""
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_derivative = (error - prev_error) / 0.1  # dt = 0.1s
    
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
    
    # [PID terms, current state, future]
    state_vec = [error, error_derivative, error_integral, current_lataccel, state.v_ego, state.a_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class PPOController(BaseController):
    def __init__(self, ppo):
        self.ppo = ppo
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error * 0.1  # dt = 0.1s
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)  # Anti-windup
        
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan, 
                               self.prev_error, self.error_integral)
        action = self.ppo.select_action(state_vec)
        
        self.prev_error = error
        return float(action[0]) if len(action.shape) > 0 else float(action)

def rollout_parallel(model, data_files, ppo, n_envs, epoch):
    """Run n_envs episodes in parallel, return [T, N] shaped arrays"""
    # Create simulators
    sims = [TinyPhysicsSimulator(model, f, controller=PPOController(ppo), debug=False) 
            for f in data_files[:n_envs]]
    
    # Break deterministic seeding: add epoch-dependent randomness
    # Each simulator gets deterministic seed from file path, but we add epoch variance
    for i, sim in enumerate(sims):
        base_seed = int(md5(sim.data_path.encode()).hexdigest(), 16) % 10**4
        epoch_seed = (base_seed + epoch * 1000 + i) % (2**31)
        np.random.seed(epoch_seed)
    
    # Find minimum episode length to ensure all envs run for exactly the same number of steps
    min_len = min(len(sim.data) for sim in sims)
    
    traj_rewards = []  # List of [N] arrays
    traj_dones = []    # List of [N] arrays
    
    # Run for exactly min_len steps
    while all(sim.step_idx < min_len for sim in sims):
        step_rewards = []
        step_dones = []
        
        for sim in sims:
            sim.step()
            
            if sim.step_idx >= CONTROL_START_IDX:
                error = sim.target_lataccel_history[-1] - sim.current_lataccel_history[-1]
                jerk = 0.0
                if len(sim.current_lataccel_history) >= 2:
                    jerk = (sim.current_lataccel_history[-1] - sim.current_lataccel_history[-2]) / 0.1
                
                lat_cost = (error ** 2) * LAT_ACCEL_COST_MULTIPLIER
                jerk_cost = (jerk ** 2)
                reward = -(lat_cost + jerk_cost) / 100.0
            else:
                reward = 0.0
            
            step_rewards.append(reward)
            step_dones.append(0.0)  # Mark as not done yet
        
        traj_rewards.append(step_rewards)
        traj_dones.append(step_dones)
    
    # Mark final step as done for all envs
    if traj_dones:
        traj_dones[-1] = [1.0] * n_envs
    
    # Compute costs (finish remaining steps with a dummy controller to avoid collecting more states)
    class DummyController(BaseController):
        """Controller that uses PPO for actions but doesn't log states"""
        def __init__(self, actor_critic):
            self.actor_critic = actor_critic
            self.prev_error = 0.0
            self.error_integral = 0.0
            
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            error = target_lataccel - current_lataccel
            self.error_integral += error * 0.1
            self.error_integral = np.clip(self.error_integral, -10.0, 10.0)  # Anti-windup
            
            state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                                   self.prev_error, self.error_integral)
            action, _, _ = self.actor_critic.act(state_vec, deterministic=False)
            
            self.prev_error = error
            return float(action[0]) if len(action.shape) > 0 else float(action)
    
    costs = []
    for sim in sims:
        # Switch to dummy controller for remaining steps
        sim.controller = DummyController(ppo.actor_critic)
        while sim.step_idx < len(sim.data):
            sim.step()
        costs.append(sim.compute_cost()['total_cost'])
    
    return traj_rewards, traj_dones, costs

def evaluate_policy(actor_critic, model, data_files, n_eval=20):
    """Evaluate policy deterministically"""
    costs = []
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        class EvalController(BaseController):
            def __init__(self):
                self.prev_error = 0.0
                self.error_integral = 0.0
                
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                error = target_lataccel - current_lataccel
                self.error_integral += error * 0.1
                self.error_integral = np.clip(self.error_integral, -10.0, 10.0)  # Anti-windup
                
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                                       self.prev_error, self.error_integral)
                action, _, _ = actor_critic.act(state_vec, deterministic=True)
                
                self.prev_error = error
                return float(action[0]) if len(action.shape) > 0 else float(action)
        
        sim = TinyPhysicsSimulator(model, data_file, controller=EvalController(), debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)

def train():
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    all_files = sorted(glob.glob("./data/*.csv"))
    print(f"Total files available: {len(all_files)}")
    
    # Train/val split: 90% train, 10% val
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers, 
                                initial_log_std=initial_log_std).to(device)
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print("Pure PPO Training - Direct Control")
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    print(f"Epochs: {n_epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Initial σ: {np.exp(initial_log_std):.3f}, Min σ: {np.exp(min_log_std):.3f}, Target: <45")
    print("="*60)
    
    all_costs = []
    best_cost = float('inf')
    pbar = trange(n_epochs, desc="Training", unit='epoch')
    
    for epoch in range(n_epochs):
        # Parallel rollout: collect from multiple episodes simultaneously
        # Use replace=False to ensure no duplicate files per epoch
        sampled_files = np.random.choice(train_files, size=episodes_per_epoch, replace=False)
        traj_rewards, traj_dones, epoch_costs = rollout_parallel(model, sampled_files, ppo, episodes_per_epoch, epoch)
        
        # PPO update (rewards/dones are list of [N] arrays)
        ppo.update(traj_rewards, traj_dones)
        
        # Track
        all_costs.extend(epoch_costs)
        mean_cost = np.mean(epoch_costs)
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save(actor_critic.state_dict(), "ppo_pure_best.pth")
        
        pbar.update(1)
        
        if epoch % 10 == 0:
            recent_cost = np.mean(all_costs[-80:]) if len(all_costs) >= 80 else mean_cost
            std_val = torch.clamp(actor_critic.log_std, min=actor_critic.min_log_std).exp().item()
            eval_cost = evaluate_policy(actor_critic, model, val_files, n_eval=20) if epoch % 20 == 0 else mean_cost
            pbar.write(f"Epoch {epoch:4d} | Train: {mean_cost:.2f} | Recent80: {recent_cost:.2f} | "
                      f"Val: {eval_cost:.2f} | Best: {best_cost:.2f} | σ: {std_val:.3f}")
    
    pbar.close()
    print("="*60)
    print(f"Final (last 80): {np.mean(all_costs[-80:]):.2f}")
    print(f"Best: {best_cost:.2f}")
    
    torch.save(actor_critic.state_dict(), "ppo_pure_final.pth")
    print("Models saved: ppo_pure_best.pth, ppo_pure_final.pth")
    
    return actor_critic

if __name__ == '__main__':
    train()

