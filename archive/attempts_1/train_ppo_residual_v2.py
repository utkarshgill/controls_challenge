#!/usr/bin/env python3
"""
PPO Residual Learning on PID baseline - V2: FULL FUTURE PLAN

Strategy:
1. PID baseline (~73 cost)
2. PPO learns residual corrections
3. FULL 55D state: 5 current + 50 future curvatures
4. No compression - let network learn what matters

Changes from V1:
- State: 10D → 55D (all future curvatures, no summarization)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import glob
import os

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, LAT_ACCEL_COST_MULTIPLIER, CONTROL_START_IDX
from controllers import BaseController, pid

# Hyperparameters
state_dim, action_dim = 55, 1  # 55D: 5 current + 50 future curvatures
hidden_dim = 128
trunk_layers, head_layers = 1, 3
lr = 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
batch_size, K_epochs = 2048, 10
entropy_coef = 0.001
initial_log_std = -2.3  # σ ≈ 0.1
n_epochs = 300
episodes_per_epoch = 8
STEER_RANGE = (-2.0, 2.0)
RESIDUAL_SCALE = 0.5

# State normalization (55D)
OBS_SCALE = np.array(
    [10.0, 2.0, 0.03, 20.0, 1000.0] +  # 5 current features
    [1000.0] * 50,                      # 50 future curvatures
    dtype=np.float32
)

device = torch.device('cpu')

def tanh_log_prob(raw_action, dist):
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
        state_normalized = state * OBS_SCALE
        state_tensor = torch.as_tensor(state_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        residual = torch.tanh(raw_action) * RESIDUAL_SCALE
        return residual.squeeze(0).cpu().numpy(), state_tensor, raw_action

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef
        self.states, self.actions = [], []

    def select_action(self, state):
        residual_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return residual_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
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
        with torch.no_grad():
            rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.array(dones), dtype=torch.float32).to(device)
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1)
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states, self.actions = [], []

def build_state(target_lataccel, current_lataccel, state, future_plan):
    """
    Full 55D state: ALL future information, no compression
    
    [error, current_lataccel, v_ego, a_ego, curv_now] + 50 future curvatures
    """
    eps = 1e-6
    error = target_lataccel - current_lataccel
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    # Compute ALL 50 future curvatures
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    # Pad if needed
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, current_lataccel, state.v_ego, state.a_ego, curv_now] + future_curvs
    
    return np.array(state_vec, dtype=np.float32)

class ResidualPPOController(BaseController):
    """PID + PPO residual controller"""
    def __init__(self, ppo):
        self.ppo = ppo
        self.pid = pid.Controller()
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        pid_action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        residual = self.ppo.select_action(state_vec)
        residual_scalar = float(residual[0]) if len(residual.shape) > 0 else float(residual)
        combined = np.clip(pid_action + residual_scalar, STEER_RANGE[0], STEER_RANGE[1])
        return combined

def rollout_episode(model, data_file, ppo):
    """Run one episode with PID+PPO residual"""
    sim = TinyPhysicsSimulator(model, data_file, controller=ResidualPPOController(ppo), debug=False)
    
    episode_rewards = []
    episode_dones = []
    
    while sim.step_idx < len(sim.data):
        sim.step()
        
        if sim.step_idx >= CONTROL_START_IDX:
            error = sim.target_lataccel_history[-1] - sim.current_lataccel_history[-1]
            jerk = 0.0
            if len(sim.current_lataccel_history) >= 2:
                jerk = (sim.current_lataccel_history[-1] - sim.current_lataccel_history[-2]) / 0.1
            
            lat_cost = (error ** 2) * LAT_ACCEL_COST_MULTIPLIER
            jerk_cost = (jerk ** 2)
            reward = -(lat_cost + jerk_cost)
        else:
            reward = 0.0
        
        episode_rewards.append(reward)
        episode_dones.append(0.0)
    
    episode_dones[-1] = 1.0
    cost = sim.compute_cost()['total_cost']
    
    return episode_rewards, episode_dones, cost

def evaluate_policy(actor_critic, model, data_files, n_eval=20):
    """Evaluate PID+PPO residual policy"""
    costs = []
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        class EvalController(BaseController):
            def __init__(self):
                self.pid = pid.Controller()
            
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                pid_action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
                residual, _, _ = actor_critic.act(state_vec, deterministic=True)
                residual_scalar = float(residual[0]) if len(residual.shape) > 0 else float(residual)
                return np.clip(pid_action + residual_scalar, STEER_RANGE[0], STEER_RANGE[1])
        
        sim = TinyPhysicsSimulator(model, data_file, controller=EvalController(), debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)

def evaluate_pid_baseline(model, data_files, n_eval=20):
    """Check PID baseline performance"""
    costs = []
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        controller = pid.Controller()
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    return np.mean(costs)

def train():
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    train_files = sorted(glob.glob("./data/*.csv"))[:100]
    
    # Check PID baseline first
    print("Checking PID baseline...")
    pid_baseline = evaluate_pid_baseline(model, train_files, n_eval=20)
    print(f"PID Baseline: {pid_baseline:.2f}")
    print()
    
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers, 
                                initial_log_std=initial_log_std).to(device)
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print("Training PPO Residual V2 - FULL FUTURE PLAN (55D)")
    print(f"Strategy: action = pid + residual, 5 current + 50 future curvatures")
    print(f"Files: {len(train_files)}, Epochs: {n_epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Initial σ: {np.exp(initial_log_std):.3f}, Target: <45, Baseline: {pid_baseline:.2f}")
    print("="*60)
    
    all_costs = []
    best_cost = float('inf')
    pbar = trange(n_epochs, desc="Training", unit='epoch')
    
    for epoch in range(n_epochs):
        # Collect N episodes
        all_rewards = []
        all_dones = []
        epoch_costs = []
        
        for _ in range(episodes_per_epoch):
            data_file = np.random.choice(train_files)
            ep_rewards, ep_dones, cost = rollout_episode(model, data_file, ppo)
            all_rewards.extend(ep_rewards)
            all_dones.extend(ep_dones)
            epoch_costs.append(cost)
        
        # PPO update
        ppo.update(all_rewards, all_dones)
        
        # Track costs
        all_costs.extend(epoch_costs)
        mean_cost = np.mean(epoch_costs)
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save(actor_critic.state_dict(), "ppo_residual_v2_best.pth")
        
        pbar.update(1)
        
        if epoch % 5 == 0:
            recent_cost = np.mean(all_costs[-50:]) if len(all_costs) >= 50 else mean_cost
            std_val = actor_critic.log_std.exp().item()
            eval_cost = evaluate_policy(actor_critic, model, train_files, n_eval=10) if epoch % 10 == 0 else mean_cost
            improvement = pid_baseline - eval_cost
            pbar.write(f"Epoch {epoch:3d} | Train: {mean_cost:.2f} | Recent50: {recent_cost:.2f} | "
                      f"Eval: {eval_cost:.2f} | Δ from PID: {improvement:+.2f} | σ: {std_val:.3f}")
    
    pbar.close()
    print("="*60)
    print(f"PID Baseline: {pid_baseline:.2f}")
    print(f"Final (last 50): {np.mean(all_costs[-50:]):.2f}")
    print(f"Best: {best_cost:.2f}")
    print(f"Improvement: {pid_baseline - best_cost:+.2f}")
    
    torch.save(actor_critic.state_dict(), "ppo_residual_v2_final.pth")
    print("Models saved: ppo_residual_v2_best.pth, ppo_residual_v2_final.pth")
    
    return actor_critic

if __name__ == '__main__':
    train()

