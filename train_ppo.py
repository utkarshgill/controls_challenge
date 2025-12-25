#!/usr/bin/env python3
"""
PPO training for TinyPhysics control challenge.
Mirrors the structure of beautiful_lander.py for reliability.
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
from controllers import BaseController

# Hyperparameters
state_dim, action_dim = 10, 1  # 10D smart summary state
hidden_dim = 128
trunk_layers, head_layers = 1, 3
lr = 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
batch_size, K_epochs = 2048, 10
entropy_coef = 0.001
initial_log_std = -1.2  # σ ≈ 0.3 (low exploration for autoregressive control)
n_epochs = 500  # More epochs needed
episodes_per_epoch = 8
STEER_RANGE = (-2.0, 2.0)

# State normalization (like LunarLander's OBS_SCALE)
# Scale factors to normalize state to ~[-1, 1] range
OBS_SCALE = np.array([
    10.0,    # error (±0.1 → ±1)
    2.0,     # lataccel (±0.5 → ±1)
    0.03,    # v_ego (33 → 1)
    20.0,    # a_ego (±0.05 → ±1)
    1000.0,  # curv_now (0.0001 → 0.1)
    1000.0,  # next1
    1000.0,  # next2
    1000.0,  # next3
    1000.0,  # future_max
    1000.0,  # future_mean
], dtype=np.float32)

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
        # Normalize state (like LunarLander)
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
    Smart summary: 10D state (like LunarLander's low-dim design)
    
    [error, current_lataccel, v_ego, a_ego, current_curv,
     next_curv_1, next_curv_2, next_curv_3,
     future_max, future_mean]
    """
    eps = 1e-6
    error = target_lataccel - current_lataccel
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    # Compute all future curvatures
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    # Near-term: next 3 steps (most critical for control)
    near_curvs = future_curvs[:3] if len(future_curvs) >= 3 else future_curvs + [0.0] * (3 - len(future_curvs))
    
    # Statistics: hardest turn + average difficulty
    if len(future_curvs) > 0:
        future_max = max(abs(c) for c in future_curvs)
        future_mean = np.mean(future_curvs)
    else:
        future_max = 0.0
        future_mean = 0.0
    
    return np.array([
        error,
        current_lataccel,
        state.v_ego,
        state.a_ego,
        curv_now,
        near_curvs[0],
        near_curvs[1],
        near_curvs[2],
        future_max,
        future_mean
    ], dtype=np.float32)

class PPOController(BaseController):
    def __init__(self, ppo):
        self.ppo = ppo
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        action = self.ppo.select_action(state_vec)
        return float(action[0]) if len(action.shape) > 0 else float(action)

def rollout_episode(model, data_file, ppo):
    """Run one episode, collect trajectory"""
    sim = TinyPhysicsSimulator(model, data_file, controller=PPOController(ppo), debug=False)
    
    episode_rewards = []
    episode_dones = []
    
    while sim.step_idx < len(sim.data):
        sim.step()
        
        # Compute reward
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
    
    episode_dones[-1] = 1.0  # Mark last step as terminal
    cost = sim.compute_cost()['total_cost']
    
    return episode_rewards, episode_dones, cost

def evaluate_policy(actor_critic, model, data_files, n_eval=20):
    """Evaluate policy deterministically"""
    costs = []
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        class EvalController(BaseController):
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
                action, _, _ = actor_critic.act(state_vec, deterministic=True)
                return float(action[0]) if len(action.shape) > 0 else float(action)
        
        sim = TinyPhysicsSimulator(model, data_file, controller=EvalController(), debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)

def train():
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    train_files = sorted(glob.glob("./data/*.csv"))[:100]
    
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers, 
                                initial_log_std=initial_log_std).to(device)
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print("Training PPO on TinyPhysics - Smart Summary (10D) + Normalized")
    print(f"Files: {len(train_files)}, Epochs: {n_epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Initial σ: {np.exp(initial_log_std):.3f}, OBS_SCALE: ON")
    print("="*60)
    
    all_costs = []
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
        pbar.update(1)
        
        if epoch % 5 == 0:
            mean_cost = np.mean(epoch_costs)
            recent_cost = np.mean(all_costs[-50:]) if len(all_costs) >= 50 else mean_cost
            std_val = actor_critic.log_std.exp().item()
            eval_cost = evaluate_policy(actor_critic, model, train_files, n_eval=10) if epoch % 10 == 0 else mean_cost
            pbar.write(f"Epoch {epoch:3d} | Train: {mean_cost:.2f} | Recent50: {recent_cost:.2f} | Eval: {eval_cost:.2f} | σ: {std_val:.3f}")
    
    pbar.close()
    print("="*60)
    print(f"Final cost (last 50): {np.mean(all_costs[-50:]):.2f}")
    print(f"Best: {np.min(all_costs):.2f}")
    
    torch.save(actor_critic.state_dict(), "ppo_policy.pth")
    print("Model saved to ppo_policy.pth")
    
    return actor_critic

if __name__ == '__main__':
    train()

