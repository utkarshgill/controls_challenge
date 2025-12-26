#!/usr/bin/env python3
"""
Experiment 005: PPO with Data Augmentation
Fix asymmetry by training on original + horizontally flipped data
Architecture based on beautiful_lander.py
State: 55D = [error, roll, v_ego, a_ego, current, future×50]
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import glob
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, STEER_RANGE

# Hyperparameters (from beautiful_lander.py)
STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
TRUNK_LAYERS = 1
HEAD_LAYERS = 3
LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
ENTROPY_COEF = 0.001
K_EPOCHS = 10
BATCH_SIZE = 10_000
NUM_ENVS = 8
STEPS_PER_EPOCH = 10_000
MAX_EPOCHS = 100
LOG_INTERVAL = 5

# Normalization (from exp003)
NORM_SCALE = np.array([
    2.0, 5.0, 0.05, 5.0, 2.0,  # error, roll, v_ego, a_ego, current
    *[2.0]*50  # future_lataccel
], dtype=np.float32)

device = torch.device('cpu')

def build_state(target_lataccel, current_lataccel, state, future_plan, flip=False):
    """Build normalized 55D state with optional horizontal flip for augmentation"""
    error = target_lataccel - current_lataccel
    future_lataccel = np.array(future_plan.lataccel)
    
    if len(future_lataccel) == 0:
        future_lataccel = np.zeros(50, dtype=np.float32)
    elif len(future_lataccel) < 50:
        future_lataccel = np.pad(future_lataccel, (0, 50 - len(future_lataccel)), 'edge')
    else:
        future_lataccel = future_lataccel[:50]
    
    # Data augmentation: horizontal flip (negate all lateral values)
    if flip:
        error = -error
        current_lataccel = -current_lataccel
        future_lataccel = -future_lataccel
        roll_lataccel = -state.roll_lataccel
    else:
        roll_lataccel = state.roll_lataccel
    
    state_vec = np.array([
        error, roll_lataccel, state.v_ego, state.a_ego, current_lataccel,
        *future_lataccel
    ], dtype=np.float32)
    
    return state_vec * NORM_SCALE

def tanh_log_prob(raw_action, dist):
    """Log probability with tanh squashing (from beautiful_lander.py)"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    """ActorCritic with shared trunk (from beautiful_lander.py)"""
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
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
    def act(self, state, deterministic=False, return_internals=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return (action.cpu().numpy(), state_tensor, raw_action) if return_internals else action.cpu().numpy()

class TinyPhysicsGymEnv(gym.Env):
    """Gym wrapper for TinyPhysics with data augmentation"""
    def __init__(self, model_path, data_files):
        super().__init__()
        self.model_path = model_path
        self.data_files = data_files
        self.current_file_idx = 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,))
        self.action_space = gym.spaces.Box(-STEER_RANGE[1], STEER_RANGE[1], shape=(ACTION_DIM,))
        self.reset()
    
    def reset(self, **kwargs):
        file_path = self.data_files[self.current_file_idx % len(self.data_files)]
        self.current_file_idx += 1
        
        model = TinyPhysicsModel(self.model_path, debug=False)
        self.sim = TinyPhysicsSimulator(model, file_path, controller=None, debug=False)
        self.episode_cost = 0.0
        
        # Data augmentation: 50% chance to flip this episode
        self.flip = np.random.random() < 0.5
        
        state, target, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        obs = build_state(target, self.sim.current_lataccel, state, future_plan, flip=self.flip)
        return obs, {}
    
    def step(self, action):
        # Flip action back if episode is flipped
        action_to_apply = -action[0] if self.flip else action[0]
        action_clipped = np.clip(action_to_apply, STEER_RANGE[0], STEER_RANGE[1])
        self.sim.action_history.append(action_clipped)
        
        # Simulate
        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1
        
        # Check if done
        done = self.sim.step_idx >= len(self.sim.data) - 50
        
        # Get next observation (with same flip as episode)
        if not done:
            state, target, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            self.sim.futureplan = future_plan
            obs = build_state(target, self.sim.current_lataccel, state, future_plan, flip=self.flip)
        else:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
        
        # Compute reward (negative cost) - same regardless of flip
        if len(self.sim.current_lataccel_history) > 1:
            lat_cost = (self.sim.target_lataccel_history[-1] - self.sim.current_lataccel) ** 2
            jerk = (self.sim.current_lataccel_history[-1] - self.sim.current_lataccel_history[-2]) / 0.1
            jerk_cost = jerk ** 2
            step_cost = (50 * lat_cost + jerk_cost)
            self.episode_cost += step_cost
            reward = -step_cost / 100.0  # Scale down
        else:
            reward = 0.0
        
        info = {}
        if done:
            info['episode_cost'] = self.episode_cost
        
        return obs, reward, done, False, info

class PPO:
    """PPO implementation (from beautiful_lander.py)"""
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.states = []
        self.actions = []
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
    
    def __call__(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False, return_internals=True)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np
    
    def compute_advantages(self, rewards, state_values, is_terminals):
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
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(device)
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1).view(-1, rewards.size(1))
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states = []
        self.actions = []

def rollout(env, policy, num_steps):
    """Collect trajectories"""
    states, _ = env.reset()
    traj_rewards = []
    traj_dones = []
    ep_costs = []
    step_count = 0
    
    while step_count < num_steps:
        states, rewards, terminated, truncated, infos = env.step(policy(states))
        traj_rewards.append(rewards)
        traj_dones.append(np.logical_or(terminated, truncated))
        step_count += env.num_envs
        
        # Extract episode costs
        if 'episode_cost' in infos:
            costs = infos['episode_cost'][traj_dones[-1]]
            ep_costs.extend(costs[~np.isnan(costs)])
    
    return traj_rewards, traj_dones, ep_costs

def main():
    print("\n" + "="*60)
    print("Experiment 005: PPO with Data Augmentation")
    print("="*60)
    print(f"State: {STATE_DIM}D normalized")
    print(f"Data Aug: 50% horizontal flip (symmetric training)")
    print(f"Architecture: Trunk({TRUNK_LAYERS}) + Heads({HEAD_LAYERS}) × {HIDDEN_DIM}")
    print(f"Envs: {NUM_ENVS}, Steps/epoch: {STEPS_PER_EPOCH}")
    print(f"LR: {LR}, Entropy: {ENTROPY_COEF}, K_epochs: {K_EPOCHS}")
    print("="*60 + "\n")
    
    # Setup
    model_path = "../../models/tinyphysics.onnx"
    all_files = sorted(glob.glob("../../data/*.csv"))[:1000]
    
    actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS).to(device)
    ppo = PPO(actor_critic, LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, ENTROPY_COEF)
    
    # Create parallel envs
    env = gym.vector.AsyncVectorEnv([
        lambda: TinyPhysicsGymEnv(model_path, all_files) for _ in range(NUM_ENVS)
    ])
    
    best_cost = float('inf')
    pbar = trange(MAX_EPOCHS, desc="Training PPO")
    
    for epoch in pbar:
        # Train
        rewards, dones, ep_costs = rollout(env, ppo, STEPS_PER_EPOCH)
        ppo.update(rewards, dones)
        
        # Log
        if len(ep_costs) > 0:
            mean_cost = np.mean(ep_costs)
            if mean_cost < best_cost:
                best_cost = mean_cost
                torch.save(actor_critic.state_dict(), 'results/checkpoints/ppo_best.pth')
            
            if epoch % LOG_INTERVAL == 0:
                std = actor_critic.log_std.exp().detach().cpu().numpy()[0]
                pbar.write(f"Epoch {epoch:3d} | Episodes: {len(ep_costs):3d} | Cost: {mean_cost:7.2f} | Best: {best_cost:7.2f} | σ: {std:.3f}")
            
            # Early stopping if explodes
            if epoch > 20 and mean_cost > 10 * best_cost:
                pbar.write(f"\n⚠️  Cost exploded to {mean_cost:.2f}. Stopping.")
                break
        else:
            if epoch % LOG_INTERVAL == 0:
                pbar.write(f"Epoch {epoch:3d} | ⚠️  No episodes completed")
    
    env.close()
    pbar.close()
    
    print("\n" + "="*60)
    print(f"Training complete! Best cost: {best_cost:.2f}")
    print("="*60)
    print(f"✅ Saved: results/checkpoints/ppo_best.pth")
    print(f"✅ Next: Create controllers/ppo.py and evaluate")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

