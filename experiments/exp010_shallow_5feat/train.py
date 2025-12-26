"""
exp010: Shallow network (5 features: add future preview)
State: [error, error_integral, error_derivative, v_ego, future_severity]
Network: Linear(5, 16) -> Tanh -> Linear(16, 1)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_state(target_lataccel, current_lataccel, state, prev_error, error_integral, future_plan):
    error = target_lataccel - current_lataccel
    error_derivative = error - prev_error
    v_ego = state.v_ego
    # Future severity: mean of next 1 second (10 steps) of abs lateral accel
    future_lataccel = np.array(future_plan.lataccel[:10]) if len(future_plan.lataccel) >= 10 else np.array(future_plan.lataccel)
    future_severity = np.mean(np.abs(future_lataccel)) if len(future_lataccel) > 0 else 0.0
    return np.array([error, error_integral, error_derivative, v_ego, future_severity], dtype=np.float32)

class ShallowActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=16):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std = -20
    
    def forward(self, x):
        action_mean = self.actor(x)
        value = self.critic(x)
        return action_mean, value
    
    def act(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_mean, _ = self.forward(state)
            if deterministic:
                return torch.tanh(action_mean).cpu().numpy()[0]
            std = torch.clamp(self.log_std, min=self.min_log_std).exp()
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            action = torch.tanh(action_sample)
            return action.cpu().numpy()[0]

class TinyPhysicsGymEnv(gym.Env):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.sim = None
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_cost = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        model_path = '/Users/engelbart/Desktop/stuff/controls_challenge/models/tinyphysics.onnx'
        model = TinyPhysicsModel(model_path, debug=False)
        self.sim = TinyPhysicsSimulator(model, self.data_path, controller=None, debug=False)
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_cost = 0.0
        
        state, target, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        current_lataccel = self.sim.current_lataccel
        
        obs = build_state(target, current_lataccel, state, self.prev_error, self.error_integral, future_plan)
        return obs, {}
    
    def step(self, action):
        action_clipped = np.clip(action[0], -2.0, 2.0)
        self.sim.action_history.append(action_clipped)
        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1
        
        done = self.sim.step_idx >= len(self.sim.data) - 50
        
        if not done:
            state, target, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
            current_lataccel = self.sim.current_lataccel
            
            error = target - current_lataccel
            self.error_integral += error
            self.error_integral = np.clip(self.error_integral, -14, 14)
            
            obs = build_state(target, current_lataccel, state, self.prev_error, self.error_integral, future_plan)
            self.prev_error = error
        else:
            obs = np.zeros(5, dtype=np.float32)
        
        if len(self.sim.current_lataccel_history) > 1:
            lat_cost = (self.sim.target_lataccel_history[-1] - self.sim.current_lataccel) ** 2
            jerk = (self.sim.current_lataccel_history[-1] - self.sim.current_lataccel_history[-2]) / 0.1
            jerk_cost = jerk ** 2
            step_cost = (50 * lat_cost + jerk_cost)
            self.episode_cost += step_cost
            reward = -step_cost / 100.0
        else:
            reward = 0.0
        
        info = {}
        if done:
            info['episode_cost'] = self.episode_cost
        
        return obs, reward, done, False, info

def tanh_log_prob(action_mean, log_std, action_tanh):
    std = log_std.exp()
    action_raw = torch.atanh(torch.clamp(action_tanh, -0.9999, 0.9999))
    log_prob = -0.5 * ((action_raw - action_mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
    log_prob -= torch.log(1 - action_tanh ** 2 + 1e-6)
    return log_prob.sum(dim=-1)

def collect_rollout(env, actor_critic, steps=1000):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    obs, _ = env.reset()
    episode_costs = []
    
    for _ in range(steps):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_mean, value = actor_critic(state_tensor)
            std = torch.clamp(actor_critic.log_std, min=actor_critic.min_log_std).exp()
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            action = torch.tanh(action_sample)
            log_prob = tanh_log_prob(action_mean, actor_critic.log_std, action)
        
        states.append(obs)
        actions.append(action.cpu().numpy()[0])
        log_probs.append(log_prob.item())
        values.append(value.item())
        
        obs, reward, done, _, info = env.step(action.cpu().numpy()[0])
        rewards.append(reward)
        dones.append(done)
        
        if done:
            if 'episode_cost' in info:
                episode_costs.append(info['episode_cost'])
            obs, _ = env.reset()
    
    return (np.array(states), np.array(actions), np.array(rewards), 
            np.array(dones), np.array(log_probs), np.array(values), episode_costs)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns

def ppo_update(actor_critic, optimizer, states, actions, old_log_probs, advantages, returns, 
               clip_epsilon=0.2, K_epochs=4, batch_size=256):
    states = torch.FloatTensor(states).to(DEVICE)
    actions = torch.FloatTensor(actions).to(DEVICE)
    old_log_probs = torch.FloatTensor(old_log_probs).to(DEVICE)
    advantages = torch.FloatTensor(advantages).to(DEVICE)
    returns = torch.FloatTensor(returns).to(DEVICE)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(K_epochs):
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            action_mean, values = actor_critic(batch_states)
            new_log_probs = tanh_log_prob(action_mean, actor_critic.log_std, batch_actions)
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = ((values.squeeze() - batch_returns) ** 2).mean()
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
            optimizer.step()

def train():
    print("=" * 60)
    print("EXP010: Shallow Network (5 features)")
    print("State: [error, error_integral, error_derivative, v_ego, future_severity]")
    print("Network: Linear(5, 16) -> Tanh -> Linear(16, 1)")
    print("=" * 60)
    
    data_folder = '/Users/engelbart/Desktop/stuff/controls_challenge/data'
    train_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])[:100]
    train_path = os.path.join(data_folder, train_files[0])
    
    env = TinyPhysicsGymEnv(train_path)
    actor_critic = ShallowActorCritic(state_dim=5, action_dim=1, hidden_size=16).to(DEVICE)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    
    best_cost = float('inf')
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(100):
        states, actions, rewards, dones, log_probs, values, episode_costs = collect_rollout(env, actor_critic, steps=2000)
        advantages, returns = compute_gae(rewards, values, dones)
        ppo_update(actor_critic, optimizer, states, actions, log_probs, advantages, returns)
        
        if episode_costs:
            mean_cost = np.mean(episode_costs)
            if mean_cost < best_cost:
                best_cost = mean_cost
                torch.save(actor_critic.state_dict(), 'results/ppo_best.pth')
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Mean Cost: {mean_cost:7.1f} | Best: {best_cost:7.1f} | Episodes: {len(episode_costs)}")
    
    print("\nTraining complete!")
    print(f"Best cost: {best_cost:.1f}")

if __name__ == '__main__':
    train()

