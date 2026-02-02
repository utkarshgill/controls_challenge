import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import warnings
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

env_name = 'LunarLanderContinuous-v3'
state_dim, action_dim = 8, 2
num_envs = int(os.getenv('NUM_ENVS', 8))

# https://gymnasium.farama.org/environments/box2d/lunar_lander/#:~:text=For%20the%20default%20values%20of%20VIEWPORT_W%2C%20VIEWPORT_H%2C%20SCALE%2C%20and%20FPS%2C%20the%20scale%20factors%20equal%3A%20%E2%80%98x%E2%80%99%3A%2010%2C%20%E2%80%98y%E2%80%99%3A%206.666%2C%20%E2%80%98vx%E2%80%99%3A%205%2C%20%E2%80%98vy%E2%80%99%3A%207.5%2C%20%E2%80%98angle%E2%80%99%3A%201%2C%20%E2%80%98angular%20velocity%E2%80%99%3A%202.5
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)

max_epochs, max_timesteps, steps_per_epoch = 100, 1000, 100_000
log_interval, eval_interval = 5, 10
batch_size, K_epochs = 10_000, 10
hidden_dim = 128
trunk_layers, head_layers = 1, 3
lr = 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
entropy_coef = 0.001
solved_threshold = 250

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
RENDER_EPISODES = int(os.getenv('RENDER_EPISODES', '3'))
METAL = bool(int(os.getenv('METAL', '0')))

device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

if PLOT: import matplotlib.pyplot as plt

def make_env(n, render=False):
    render_mode = 'human' if render else None
    return gym.vector.AsyncVectorEnv([lambda: gym.make(env_name, render_mode=render_mode) for _ in range(n)])

def update_plot(ax, returns, threshold):
    ax.clear()
    ax.plot(returns, alpha=0.3, label='Episode Returns')
    if len(returns) >= 100:
        ma = np.convolve(returns, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(returns)), ma, label='100-ep MA', linewidth=2)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({threshold})')
    ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)

def tanh_log_prob(raw_action, dist):
    # Change of variables for tanh squashing
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
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action)
        return (action.cpu().numpy(), state_tensor, raw_action) if return_internals else action.cpu().numpy()

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic, self.states, self.actions = actor_critic, [], []
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef

    def __call__(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False, return_internals=True)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
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
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        self.states, self.actions = [], []

def rollout(env, policy, num_steps=None, num_episodes=None):
    assert (num_steps is None) != (num_episodes is None), "Specify exactly one: num_steps or num_episodes"
    states, _ = env.reset()
    traj_rewards, traj_dones, ep_returns, ep_rets, step_count = [], [], [], np.zeros(env.num_envs), 0
    while True:
        states, rewards, terminated, truncated, _ = env.step(policy(states))
        traj_rewards.append(rewards)
        traj_dones.append(np.logical_or(terminated, truncated))
        ep_rets += rewards
        step_count += env.num_envs
        if np.any(traj_dones[-1]):
            for idx in np.where(traj_dones[-1])[0]:
                ep_returns.append(ep_rets[idx])
                ep_rets[idx] = 0.0
        if (num_steps and step_count >= num_steps) or (num_episodes and len(ep_returns) >= num_episodes):
            break
    return traj_rewards, traj_dones, ep_returns

def train_one_epoch(env, ppo):
    rewards, dones, ep_rets = rollout(env, ppo, num_steps=steps_per_epoch)
    ppo.update(rewards, dones)
    return ep_rets

def train():
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers).to(device)
    ppo, env = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef), make_env(num_envs)
    env_viz = make_env(1, render=True) if RENDER else None
    all_episode_returns, last_eval = [], float('-inf')
    
    def render_policy():
        nonlocal env_viz
        if env_viz is None:
            env_viz = make_env(1, render=True)
        evaluate_policy(actor_critic, render=True, num_episodes=RENDER_EPISODES, env=env_viz)
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    pbar = trange(max_epochs, desc="Training", unit='epoch')
    for epoch in range(max_epochs):
        ep_rets = train_one_epoch(env, ppo)
        all_episode_returns.extend(ep_rets)
        pbar.update(1)
        train_100 = np.mean(all_episode_returns[-100:]) if all_episode_returns else 0.0
        if epoch % eval_interval == 0:
            last_eval = evaluate_policy(actor_critic)
            if RENDER:
                render_policy()
        if epoch % log_interval == 0:
            s = actor_critic.log_std.exp().detach().cpu().numpy()
            pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_rets):3d}  ret={np.mean(ep_rets):7.1f}±{np.std(ep_rets):5.1f}  train_100={train_100:6.1f}  eval={last_eval:6.1f}  σ=[{s[0]:.2f} {s[1]:.2f}]")
        if train_100 >= solved_threshold:
            pbar.write(f"\n{'='*60}\nSOLVED at epoch {epoch}! train_100={train_100:.1f} ≥ {solved_threshold}\n{'='*60}")
            render_policy()
            break
        if PLOT and epoch % (log_interval * 2) == 0:
            update_plot(ax, all_episode_returns, solved_threshold)
    env.close()
    if env_viz is not None:
        env_viz.close()
    pbar.close()
    if PLOT:
        plt.ioff()
        plt.show()

def evaluate_policy(actor_critic, n=16, render=False, num_episodes=None, env=None):
    close_env = env is None
    if env is None:
        env = make_env(1 if render else n, render)
    def policy(s): return actor_critic.act(s, deterministic=True)
    if render and num_episodes:
        _, _, ep_rets = rollout(env, policy, num_episodes=num_episodes)
    else:
        _, _, ep_rets = rollout(env, policy, num_steps=max_timesteps * (1 if render else n))
    if close_env:
        env.close()
    return float(np.mean(ep_rets)) if ep_rets else 0.0

if __name__ == '__main__':
    print(f"Using {device} device")
    train()