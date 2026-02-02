import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import warnings
from tqdm import trange

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

env_name = 'LunarLanderContinuous-v3'
state_dim, action_dim = 8, 2
num_envs = int(os.getenv('NUM_ENVS', 24))

# https://gymnasium.farama.org/environments/box2d/lunar_lander/#:~:text=For%20the%20default%20values%20of%20VIEWPORT_W%2C%20VIEWPORT_H%2C%20SCALE%2C%20and%20FPS%2C%20the%20scale%20factors%20equal%3A%20%E2%80%98x%E2%80%99%3A%2010%2C%20%E2%80%98y%E2%80%99%3A%206.666%2C%20%E2%80%98vx%E2%80%99%3A%205%2C%20%E2%80%98vy%E2%80%99%3A%207.5%2C%20%E2%80%98angle%E2%80%99%3A%201%2C%20%E2%80%98angular%20velocity%E2%80%99%3A%202.5
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)

max_epochs, steps_per_epoch = 100, 100_000
log_interval, eval_interval = 5, 10
batch_size, K_epochs = 5000, 20
hidden_dim = 128
actor_layers, critic_layers = 5, 3
pi_lr, vf_lr = 3e-4, 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
vf_coef, entropy_coef = 0.5, 0.001
solved_threshold = 250

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
RENDER_EPISODES = int(os.getenv('RENDER_EPISODES', '3'))
METAL = bool(int(os.getenv('METAL', '0')))

device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
OBS_SCALE_T_CPU = torch.tensor(OBS_SCALE, dtype=torch.float32, device='cpu')
OBS_SCALE_T_DEVICE = torch.tensor(OBS_SCALE, dtype=torch.float32, device=device)

if PLOT:
    import matplotlib.pyplot as plt

def make_env(n, render=False):
    render_mode = 'human' if render else None
    return gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_name, render_mode=render_mode) for _ in range(n)],
    )

def update_plot(ax, returns, threshold):
    ax.clear()
    ax.plot(returns, alpha=0.3, label='Episode Returns')
    if len(returns) >= 100:
        ma = np.convolve(returns, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(returns)), ma, label='100-ep MA', linewidth=2)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({threshold})')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    plt.pause(0.01)

def tanh_log_prob(raw_action, dist):
    # log π(tanh(x)) = log π(x) - log|det J| for change of variables
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

def track_episode_returns(done_mask, ep_returns, ep_rets):
    for idx in np.where(done_mask)[0]:
        ep_returns.append(ep_rets[idx])
        ep_rets[idx] = 0.0

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        actor = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        critic = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(critic_layers - 1):
            critic.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        critic.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*critic)

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        value = self.critic(state)
        return action_mean, action_std, value
    
    @torch.inference_mode()
    def act(self, state, deterministic=False):
        dev = next(self.parameters()).device
        scale = OBS_SCALE_T_CPU if dev.type == 'cpu' else OBS_SCALE_T_DEVICE
        state_tensor = torch.from_numpy(state).to(device=dev, dtype=torch.float32) * scale
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action)
        return action.cpu().numpy(), raw_action.cpu().numpy()

class PPO:
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.pi_optimizer = optim.Adam(list(actor_critic.actor.parameters()) + [actor_critic.log_std], lr=pi_lr)
        self.vf_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.vf_coef, self.entropy_coef = eps_clip, batch_size, vf_coef, entropy_coef

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
    
    def compute_loss(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        
        # Compute entropy of tanh-transformed distribution: H[tanh(X)] = H[X] - E[log(1 - tanh²(X))]
        gaussian_entropy = dist.entropy().sum(-1)  # [batch_size]
        actions_squashed = torch.tanh(batch_actions)
        jacobian_correction = torch.log(1 - actions_squashed**2 + 1e-6).sum(-1)  # [batch_size]
        entropy = (gaussian_entropy - jacobian_correction).mean()
        
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
    
    def update(self, obs, raw_act, rew, done):
        dev = next(self.actor_critic.parameters()).device
        T, N = rew.shape
        obs_flat = torch.from_numpy(obs.reshape(-1, state_dim)).to(device=dev, dtype=torch.float32)
        raw_act_flat = torch.from_numpy(raw_act.reshape(-1, action_dim)).to(device=dev, dtype=torch.float32)
        rew_t = torch.from_numpy(rew).to(device=dev, dtype=torch.float32)
        done_t = torch.from_numpy(done).to(device=dev, dtype=torch.float32)
        
        scale = OBS_SCALE_T_CPU if dev.type == 'cpu' else OBS_SCALE_T_DEVICE
        obs_flat = obs_flat * scale
        
        with torch.no_grad():
            mean, std, val = self.actor_critic(obs_flat)
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = tanh_log_prob(raw_act_flat, dist)
            old_values = val.squeeze(-1).view(T, N)
            advantages, returns = self.compute_advantages(rew_t, old_values, done_t)
        
        num_samples = obs_flat.size(0)
        for _ in range(self.K_epochs):
            perm = torch.randperm(num_samples, device=dev)
            for start in range(0, num_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss = self.compute_loss(obs_flat[idx], raw_act_flat[idx], old_logprobs[idx], advantages[idx], returns[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std], max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()

def rollout(env, actor_critic, num_steps=None, num_episodes=None, deterministic=False):
    assert (num_steps is None) != (num_episodes is None), "Specify exactly one: num_steps or num_episodes"
    
    N = env.num_envs
    states, _ = env.reset()
    ep_returns, ep_rets = [], np.zeros(N)
    
    collect = num_steps is not None
    if collect:
        T = num_steps // N
        obs = np.empty((T, N, state_dim), dtype=np.float32)
        raw_act = np.empty((T, N, action_dim), dtype=np.float32)
        rew = np.empty((T, N), dtype=np.float32)
        done = np.empty((T, N), dtype=np.float32)
    
    t = 0
    while True:
        actions, raw_actions = actor_critic.act(states, deterministic=deterministic)
        if collect:
            obs[t], raw_act[t] = states, raw_actions
        
        states, rewards, terminated, truncated, _ = env.step(actions)
        d = np.logical_or(terminated, truncated)
        if collect:
            rew[t], done[t] = rewards, d
        
        ep_rets += rewards
        track_episode_returns(d, ep_returns, ep_rets)
        t += 1
        
        if (collect and t >= T) or (num_episodes and len(ep_returns) >= num_episodes):
            break
    
    return (obs, raw_act, rew, done, ep_returns) if collect else ep_returns

class TrainingContext:
    def __init__(self):
        # dual models: cpu for rollout, device for update (avoids mps transfer latency)
        self.ac_cpu = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers).to('cpu')
        self.ac_device = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers).to(device)
        self.ppo = PPO(self.ac_device, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef)
        
        self.env = make_env(num_envs)
        self.eval_env = make_env(16)
        self.all_episode_returns = []
        self.last_eval = float('-inf')
        self.pbar = trange(max_epochs, desc="Training", unit='epoch')
        self.rollout_times = []
        self.update_times = []
        
        if PLOT:
            plt.ion()
            _, self.ax = plt.subplots()
        else:
            self.ax = None
    
    def cleanup(self):
        self.env.close()
        self.eval_env.close()
        self.pbar.close()
        if PLOT:
            plt.ioff()
            plt.show()

def train_one_epoch(epoch, ctx):
    ctx.ac_cpu.load_state_dict(ctx.ac_device.state_dict())
    
    t0 = time.perf_counter()
    obs, raw_act, rew, done, ep_rets = rollout(ctx.env, ctx.ac_cpu, num_steps=steps_per_epoch)
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    t0 = time.perf_counter()
    ctx.ppo.update(obs, raw_act, rew, done)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    ctx.all_episode_returns.extend(ep_rets)
    ctx.pbar.update(1)
    
    train_100 = np.mean(ctx.all_episode_returns[-100:]) if ctx.all_episode_returns else 0.0
    
    if epoch % eval_interval == 0:
        ctx.last_eval = evaluate_policy(ctx.ac_cpu, env=ctx.eval_env)
        if RENDER:
            evaluate_policy(ctx.ac_cpu, render=True, num_episodes=RENDER_EPISODES)
    
    if epoch % log_interval == 0:
        s = ctx.ac_device.log_std.exp().detach().cpu().numpy()
        rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
        update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
        total_ms = rollout_ms + update_ms
        ctx.pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_rets):3d}  ret={np.mean(ep_rets):7.1f}±{np.std(ep_rets):5.1f}  train_100={train_100:6.1f}  eval={ctx.last_eval:6.1f}  σ=[{s[0]:.2f} {s[1]:.2f}]  ⏱ {total_ms:.0f}ms (rollout:{rollout_ms:.0f}ms update:{update_ms:.0f}ms)")
    
    if PLOT and epoch % (log_interval * 2) == 0:
        update_plot(ctx.ax, ctx.all_episode_returns, solved_threshold)
    
    if train_100 >= solved_threshold:
        ctx.pbar.write(f"\n{'='*60}\nSOLVED at epoch {epoch}! train_100={train_100:.1f} ≥ {solved_threshold}\n{'='*60}")
        if RENDER:
            evaluate_policy(ctx.ac_cpu, render=True, num_episodes=RENDER_EPISODES)
        return True
    
    return False

def train():
    ctx = TrainingContext()
    for epoch in range(max_epochs):
        if train_one_epoch(epoch, ctx):
            break
    ctx.cleanup()

def evaluate_policy(actor_critic, num_episodes=16, render=False, env=None):
    close_env = env is None
    if env is None:
        env = make_env(1 if render else num_episodes, render)
    ep_rets = rollout(env, actor_critic, num_episodes=num_episodes, deterministic=True)
    if close_env:
        env.close()
    return float(np.mean(ep_rets)) if ep_rets else 0.0

if __name__ == '__main__':
    print(f"Using {device} device")
    if METAL:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        if device.type == 'mps':
            print("✅ MPS device active")
        else:
            print("⚠️  METAL=1 but device is not MPS")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    train()