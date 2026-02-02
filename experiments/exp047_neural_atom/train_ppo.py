"""
Train tiny tanh neuron with PPO using beautiful_lander scaffold
"""

import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.exp047_neural_atom.env import TinyPhysicsEnv

# Environment
model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
data_dir = Path(__file__).parent.parent.parent / 'data'

# Load same 2000 routes used for data collection
all_files = sorted(list(data_dir.glob('*.csv')))
np.random.seed(42)
all_files_shuffled = all_files.copy()
np.random.shuffle(all_files_shuffled)
all_routes = all_files_shuffled[:2000]

state_dim, action_dim = 4, 1
num_envs = 16  # More workers for more routes

# Training
max_epochs, steps_per_epoch = 50, 40_000
log_interval, eval_interval = 5, 10
batch_size, K_epochs = 2000, 10
hidden_dim = 16  # Tiny network
actor_layers, critic_layers = 2, 2  # Shallow
pi_lr, vf_lr = 3e-4, 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
vf_coef, entropy_coef = 0.5, 0.01
target_cost = 85.0  # Beat PID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_env(routes):
    """Create vectorized environment with given routes"""
    def _make():
        route = np.random.choice(routes)
        return TinyPhysicsEnv(model_path, route)
    
    return gym.vector.AsyncVectorEnv([_make for _ in range(num_envs)])


def track_episode_returns(done_mask, ep_returns, ep_rets, ep_costs, infos):
    """Track returns and compute equivalent costs"""
    for idx in np.where(done_mask)[0]:
        # AsyncVectorEnv returns infos = {'key': [val0, val1, ...]}
        # where each array element corresponds to an environment
        official_cost = infos.get('official_cost', [0.0] * len(ep_rets))[idx]
        
        avg_reward = ep_rets[idx]
        ep_returns.append(avg_reward)
        ep_costs.append(official_cost)
        ep_rets[idx] = 0.0


class ActorCritic(nn.Module):
    """Tiny network: 4 -> 16 -> 16 -> 1"""
    
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        # Actor
        actor = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        critic = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(critic_layers - 1):
            critic.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
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
        state_tensor = torch.from_numpy(state).to(device=dev, dtype=torch.float32)
        action_mean, action_std, _ = self(state_tensor)
        
        if deterministic:
            action = action_mean
        else:
            action = torch.distributions.Normal(action_mean, action_std).sample()
        
        # Clip to [-2, 2]
        action = torch.clamp(action, -2.0, 2.0)
        return action.cpu().numpy()


class PPO:
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.pi_optimizer = optim.Adam(
            list(actor_critic.actor.parameters()) + [actor_critic.log_std], 
            lr=pi_lr
        )
        self.vf_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size
        self.vf_coef, self.entropy_coef = vf_coef, entropy_coef

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
    
    def compute_loss(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = dist.log_prob(batch_actions).sum(-1)
        
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(
            ratios * batch_advantages,
            torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
        ).mean()
        
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().mean()
        
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
    
    def update(self, obs, act, rew, done):
        dev = next(self.actor_critic.parameters()).device
        T, N = rew.shape
        
        obs_flat = torch.from_numpy(obs.reshape(-1, state_dim)).to(device=dev, dtype=torch.float32)
        act_flat = torch.from_numpy(act.reshape(-1, action_dim)).to(device=dev, dtype=torch.float32)
        rew_t = torch.from_numpy(rew).to(device=dev, dtype=torch.float32)
        done_t = torch.from_numpy(done).to(device=dev, dtype=torch.float32)
        
        with torch.no_grad():
            mean, std, val = self.actor_critic(obs_flat)
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = dist.log_prob(act_flat).sum(-1)
            old_values = val.squeeze(-1).view(T, N)
            advantages, returns = self.compute_advantages(rew_t, old_values, done_t)
        
        num_samples = obs_flat.size(0)
        for _ in range(self.K_epochs):
            perm = torch.randperm(num_samples, device=dev)
            for start in range(0, num_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss = self.compute_loss(
                    obs_flat[idx], act_flat[idx], old_logprobs[idx], 
                    advantages[idx], returns[idx]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std],
                    max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()


def rollout(env, actor_critic, num_steps=None, num_episodes=None, deterministic=False):
    assert (num_steps is None) != (num_episodes is None)
    
    N = env.num_envs
    states, _ = env.reset()
    ep_returns, ep_costs = [], []
    ep_rets = np.zeros(N)
    
    collect = num_steps is not None
    if collect:
        T = num_steps // N
        obs = np.empty((T, N, state_dim), dtype=np.float32)
        act = np.empty((T, N, action_dim), dtype=np.float32)
        rew = np.empty((T, N), dtype=np.float32)
        done = np.empty((T, N), dtype=np.float32)
    
    t = 0
    while True:
        actions = actor_critic.act(states, deterministic=deterministic)
        if collect:
            obs[t], act[t] = states, actions
        
        states, rewards, terminated, truncated, infos = env.step(actions)
        d = np.logical_or(terminated, truncated)
        if collect:
            rew[t], done[t] = rewards, d
        
        ep_rets += rewards
        track_episode_returns(d, ep_returns, ep_rets, ep_costs, infos)
        t += 1
        
        if (collect and t >= T) or (num_episodes and len(ep_returns) >= num_episodes):
            break
    
    return (obs, act, rew, done, ep_returns, ep_costs) if collect else (ep_returns, ep_costs)


class TrainingContext:
    def __init__(self):
        # Dual models: CPU for rollout, device for update
        self.ac_cpu = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers).to('cpu')
        self.ac_device = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers).to(device)
        self.ppo = PPO(
            self.ac_device, pi_lr, vf_lr, gamma, gae_lambda, 
            K_epochs, eps_clip, batch_size, vf_coef, entropy_coef
        )
        
        # 80/20 split
        split_idx = int(0.8 * len(all_routes))
        self.train_routes = all_routes[:split_idx]
        self.eval_routes = all_routes[split_idx:]
        
        self.env = make_env(self.train_routes)
        self.eval_env = make_env(self.eval_routes)
        
        self.all_episode_costs = []
        self.best_cost = float('inf')
        self.last_eval = float('inf')
        
    def cleanup(self):
        self.env.close()
        self.eval_env.close()


def train_one_epoch(epoch, ctx):
    ctx.ac_cpu.load_state_dict(ctx.ac_device.state_dict())
    
    obs, act, rew, done, ep_rets, ep_costs = rollout(ctx.env, ctx.ac_cpu, num_steps=steps_per_epoch)
    ctx.ppo.update(obs, act, rew, done)
    
    ctx.all_episode_costs.extend(ep_costs)
    
    train_100 = np.mean(ctx.all_episode_costs[-100:]) if len(ctx.all_episode_costs) >= 100 else np.mean(ctx.all_episode_costs)
    
    # Always evaluate
    eval_cost = ctx.last_eval
    if epoch % eval_interval == 0:
        eval_rets, eval_costs = rollout(ctx.eval_env, ctx.ac_cpu, num_episodes=16, deterministic=True)
        eval_cost = np.mean(eval_costs)
        ctx.last_eval = eval_cost
        
        if eval_cost < ctx.best_cost:
            ctx.best_cost = eval_cost
            torch.save(
                ctx.ac_cpu.state_dict(), 
                Path(__file__).parent / 'best_policy.pt'
            )
    
    # Always log (use print, not pbar.write to avoid tqdm buffering)
    if epoch % log_interval == 0:
        s = ctx.ac_device.log_std.exp().detach().cpu().numpy()
        msg = f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  "
        msg += f"cost={np.mean(ep_costs):6.1f}±{np.std(ep_costs):5.1f}  "
        msg += f"train_100={train_100:6.1f}  "
        msg += f"eval={eval_cost:6.1f}  best={ctx.best_cost:6.1f}  "
        msg += f"σ={s[0]:.3f}"
        print(msg, flush=True)
    
    if train_100 <= target_cost:
        print(f"\n{'='*60}", flush=True)
        print(f"TARGET REACHED at epoch {epoch}!", flush=True)
        print(f"train_100={train_100:.1f} ≤ {target_cost}", flush=True)
        print(f"{'='*60}\n", flush=True)
        return True
    
    return False


def train():
    print("="*60)
    print("Training tiny tanh neuron with PPO")
    print("="*60)
    print(f"Device: {device}")
    
    ctx = TrainingContext()
    print(f"Routes: {len(all_routes)} (train={len(ctx.train_routes)}, eval={len(ctx.eval_routes)})")
    print(f"Network: {state_dim} -> {hidden_dim} -> {hidden_dim} -> {action_dim}")
    print(f"Target: cost ≤ {target_cost}")
    print(f"Num envs: {num_envs}")
    print("="*60)
    
    for epoch in range(max_epochs):
        if train_one_epoch(epoch, ctx):
            break
    ctx.cleanup()


if __name__ == '__main__':
    train()
