"""
Step 6: Train 50-weight FF using battle-tested beautiful_lander PPO
Goal: Discover tau=0.9 pattern from scratch
Expected: Cost ~81.46 (matching hand-crafted decay)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController
from tinyphysics import CONTROL_START_IDX, LAT_ACCEL_COST_MULTIPLIER, DEL_T

# Config
model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
data_dir = Path(__file__).parent.parent.parent / 'data'

all_files = sorted(list(data_dir.glob('*.csv')))
np.random.seed(42)
np.random.shuffle(all_files)
all_routes = all_files[:2000]
split_idx = int(0.8 * len(all_routes))
train_routes = all_routes[:split_idx]
eval_routes = all_routes[split_idx:]

# Training params (from beautiful_lander)
max_epochs = 100
log_interval, eval_interval = 5, 10
num_envs = 16
steps_per_epoch = 40_000
batch_size = 2000
K_epochs = 10
pi_lr = 3e-4
gamma, gae_lambda = 0.99, 0.95
eps_clip = 0.2
entropy_coef = 0.001

device = torch.device('cpu')


class PIDPlusFFController(BaseController):
    def __init__(self):
        self.p, self.i, self.d = 0.195, 0.100, -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.ff_action = 0.0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid = self.p * error + self.i * self.error_integral + self.d * error_diff
        return pid + self.ff_action


class TinyPhysicsEnv(gym.Env):
    def __init__(self, model_path, route_pool, worker_id=0):
        super().__init__()
        self.model_path = model_path
        self.route_pool = list(route_pool)  # Pool of routes to sample from
        self.worker_id = worker_id
        self.data_path = None  # Will be set on reset
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.model = TinyPhysicsModel(str(model_path), debug=False)
        # Use process ID + worker_id for unique seed per worker
        import os
        unique_seed = os.getpid() + worker_id * 1000
        self._np_random = np.random.RandomState(unique_seed)
        self.reset_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a new random route from the pool on each reset
        # Use reset_count to ensure different routes on subsequent resets
        self.reset_count += 1
        route_idx = (self._np_random.randint(0, 1000000) + self.reset_count) % len(self.route_pool)
        self.data_path = self.route_pool[route_idx]
        self.controller = PIDPlusFFController()
        self.sim = TinyPhysicsSimulator(self.model, str(self.data_path), controller=self.controller, debug=False)
        return self._get_obs(), {}
    
    def _get_obs(self):
        target = self.sim.target_lataccel_history[-1]
        _, _, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        future_padded = list(future_plan.lataccel[:50]) + [target] * (50 - len(future_plan.lataccel))
        return np.array([(f - target) for f in future_padded[:50]], dtype=np.float32)
    
    def step(self, action):
        self.controller.ff_action = float(action[0])
        prev_lataccel = self.sim.current_lataccel
        self.sim.step()
        
        target = self.sim.target_lataccel_history[-2]
        tracking_error = (target - self.sim.current_lataccel) ** 2
        jerk = (self.sim.current_lataccel - prev_lataccel) / DEL_T
        reward = -(tracking_error * LAT_ACCEL_COST_MULTIPLIER + jerk**2) * 0.01
        
        done = self.sim.step_idx >= len(self.sim.data)
        info = {}
        if done:
            info['official_cost'] = self.sim.compute_cost()['total_cost']
        
        obs = self._get_obs() if not done else np.zeros(50, dtype=np.float32)
        return obs, reward, done, False, info


def make_env(routes):
    # Pass entire route pool to each env, they'll sample on reset
    # Give each worker a unique ID for seeding
    def _make(worker_id):
        return TinyPhysicsEnv(model_path, routes, worker_id=worker_id)
    return gym.vector.AsyncVectorEnv([lambda i=i: _make(i) for i in range(num_envs)])


class FFPolicy(nn.Module):
    """Single linear layer: 50 weights, no bias"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(50, 1, bias=False)
        # Fixed exploration noise (not learnable)
        self.register_buffer('log_std', torch.tensor([-3.9]))  # exp(-3.9) ≈ 0.02
        nn.init.zeros_(self.fc.weight)  # Start from zero
    
    def forward(self, x):
        return self.fc(x)
    
    @torch.inference_mode()
    def act(self, obs, deterministic=False):
        obs_t = torch.from_numpy(obs).to(dtype=torch.float32, device=device)
        action_mean = self(obs_t)
        action_std = self.log_std.exp()
        
        if deterministic:
            action = action_mean
        else:
            action = torch.distributions.Normal(action_mean, action_std).sample()
        
        return torch.clamp(action, -1, 1).cpu().numpy()


class PPO:
    def __init__(self, policy, pi_lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=pi_lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size
        self.entropy_coef = entropy_coef

    def compute_advantages(self, rewards, is_terminals):
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for t in reversed(range(T)):
            if t == T - 1:
                returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - is_terminals[t])
        
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def update(self, obs, act, rew, done):
        T, N = rew.shape
        obs_flat = torch.from_numpy(obs.reshape(-1, 50)).to(dtype=torch.float32, device=device)
        act_flat = torch.from_numpy(act.reshape(-1, 1)).to(dtype=torch.float32, device=device)
        rew_t = torch.from_numpy(rew).to(dtype=torch.float32, device=device)
        done_t = torch.from_numpy(done).to(dtype=torch.float32, device=device)
        
        with torch.no_grad():
            mean = self.policy(obs_flat)
            std = self.policy.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = dist.log_prob(act_flat).sum(-1)
            advantages, returns = self.compute_advantages(rew_t, done_t)
        
        for _ in range(self.K_epochs):
            perm = torch.randperm(len(obs_flat), device=device)
            for start in range(0, len(obs_flat), self.batch_size):
                idx = perm[start:start + self.batch_size]
                
                mean = self.policy(obs_flat[idx])
                std = self.policy.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                logprobs = dist.log_prob(act_flat[idx]).sum(-1)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[idx]
                loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()


def track_costs(done_mask, ep_costs, infos):
    for idx in np.where(done_mask)[0]:
        cost = infos.get('official_cost', [0] * len(done_mask))[idx]
        if cost > 0:
            ep_costs.append(cost)


def rollout(env, policy, num_steps, deterministic=False):
    N = env.num_envs
    states, _ = env.reset()
    ep_costs = []
    
    T = num_steps // N
    obs = np.empty((T, N, 50), dtype=np.float32)
    act = np.empty((T, N, 1), dtype=np.float32)
    rew = np.empty((T, N), dtype=np.float32)
    done = np.empty((T, N), dtype=np.float32)
    
    for t in range(T):
        actions = policy.act(states, deterministic=deterministic)
        obs[t], act[t] = states, actions
        
        states, rewards, terminated, truncated, infos = env.step(actions)
        d = np.logical_or(terminated, truncated)
        rew[t], done[t] = rewards, d
        
        track_costs(d, ep_costs, infos)
    
    return obs, act, rew, done, ep_costs


def train():
    print("="*60)
    print("Step 6: Learn 50-weight FF with PPO (beautiful_lander)")
    print("="*60)
    print(f"PID: FIXED (0.195, 0.100, -0.053)")
    print(f"FF: 50 weights, no bias, zero-initialized")
    print(f"Routes: train={len(train_routes)}, eval={len(eval_routes)}")
    print(f"PID baseline (eval set median): ~70")
    print("="*60)
    
    env = make_env(train_routes)
    eval_env = make_env(eval_routes)
    
    policy = FFPolicy().to(device)
    ppo = PPO(policy, pi_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    best_cost = float('inf')
    last_eval = float('inf')
    all_costs = []
    
    for epoch in range(max_epochs):
        obs, act, rew, done, ep_costs = rollout(env, policy, steps_per_epoch, deterministic=False)
        ppo.update(obs, act, rew, done)
        
        all_costs.extend(ep_costs)
        train_100 = np.mean(all_costs[-100:]) if len(all_costs) >= 100 else np.mean(all_costs)
        
        if epoch % eval_interval == 0:
            _, _, _, _, eval_costs = rollout(eval_env, policy, 20000, deterministic=True)
            last_eval = np.mean(eval_costs) if len(eval_costs) > 0 else float('inf')
            if last_eval < best_cost and last_eval < float('inf'):
                best_cost = last_eval
                torch.save(policy.state_dict(), Path(__file__).parent / 'ff_policy_best.pt')
                
                # Save weights as numpy for inspection
                w = policy.fc.weight.data.cpu().numpy().flatten()
                np.save(Path(__file__).parent / 'ppo_learned_weights.npy', w)
        
        if epoch % log_interval == 0:
            σ = policy.log_std.exp().item()
            w = policy.fc.weight.data.cpu().numpy().flatten()
            print(f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  cost={np.mean(ep_costs):6.1f}±{np.std(ep_costs):5.1f}  train_100={train_100:6.1f}  eval={last_eval:6.1f}  best={best_cost:6.1f}  σ={σ:.3f}  w[0]={w[0]:.4f}", flush=True)
    
    env.close()
    eval_env.close()
    
    # Analyze learned weights
    print(f"\n{'='*60}")
    print("Learned weights vs. optimal tau=0.9:")
    print(f"{'='*60}")
    w = policy.fc.weight.data.cpu().numpy().flatten()
    optimal = np.array([0.267 * np.exp(-i/0.9) for i in range(50)])
    
    for i in [0, 1, 2, 3, 5, 10]:
        print(f"w[{i}]: learned={w[i]:+.6f}, optimal={optimal[i]:+.6f}")


if __name__ == '__main__':
    train()
