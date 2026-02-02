"""
Step 5: Learn 50-weight FF using PPO
PID is FIXED. FF is learned. Direct cost optimization.
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
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, FUTURE_PLAN_STEPS
from tinyphysics import CONTROL_START_IDX, LAT_ACCEL_COST_MULTIPLIER, DEL_T

# Config
model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
data_dir = Path(__file__).parent.parent.parent / 'data'

all_files = sorted(list(data_dir.glob('*.csv')))
np.random.seed(42)
np.random.shuffle(all_files)
all_routes = all_files[:2000]

max_epochs = 50
log_interval = 5
eval_interval = 10
num_envs = 16
steps_per_epoch = 40_000
batch_size = 2000
K_epochs = 10
lr = 1e-3
gamma, gae_lambda = 0.99, 0.95
eps_clip = 0.2

device = torch.device('cpu')


class PIDFixedFFLearnableController(BaseController):
    """PID fixed, FF comes from RL agent"""
    def __init__(self):
        self.p, self.i, self.d = 0.195, 0.100, -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.current_ff_action = 0.0  # Set by RL agent
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Total = PID + FF (FF comes from RL)
        return pid_action + self.current_ff_action


class TinyPhysicsEnv(gym.Env):
    def __init__(self, model_path, data_path):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        
        # Obs: future errors (50)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )
        # Action: FF contribution
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.model = TinyPhysicsModel(str(model_path), debug=False)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.controller = PIDFixedFFLearnableController()
        self.sim = TinyPhysicsSimulator(
            self.model, str(self.data_path), 
            controller=self.controller, debug=False
        )
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        target = self.sim.target_lataccel_history[-1]
        _, _, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        
        future_padded = list(future_plan.lataccel[:50]) + [target] * (50 - len(future_plan.lataccel))
        future_errors = np.array([(f - target) for f in future_padded[:50]], dtype=np.float32)
        return future_errors
    
    def _get_reward(self, prev_lataccel, curr_lataccel, target_lataccel):
        tracking_error = (target_lataccel - curr_lataccel) ** 2
        jerk = (curr_lataccel - prev_lataccel) / DEL_T
        jerk_penalty = jerk ** 2
        reward = -(tracking_error * LAT_ACCEL_COST_MULTIPLIER + jerk_penalty) * 0.01
        return reward
    
    def step(self, action):
        # Set FF action in controller
        self.controller.current_ff_action = float(action[0])
        
        prev_lataccel = self.sim.current_lataccel
        
        # Step sim
        self.sim.step()
        
        # Reward
        target = self.sim.target_lataccel_history[-2]
        reward = self._get_reward(prev_lataccel, self.sim.current_lataccel, target)
        
        done = self.sim.step_idx >= len(self.sim.data)
        info = {}
        if done:
            info['official_cost'] = self.sim.compute_cost()['total_cost']
        
        obs = self._get_obs() if not done else np.zeros(50, dtype=np.float32)
        
        return obs, reward, done, False, info


def make_env(routes):
    def _make():
        route = np.random.choice(routes)
        return TinyPhysicsEnv(model_path, route)
    return gym.vector.AsyncVectorEnv([_make for _ in range(num_envs)])


class FFPolicy(nn.Module):
    """50 weights: future_errors -> ff_action"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(50, 1, bias=False)
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # Initialize weights small
        nn.init.zeros_(self.fc.weight)
    
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
    def __init__(self, policy, lr, gamma, lamda, K_epochs, eps_clip, batch_size):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size

    def compute_advantages(self, rewards, is_terminals):
        # Simple advantage: discounted returns
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
                
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[idx]
                loss = -torch.min(surr1, surr2).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()


def track_episode_returns(done_mask, ep_costs, infos):
    for idx in np.where(done_mask)[0]:
        info = infos.get('official_cost', [0] * len(done_mask))[idx]
        ep_costs.append(info if info > 0 else 0)


def rollout(env, policy, num_steps):
    N = env.num_envs
    states, _ = env.reset()
    ep_costs = []
    
    T = num_steps // N
    obs = np.empty((T, N, 50), dtype=np.float32)
    act = np.empty((T, N, 1), dtype=np.float32)
    rew = np.empty((T, N), dtype=np.float32)
    done = np.empty((T, N), dtype=np.float32)
    
    for t in range(T):
        actions = policy.act(states, deterministic=False)
        obs[t], act[t] = states, actions
        
        states, rewards, terminated, truncated, infos = env.step(actions)
        d = np.logical_or(terminated, truncated)
        rew[t], done[t] = rewards, d
        
        track_episode_returns(d, ep_costs, infos)
    
    return obs, act, rew, done, ep_costs


def train():
    print("="*60)
    print("Learning 50-weight FF with PPO")
    print("="*60)
    print(f"PID: FIXED (0.195, 0.100, -0.053)")
    print(f"FF: 50 weights (linear)")
    print(f"Routes: {len(all_routes)}")
    print("="*60)
    
    split_idx = int(0.8 * len(all_routes))
    train_routes = all_routes[:split_idx]
    eval_routes = all_routes[split_idx:]
    
    env = make_env(train_routes)
    eval_env = make_env(eval_routes)
    
    policy = FFPolicy().to(device)
    ppo = PPO(policy, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size)
    
    best_cost = float('inf')
    last_eval = float('inf')
    
    for epoch in range(max_epochs):
        obs, act, rew, done, ep_costs = rollout(env, policy, steps_per_epoch)
        ppo.update(obs, act, rew, done)
        
        train_100 = np.mean(ep_costs[-100:]) if len(ep_costs) >= 100 else np.mean(ep_costs)
        
        if epoch % eval_interval == 0:
            _, _, _, _, eval_costs = rollout(eval_env, policy, 16000)  # Enough for ~1.7 episodes per env
            last_eval = np.mean(eval_costs) if len(eval_costs) > 0 else float('inf')
            if last_eval < best_cost and last_eval < float('inf'):
                best_cost = last_eval
                torch.save(policy.state_dict(), Path(__file__).parent / 'ff_policy_best.pt')
        
        if epoch % log_interval == 0:
            σ = policy.log_std.exp().item()
            print(f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  cost={np.mean(ep_costs):6.1f}±{np.std(ep_costs):5.1f}  train_100={train_100:6.1f}  eval={last_eval:6.1f}  best={best_cost:6.1f}  σ={σ:.3f}", flush=True)
        
        if train_100 < 82:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED! train_100={train_100:.1f}")
            print(f"{'='*60}")
            break
    
    env.close()
    eval_env.close()


if __name__ == '__main__':
    train()
