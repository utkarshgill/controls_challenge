"""
Experiment 031: Beautiful PPO for Controls Challenge

Surgical adaptation of beautiful_lander.py for lateral control.
- Clean PPO from scratch (no BC initialization)
- Battle-tested scaffolding from LunarLander
- Vectorized parallel environments
- Dual models (CPU rollout, device update)
- Proper cost function matching
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, State, FuturePlan

# Task dimensions
STATE_DIM = 55  # [error, error_integral, v_ego, a_ego, roll_lataccel, curvatures[50]]
ACTION_DIM = 1  # [steer_command]

# Training settings
NUM_ENVS = 8
MAX_EPOCHS = 200
STEPS_PER_EPOCH = 20_000  # Total steps across all envs
LOG_INTERVAL = 5
EVAL_INTERVAL = 10

# PPO hyperparameters
BATCH_SIZE = 2048
K_EPOCHS = 10
HIDDEN_DIM = 128
ACTOR_LAYERS = 3
CRITIC_LAYERS = 3
PI_LR = 3e-4
VF_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
VF_COEF = 0.5
ENTROPY_COEF = 0.01

# Device setup
METAL = False  # Set to True for M-series Mac
device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

# Load data
all_files = sorted(DATA_PATH.glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:15000]
test_files = all_files[15000:20000]

# Load simulator model
model_onnx = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# Normalization scales (from data statistics)
LATACCEL_SCALE = 3.0
V_SCALE = 40.0
A_SCALE = 3.0


class TinyPhysicsGymEnv:
    """Gym-like environment wrapper for TinyPhysics simulator"""
    
    def __init__(self, data_files, model):
        self.data_files = data_files
        self.model = model
        self.current_file_idx = 0
        self.controller = None
        self.sim = None
        self.step_idx = 0
        self.prev_lataccel = 0.0
        self.error_integral = 0.0
        self.max_steps = 1000  # Typical episode length
        
    def reset(self):
        """Reset to a random route"""
        file_path = str(random.choice(self.data_files))
        self.controller = EnvController()
        self.sim = TinyPhysicsSimulator(self.model, file_path, controller=self.controller, debug=False)
        self.step_idx = 0
        self.prev_lataccel = 0.0
        self.error_integral = 0.0
        
        # Get initial state
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Construct 55D state vector"""
        if self.sim is None or self.step_idx >= len(self.sim.data):
            return np.zeros(STATE_DIM, dtype=np.float32)
        
        # Current quantities
        target_lataccel = self.sim.target_lataccel_history[self.step_idx]
        current_lataccel = self.sim.current_lataccel_history[self.step_idx]
        state_obj = self.sim.state_history[self.step_idx]
        
        error = target_lataccel - current_lataccel
        
        # Update integral (simple accumulation)
        self.error_integral += error * 0.1  # dt = 0.1s
        self.error_integral = np.clip(self.error_integral, -10, 10)  # Anti-windup
        
        # Future curvatures (50 steps) - use simulator's processed data
        future_lataccels = []
        for i in range(50):
            future_idx = self.step_idx + i
            if future_idx < len(self.sim.data):
                future_lat = self.sim.data['target_lataccel'].values[future_idx]
                future_roll = self.sim.data['roll_lataccel'].values[future_idx]
                future_v = self.sim.data['v_ego'].values[future_idx]
                curvature = (future_lat - future_roll) / (future_v ** 2 + 1e-6)
            else:
                curvature = 0.0
            future_lataccels.append(curvature)
        
        # Normalize
        state = np.array([
            error / LATACCEL_SCALE,
            self.error_integral / LATACCEL_SCALE,
            state_obj.v_ego / V_SCALE,
            state_obj.a_ego / A_SCALE,
            state_obj.roll_lataccel / LATACCEL_SCALE,
            *future_lataccels
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute action, return (state, reward, terminated, truncated)"""
        if self.sim is None:
            return self._get_state(), 0.0, True, True
        
        # Store action for controller
        self.controller.action = float(action[0])
        
        # Execute simulator step
        try:
            self.sim.step()
            self.step_idx += 1
        except (IndexError, StopIteration):
            # Episode ended
            final_costs = self.sim.compute_cost()
            return self._get_state(), 0.0, True, False
        
        # Compute reward (negative cost)
        if self.step_idx >= len(self.sim.data):
            # Episode ended normally
            final_costs = self.sim.compute_cost()
            total_cost = final_costs['lataccel_cost'] * 50 + final_costs['jerk_cost']
            reward = -total_cost / 100.0  # Normalize
            return self._get_state(), reward, True, False
        
        # Step reward (negative instantaneous cost)
        target = self.sim.target_lataccel_history[self.step_idx]
        current = self.sim.current_lataccel_history[self.step_idx]
        lat_cost = (target - current) ** 2 * 100
        
        if self.step_idx > 0:
            prev = self.sim.current_lataccel_history[self.step_idx - 1]
            jerk_cost = ((current - prev) / 0.1) ** 2 * 100
        else:
            jerk_cost = 0.0
        
        step_cost = lat_cost * 50 + jerk_cost
        reward = -step_cost / 100.0
        
        terminated = self.step_idx >= len(self.sim.data) - 1
        truncated = self.step_idx >= self.max_steps
        
        next_state = self._get_state()
        return next_state, reward, terminated, truncated


class EnvController:
    """Dummy controller that uses actions from PPO"""
    def __init__(self):
        self.action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return float(np.clip(self.action, -2.0, 2.0))


def tanh_log_prob(raw_action, dist):
    """Log probability for tanh-squashed actions"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        # Actor network
        actor = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(actor_layers - 1):
            actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        actor.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
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
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * 2.0  # Scale to [-2, 2]
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
        
        # Compute entropy
        gaussian_entropy = dist.entropy().sum(-1)
        actions_squashed = torch.tanh(batch_actions)
        jacobian_correction = torch.log(1 - actions_squashed**2 + 1e-6).sum(-1)
        entropy = (gaussian_entropy - jacobian_correction).mean()
        
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
    
    def update(self, obs, raw_act, rew, done):
        dev = next(self.actor_critic.parameters()).device
        T, N = rew.shape
        obs_flat = torch.from_numpy(obs.reshape(-1, STATE_DIM)).to(device=dev, dtype=torch.float32)
        raw_act_flat = torch.from_numpy(raw_act.reshape(-1, ACTION_DIM)).to(device=dev, dtype=torch.float32)
        rew_t = torch.from_numpy(rew).to(device=dev, dtype=torch.float32)
        done_t = torch.from_numpy(done).to(device=dev, dtype=torch.float32)
        
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


def rollout_single_env(env, actor_critic, deterministic=False):
    """Rollout one episode in one environment"""
    state = env.reset()
    obs_list, raw_act_list, rew_list, done_list = [], [], [], []
    total_reward = 0.0
    
    while True:
        action, raw_action = actor_critic.act(state[None], deterministic=deterministic)
        obs_list.append(state)
        raw_act_list.append(raw_action[0])
        
        next_state, reward, terminated, truncated = env.step(action[0])
        rew_list.append(reward)
        done_list.append(float(terminated or truncated))
        total_reward += reward
        
        state = next_state
        
        if terminated or truncated:
            break
    
    return obs_list, raw_act_list, rew_list, done_list, total_reward


def rollout_parallel(envs, actor_critic, num_steps):
    """Collect num_steps total across parallel environments"""
    N = len(envs)
    T = num_steps // N
    
    obs = np.empty((T, N, STATE_DIM), dtype=np.float32)
    raw_act = np.empty((T, N, ACTION_DIM), dtype=np.float32)
    rew = np.empty((T, N), dtype=np.float32)
    done = np.empty((T, N), dtype=np.float32)
    
    # Initialize all envs
    states = np.array([env.reset() for env in envs])
    episode_rewards = []
    ep_rews = np.zeros(N)
    
    for t in range(T):
        # Get actions for all envs
        actions, raw_actions = actor_critic.act(states, deterministic=False)
        obs[t] = states
        raw_act[t] = raw_actions
        
        # Step all envs
        next_states = []
        for i, env in enumerate(envs):
            next_state, reward, terminated, truncated = env.step(actions[i])
            rew[t, i] = reward
            done[t, i] = float(terminated or truncated)
            ep_rews[i] += reward
            
            if terminated or truncated:
                episode_rewards.append(ep_rews[i])
                ep_rews[i] = 0.0
                next_state = env.reset()
            
            next_states.append(next_state)
        
        states = np.array(next_states)
    
    return obs, raw_act, rew, done, episode_rewards


def evaluate_policy(actor_critic, num_routes=20):
    """Evaluate policy on test routes"""
    from controllers.pid import Controller as PIDController
    
    costs = []
    for i, file_path in enumerate(test_files[:num_routes]):
        env = TinyPhysicsGymEnv([file_path], model_onnx)
        _, _, _, _, total_reward = rollout_single_env(env, actor_critic, deterministic=True)
        
        # Convert reward back to cost (reward = -cost/100)
        cost = -total_reward * 100
        costs.append(cost)
    
    return np.mean(costs)


class TrainingContext:
    def __init__(self):
        # Dual models: cpu for rollout, device for update
        self.ac_cpu = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS).to('cpu')
        self.ac_device = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS).to(device)
        self.ppo = PPO(self.ac_device, PI_LR, VF_LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, VF_COEF, ENTROPY_COEF)
        
        # Create parallel environments
        self.envs = [TinyPhysicsGymEnv(train_files, model_onnx) for _ in range(NUM_ENVS)]
        
        self.all_episode_returns = []
        self.best_cost = float('inf')
        self.last_eval = float('inf')
        self.pbar = trange(MAX_EPOCHS, desc="Training", unit='epoch')
        self.rollout_times = []
        self.update_times = []
    
    def cleanup(self):
        self.pbar.close()


def train_one_epoch(epoch, ctx):
    ctx.ac_cpu.load_state_dict(ctx.ac_device.state_dict())
    
    t0 = time.perf_counter()
    obs, raw_act, rew, done, ep_rets = rollout_parallel(ctx.envs, ctx.ac_cpu, STEPS_PER_EPOCH)
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    t0 = time.perf_counter()
    ctx.ppo.update(obs, raw_act, rew, done)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    ctx.all_episode_returns.extend(ep_rets)
    ctx.pbar.update(1)
    
    # Convert returns to costs (return = -cost/100)
    ep_costs = [-r * 100 for r in ep_rets]
    mean_cost = np.mean(ep_costs) if ep_costs else float('inf')
    
    if epoch % EVAL_INTERVAL == 0:
        ctx.last_eval = evaluate_policy(ctx.ac_cpu, num_routes=20)
        
        if ctx.last_eval < ctx.best_cost:
            ctx.best_cost = ctx.last_eval
            save_path = Path(__file__).parent / 'ppo_best.pth'
            torch.save({
                'model_state_dict': ctx.ac_device.state_dict(),
                'epoch': epoch,
                'cost': ctx.best_cost
            }, str(save_path))
            ctx.pbar.write(f"✅ New best: {ctx.best_cost:.2f}")
    
    if epoch % LOG_INTERVAL == 0:
        s = ctx.ac_device.log_std.exp().detach().cpu().numpy()
        rollout_ms = np.mean(ctx.rollout_times[-LOG_INTERVAL:]) * 1000
        update_ms = np.mean(ctx.update_times[-LOG_INTERVAL:]) * 1000
        total_ms = rollout_ms + update_ms
        ctx.pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  cost={mean_cost:7.1f}±{np.std(ep_costs):5.1f}  eval={ctx.last_eval:6.1f}  best={ctx.best_cost:6.1f}  σ={s[0]:.3f}  ⏱ {total_ms:.0f}ms")
    
    return False


def train():
    print("="*80)
    print("Experiment 031: Beautiful PPO for Controls Challenge")
    print("="*80)
    print(f"Device: {device}")
    print(f"State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"Num envs: {NUM_ENVS}, Steps per epoch: {STEPS_PER_EPOCH}")
    print(f"Training files: {len(train_files)}, Test files: {len(test_files)}")
    print("="*80)
    
    ctx = TrainingContext()
    for epoch in range(MAX_EPOCHS):
        train_one_epoch(epoch, ctx)
    
    # Save final model
    save_path = Path(__file__).parent / 'ppo_final.pth'
    torch.save({
        'model_state_dict': ctx.ac_device.state_dict(),
        'epoch': MAX_EPOCHS,
        'cost': ctx.last_eval
    }, str(save_path))
    
    ctx.cleanup()
    
    print("="*80)
    print(f"Training complete!")
    print(f"Best cost: {ctx.best_cost:.2f}")
    print("="*80)


if __name__ == '__main__':
    train()

