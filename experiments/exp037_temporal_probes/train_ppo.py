"""
Experiment 037: NNFF-Style Temporal Probes

**The breakthrough insight from NNFF analysis**:
Winners didn't use 50 raw future values. They used 6-8 carefully chosen temporal probes.

NNFF philosophy:
- Don't feed raw sequences (high-dim, correlated)
- Feed orthogonal physical questions at specific horizons
- Learn gains on interpretable basis functions

The 7 features (stolen directly from NNFF structure):
1. Δlat_03:  future.lataccel[3]  - target_lataccel(t)  # 0.3s ahead
2. Δlat_06:  future.lataccel[6]  - target_lataccel(t)  # 0.6s ahead
3. Δlat_10:  future.lataccel[10] - target_lataccel(t)  # 1.0s ahead
4. Δlat_15:  future.lataccel[15] - target_lataccel(t)  # 1.5s ahead
5. Δroll_03: future.roll[3]  - roll_lataccel(t)
6. Δroll_10: future.roll[10] - roll_lataccel(t)
7. v_ego

These are **basis functions** (like NNFF's error_m03, lataccel_p06, etc.)

Policy: Start LINEAR (7 weights), allow 1 hidden layer only if linear improves.

This is how winners got <45: not smarter PPO, but better questions.
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

# Task dimensions - NNFF-STYLE TEMPORAL PROBES
STATE_DIM = 7   # [Δlat_03, Δlat_06, Δlat_10, Δlat_15, Δroll_03, Δroll_10, v_ego]
ACTION_DIM = 1  # [anticipatory correction]

# Training settings
NUM_ENVS = 8
MAX_EPOCHS = 200
STEPS_PER_EPOCH = 20_000
LOG_INTERVAL = 5
EVAL_INTERVAL = 10

# PPO hyperparameters
BATCH_SIZE = 2048
K_EPOCHS = 10
HIDDEN_DIM = 0    # NO hidden layers - pure linear
ACTOR_LAYERS = 0  # Zero hidden layers (just input → output)
CRITIC_LAYERS = 2  # Critic can be shallow MLP
PI_LR = 3e-4
VF_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
VF_COEF = 0.05  # Nerfed - critic is blind to PID/simulator hidden state
ENTROPY_COEF = 0.0  # ZERO - no exploration, just parameter tuning

# 3-PARAMETER LINEAR ARCHITECTURE (Experiment 034):
# 1. State: 3 preview features (f1, f2, f3) - NO error, NO current state
# 2. Policy: Pure linear (no hidden layers) - r = w1×f1 + w2×f2 + w3×f3
# 3. Low-pass: α=0.05 (very aggressive filtering)
# 4. This is parameter tuning, not behavior learning

# Residual learning parameters
RESIDUAL_SCALE = 0.1  # ε: residual is 10% of full action range
RESIDUAL_CLIP = 0.5   # Clip residual to ±0.5 before scaling
LOWPASS_ALPHA = 0.05  # VERY AGGRESSIVE low-pass: blocks high frequency reactive leakage
RESIDUAL_WARMUP = 50  # Steps before residual activates (PID settles first)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
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

# Normalization scales
LATACCEL_SCALE = 3.0
V_SCALE = 40.0
A_SCALE = 3.0
CURVATURE_SCALE = 0.01  # Typical curvature magnitude


class PIDController:
    """Frozen PID baseline - matches controllers/pid.py exactly"""
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        output = self.p * error + self.i * self.error_integral + self.d * error_diff
        return float(np.clip(output, -2.0, 2.0))


class ResidualController:
    """Simple controller that executes given actions (no policy calls)"""
    def __init__(self):
        self.current_action = 0.0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Interface method called by simulator - returns stored action"""
        return self.current_action
    
    def set_action(self, action):
        """Set action to be executed (called by environment)"""
        self.current_action = float(np.clip(action, -2.0, 2.0))


class TinyPhysicsGymEnv:
    """Gym-like environment with residual learning architecture"""
    
    def __init__(self, data_files, model):
        self.data_files = data_files
        self.model = model
        self.controller = None
        self.sim = None
        self.step_idx = 0
        self.error_integral = 0.0
        self.max_steps = 1000
        self.pid = PIDController()  # Env owns PID
        self.prev_residual = 0.0  # For low-pass filtering
        self.prev_error_squared = 0.0  # For reward shaping (error improvement)
        
    def reset(self):
        """Reset to a random route"""
        file_path = str(random.choice(self.data_files))
        self.controller = ResidualController()
        self.sim = TinyPhysicsSimulator(self.model, file_path, controller=self.controller, debug=False)
        self.step_idx = 0
        self.error_integral = 0.0
        self.pid.reset()
        self.prev_residual = 0.0
        self.prev_error_squared = 0.0
        
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Construct 7D NNFF-STYLE TEMPORAL PROBES
        
        Key insight from NNFF analysis:
        - Don't feed 50 raw future values (high-dim, correlated)
        - Feed 6-8 orthogonal temporal probes at specific horizons
        
        The 7 probes:
        1-4: Anticipation probes at 0.3s, 0.6s, 1.0s, 1.5s
        5-6: Roll compensation at 0.3s, 1.0s
        7:   Speed conditioning
        
        These are basis functions (like NNFF's error_m03, lataccel_p06, etc.)
        NOT raw data dumps.
        """
        if self.sim is None or self.step_idx >= len(self.sim.data):
            return np.zeros(STATE_DIM, dtype=np.float32)
        
        # Current state baselines
        target_lat = self.sim.target_lataccel_history[self.step_idx]
        state_obj = self.sim.state_history[self.step_idx]
        current_roll = state_obj.roll_lataccel
        current_v = state_obj.v_ego
        
        # Extract future values at specific horizons
        def get_future(signal_name, idx_offset):
            future_idx = self.step_idx + idx_offset
            if future_idx < len(self.sim.data):
                return self.sim.data[signal_name].values[future_idx]
            else:
                # Pad with last valid
                return self.sim.data[signal_name].values[-1]
        
        # Temporal probes (indices chosen to match NNFF philosophy)
        # @ 10 Hz: step 3 = 0.3s, step 6 = 0.6s, step 10 = 1.0s, step 15 = 1.5s
        delta_lat_03 = (get_future('target_lataccel', 3) - target_lat) / LATACCEL_SCALE
        delta_lat_06 = (get_future('target_lataccel', 6) - target_lat) / LATACCEL_SCALE
        delta_lat_10 = (get_future('target_lataccel', 10) - target_lat) / LATACCEL_SCALE
        delta_lat_15 = (get_future('target_lataccel', 15) - target_lat) / LATACCEL_SCALE
        
        delta_roll_03 = (get_future('roll_lataccel', 3) - current_roll) / LATACCEL_SCALE
        delta_roll_10 = (get_future('roll_lataccel', 10) - current_roll) / LATACCEL_SCALE
        
        v_ego_norm = current_v / V_SCALE
        
        # 7D state vector (orthogonal physical questions)
        state = np.array([
            delta_lat_03,
            delta_lat_06,
            delta_lat_10,
            delta_lat_15,
            delta_roll_03,
            delta_roll_10,
            v_ego_norm
        ], dtype=np.float32)
        
        return state
    
    def _compute_step_cost(self, target, current, prev_lataccel):
        """Compute instantaneous cost at this timestep"""
        # Tracking cost
        lat_cost = (target - current) ** 2 * 100
        
        # Jerk cost (smoothness)
        jerk_cost = ((current - prev_lataccel) / 0.1) ** 2 * 100
        
        # Total cost with official weighting
        step_cost = 50 * lat_cost + jerk_cost
        return step_cost
    
    def step(self, raw_residual):
        """Execute action computed from raw_residual
        
        CRITICAL: raw_residual is the output from actor network (before tanh).
        We compute: action = PID(state) + ε × process(raw_residual)
        This ensures the action executed matches what PPO believes was executed.
        
        Residual is DISABLED for first RESIDUAL_WARMUP steps to:
        1. Let PID state settle (integral, derivative history)
        2. Force PPO to learn anticipation, not reactive correction
        3. Match the manifold that training actually operates on
        """
        if self.sim is None:
            return self._get_state(), 0.0, True, True
        
        # Get current state
        target = self.sim.target_lataccel_history[self.step_idx]
        current = self.sim.current_lataccel_history[self.step_idx]
        prev_lataccel = current
        
        # Compute PID component (env owns PID state)
        pid_action = self.pid.update(target, current)
        
        # Process residual: tanh → clip → filter → scale
        # BUT: disable residual during warmup (let PID settle first)
        if self.step_idx < RESIDUAL_WARMUP:
            scaled_residual = 0.0  # Pure PID during warmup
        else:
            residual = np.tanh(raw_residual) * RESIDUAL_CLIP
            filtered_residual = LOWPASS_ALPHA * residual + (1 - LOWPASS_ALPHA) * self.prev_residual
            self.prev_residual = filtered_residual
            scaled_residual = filtered_residual * RESIDUAL_SCALE
        
        # Combined action
        combined_action = pid_action + scaled_residual
        combined_action = np.clip(combined_action, -2.0, 2.0)
        
        # Set action for controller
        self.controller.set_action(combined_action)
        
        # Execute simulator step
        try:
            self.sim.step()
            self.step_idx += 1
        except (IndexError, StopIteration):
            return self._get_state(), 0.0, True, False
        
        # Check termination
        terminated = self.step_idx >= len(self.sim.data) - 1
        truncated = self.step_idx >= self.max_steps
        
        # Compute SHAPED reward: base cost + credit for error reduction
        if self.step_idx < len(self.sim.current_lataccel_history):
            new_current = self.sim.current_lataccel_history[self.step_idx]
            new_target = self.sim.target_lataccel_history[self.step_idx]
            
            # Base cost (negative)
            step_cost = self._compute_step_cost(target, new_current, prev_lataccel)
            base_reward = -step_cost / 100.0
            
            # Shaping: reward error reduction (anticipation bonus)
            current_error_squared = (new_target - new_current) ** 2
            error_improvement = self.prev_error_squared - current_error_squared
            shaping_bonus = 0.01 * error_improvement  # Small bonus for reducing future error
            
            reward = base_reward + shaping_bonus
            self.prev_error_squared = current_error_squared
        else:
            reward = 0.0
        
        next_state = self._get_state()
        return next_state, reward, terminated, truncated




def tanh_log_prob(raw_action, dist):
    """Log probability for tanh-squashed actions"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super(ActorCritic, self).__init__()
        
        # Actor network: PURE LINEAR (no hidden layers, no nonlinearity)
        # Learns exactly 3 weights: r = w1×f1 + w2×f2 + w3×f3
        if actor_layers == 0:
            self.actor = nn.Linear(state_dim, action_dim)  # 3→1 linear map
        else:
            # Fallback to MLP (but exp034 uses actor_layers=0)
            actor = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
            for _ in range(actor_layers - 1):
                actor.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            actor.append(nn.Linear(hidden_dim, action_dim))
            self.actor = nn.Sequential(*actor)
        
        # Very low noise (σ≈0.1) - parameter tuning, not exploration
        self.log_std = nn.Parameter(torch.ones(action_dim) * (-2.3))
        
        # Critic network (value of state under residual policy)
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
        return None, raw_action.cpu().numpy()  # Return raw for tanh_log_prob


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


def rollout_parallel(envs, actor_critic, num_steps):
    """Collect num_steps total across parallel environments
    
    CRITICAL INVARIANT: 
    - Actor outputs raw_residual (before tanh)
    - Env computes action = PID + process(raw_residual) 
    - PPO computes log_prob of raw_residual
    - These must all use the SAME raw_residual value
    """
    N = len(envs)
    T = num_steps // N
    
    obs = np.empty((T, N, STATE_DIM), dtype=np.float32)
    raw_act = np.empty((T, N, ACTION_DIM), dtype=np.float32)  # raw residuals for log_prob
    rew = np.empty((T, N), dtype=np.float32)
    done = np.empty((T, N), dtype=np.float32)
    
    # Initialize
    states = np.array([env.reset() for env in envs])
    episode_rewards = []
    ep_rews = np.zeros(N)
    
    for t in range(T):
        # Get raw residuals from network (before tanh)
        _, raw_residuals = actor_critic.act(states, deterministic=False)
        obs[t] = states
        raw_act[t] = raw_residuals  # Store for PPO log_prob computation
        
        # Execute environments (each env computes PID + residual internally)
        next_states = []
        for i, env in enumerate(envs):
            # Pass raw_residual to env - env handles PID + processing
            next_state, reward, terminated, truncated = env.step(raw_residuals[i, 0])
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


def evaluate_policy(actor_critic, num_routes=20, warmup_steps=50):
    """Evaluate policy with learned residual on test routes (deterministic)
    
    CRITICAL: Warm up PID before activating residual to match training manifold.
    During training, PID state is mid-flight. During eval, we must replicate this.
    """
    costs = []
    for file_path in test_files[:num_routes]:
        # Create environment
        env = TinyPhysicsGymEnv([file_path], model_onnx)
        state = env.reset()
        
        # Phase 1: Warm up PID (no residual) - let PID state settle
        for _ in range(warmup_steps):
            state, _, terminated, truncated = env.step(0.0)  # Zero residual
            if terminated or truncated:
                break
        
        # Phase 2: Run with learned residual (if episode still active)
        done = terminated or truncated
        while not done:
            # Get deterministic action from network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, _, _ = actor_critic(state_tensor)
                raw_residual = action_mean.item()  # Deterministic (no sampling)
            
            # Step environment
            state, reward, terminated, truncated = env.step(raw_residual)
            done = terminated or truncated
        
        # Get final cost from simulator
        cost_dict = env.sim.compute_cost()
        costs.append(cost_dict['total_cost'])
    
    return np.mean(costs)


class TrainingContext:
    def __init__(self):
        self.ac_cpu = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS).to('cpu')
        self.ac_device = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS).to(device)
        self.ppo = PPO(self.ac_device, PI_LR, VF_LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, VF_COEF, ENTROPY_COEF)
        
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
    
    # Compute mean per-step reward (meaningful metric)
    mean_step_reward = np.mean(rew) if rew.size > 0 else 0.0
    
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
        ctx.pbar.write(f"Epoch {epoch:3d}  step_r={mean_step_reward:7.4f}  eval={ctx.last_eval:6.1f}  best={ctx.best_cost:6.1f}  σ={s[0]:.3f}  ε={RESIDUAL_SCALE:.2f}  ⏱ {total_ms:.0f}ms")
    
    return False


def train():
    print("="*80)
    print("Experiment 037: NNFF-Style Temporal Probes")
    print("="*80)
    print(f"Architecture: u = u_PID(e) + LPF(π(temporal_probes))")
    print(f"  State: 7D temporal probes (Δlat@0.3s/0.6s/1.0s/1.5s, Δroll@0.3s/1.0s, v)")
    print(f"  Policy: LINEAR (7 weights → 1)")
    print(f"  ε (residual scale): {RESIDUAL_SCALE}")
    print(f"  Residual clip: ±{RESIDUAL_CLIP}")
    print(f"  Low-pass α: {LOWPASS_ALPHA}")
    print(f"Device: {device}")
    print(f"State dim: {STATE_DIM}, Residual dim: {ACTION_DIM}")
    print(f"Num envs: {NUM_ENVS}, Steps per epoch: {STEPS_PER_EPOCH}")
    print("="*80)
    
    ctx = TrainingContext()
    for epoch in range(MAX_EPOCHS):
        train_one_epoch(epoch, ctx)
    
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

