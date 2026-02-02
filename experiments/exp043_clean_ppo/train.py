"""
Clean PPO for controls challenge.
Structure matches beautiful_lander.py exactly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController
from tqdm.contrib.concurrent import process_map


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Network architecture
state_dim, action_dim = 57, 1  # 7 current + 50 future_κ (NO PID shortcuts)
hidden_dim = 128
actor_layers, critic_layers = 4, 4

# PPO algorithm (from beautiful_lander.py)
pi_lr, vf_lr = 3e-4, 1e-3
gamma, gae_lambda = 0.99, 0.95
K_epochs = 20  # Match beautiful_lander.py exactly
eps_clip = 0.2
vf_coef, entropy_coef = 0.5, 0.002

# Training schedule
steps_per_epoch = 100_000  # Match beautiful_lander.py: 100K steps/epoch
csvs_per_epoch = 250       # 250 CSVs × 400 steps = 100K steps/epoch
max_epochs = 100           # 250 × 100 = 25K CSVs seen total
eval_interval = 1          # Eval every epoch for better visibility
num_workers = 8

# Dataset (can use up to ~20K files)
num_train_files = 10_000     # Use first N files for training
num_val_files = 1000        # Use next M files for validation

# Exploration (initial log_std)
initial_log_std_scratch = -2.3  # log(0.1) - for training from scratch
initial_log_std_bc = -1.6       # log(0.2) - for BC warm-start

# PID feedback baseline (from controllers/pid.py)
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053
pid_residual_weight = 0.1  # α: 10% PID baseline, 90% learned FF+corrections

# Temporary model checkpoint for parallel workers
TMP_MODEL_PATH = Path(__file__).parent / 'tmp_checkpoint.pt'


def tanh_log_prob(raw_action, dist):
    """
    Compute log probability for tanh-squashed actions.
    log π(tanh(x)) = log π(x) - log|det J| for change of variables
    From beautiful_lander.py
    """
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)


class ActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic learning residual over PID baseline.
    Input: 7 current + 50 future_κ = 57 (pure problem statement)
    Output: Network learns FF + corrections, combined with PID feedback
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, critic_layers):
        super().__init__()
        
        # Simple MLP with ReLU (matching beautiful_lander.py)
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
    def act(self, obs, deterministic=False):
        """Like beautiful_lander.py act() method with tanh squashing"""
        with torch.no_grad():  # No gradients needed during rollout
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, action_std, value = self(obs_t)
            
            # Sample raw action from Gaussian
            if deterministic:
                raw_action = action_mean
            else:
                raw_action = torch.distributions.Normal(action_mean, action_std).sample()
            
            # Apply tanh squashing for smooth, bounded actions
            action = torch.tanh(raw_action)
            
            return action.item(), raw_action.item(), value.item()


class PPOController(BaseController):
    """Controller learning residual over PID feedback baseline"""
    def __init__(self, actor_critic, deterministic=False):
        super().__init__()
        self.actor_critic = actor_critic
        self.deterministic = deterministic
        self.trajectory = []
        # PID state for feedback component
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Fast path: most timesteps have full 50-step future
        n_future = len(future_plan.lataccel)
        
        if n_future == 50:
            # Direct access, no padding needed (common case)
            future_lat = np.asarray(future_plan.lataccel, dtype=np.float32)
            future_roll = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
            future_v = np.asarray(future_plan.v_ego, dtype=np.float32)
        elif n_future == 0:
            # Edge case: no future plan
            future_lat = np.full(50, target_lataccel, dtype=np.float32)
            future_roll = np.full(50, state.roll_lataccel, dtype=np.float32)
            future_v = np.full(50, state.v_ego, dtype=np.float32)
        else:
            # Slow path: pad short future (only happens near episode end)
            future_lat = np.asarray(future_plan.lataccel, dtype=np.float32)
            future_roll = np.asarray(future_plan.roll_lataccel, dtype=np.float32)
            future_v = np.asarray(future_plan.v_ego, dtype=np.float32)
            pad_len = 50 - n_future
            future_lat = np.pad(future_lat, (0, pad_len), mode='edge')
            future_roll = np.pad(future_roll, (0, pad_len), mode='edge')
            future_v = np.pad(future_v, (0, pad_len), mode='edge')
        
        # Compute curvatures: κ = (lat - roll) / v² (pure geometry)
        future_v_sq = np.maximum(future_v ** 2, 25.0)
        curvature = (future_lat - future_roll) / future_v_sq
        np.clip(curvature, -1.0, 1.0, out=curvature)
        
        # Current curvature (pure geometry)
        v_sq = max(state.v_ego ** 2, 25.0)
        current_curvature = np.clip((target_lataccel - state.roll_lataccel) / v_sq, -1.0, 1.0)
        
        # Error for feedback
        error = target_lataccel - current_lataccel
        
        # PID feedback component (using official controllers/pid.py coefficients)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_feedback = PID_P * error + PID_I * self.error_integral + PID_D * error_diff
        
        # Observation: 7 current + 50 future_κ = 57 (pure problem statement)
        obs = np.empty(57, dtype=np.float32)
        obs[0] = target_lataccel
        obs[1] = current_lataccel
        obs[2] = error
        obs[3] = state.v_ego
        obs[4] = state.a_ego
        obs[5] = state.roll_lataccel
        obs[6] = current_curvature
        obs[7:57] = curvature  # 50 future curvatures
        
        # Network learns FF + corrections (tanh-squashed)
        network_action, raw_network_action, value = self.actor_critic.act(obs, deterministic=self.deterministic)
        
        # Combine: 10% PID baseline + 90% learned
        action = pid_residual_weight * pid_feedback + network_action
        
        self.trajectory.append({
            'obs': obs,
            'action': raw_network_action,  # Store raw network action for PPO updates
            'value': value,
            'target_lat': target_lataccel,
            'current_lat': current_lataccel,
        })
        
        return action


class PPO:
    """PPO trainer - matches beautiful_lander.py structure"""
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.vf_coef, self.entropy_coef = eps_clip, vf_coef, entropy_coef
        
        self.pi_optimizer = optim.Adam(
            list(actor_critic.actor.parameters()) + [actor_critic.log_std], 
            lr=pi_lr
        )
        self.vf_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)
    
    def compute_advantages(self, rewards, values, dones):
        """GAE for multiple trajectories - vectorized for speed"""
        all_advantages = []
        all_returns = []
        
        # Process each episode (can't fully vectorize due to variable lengths and GAE recursion)
        for rew, val, done in zip(rewards, values, dones):
            T = len(rew)
            advantages = np.zeros(T, dtype=np.float32)
            
            # Backward pass for GAE
            gae = 0.0
            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = val[t + 1]
                
                delta = rew[t] + self.gamma * next_value * (1 - done[t]) - val[t]
                gae = delta + self.gamma * self.lamda * (1 - done[t]) * gae
                advantages[t] = gae
            
            returns = advantages + val
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, all_obs, all_actions, all_rewards, all_values, all_dones):
        """Single update on batch - like beautiful_lander.py with minibatch shuffling"""
        obs = torch.FloatTensor(np.concatenate(all_obs, axis=0))
        actions = torch.FloatTensor(np.concatenate(all_actions, axis=0)).unsqueeze(-1)
        
        advantages, returns = self.compute_advantages(all_rewards, all_values, all_dones)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)
        
        with torch.no_grad():
            action_means, action_stds, _ = self.actor_critic(obs)
            dist = torch.distributions.Normal(action_means, action_stds)
            # Use tanh-corrected log prob (actions are raw, pre-tanh)
            old_logprobs = tanh_log_prob(actions, dist)
        
        # Multiple epochs with minibatching (standard PPO)
        num_samples = len(obs)
        batch_size = 5000  # Match beautiful_lander.py
        
        for _ in range(self.K_epochs):
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            # Process in minibatches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                
                # Forward on minibatch
                action_means, action_stds, state_values = self.actor_critic(obs[idx])
                dist = torch.distributions.Normal(action_means, action_stds)
                # Use tanh-corrected log prob (actions are raw, pre-tanh)
                logprobs = tanh_log_prob(actions[idx], dist)
                
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantages_t[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Normalize returns for stable critic training (prevents divergence from unbounded returns)
                returns_batch = returns_t[idx]
                returns_normalized = (returns_batch - returns_batch.mean()) / (returns_batch.std() + 1e-8)
                critic_loss = F.mse_loss(state_values.squeeze(-1), returns_normalized)
                
                # Compute entropy of tanh-transformed distribution
                # H[tanh(X)] = H[X] - E[log(1 - tanh²(X))]
                gaussian_entropy = dist.entropy().sum(-1)  # [batch_size]
                actions_squashed = torch.tanh(actions[idx])
                jacobian_correction = torch.log(1 - actions_squashed**2 + 1e-6).sum(-1)  # [batch_size]
                entropy = (gaussian_entropy - jacobian_correction).mean()
                
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (like beautiful_lander.py)
                pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std],
                    max_norm=0.5
                )
                vf_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.critic.parameters(), 
                    max_norm=0.5
                )
                
                self.pi_optimizer.step()
                self.vf_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'pi_grad_norm': pi_grad_norm.item(),
            'vf_grad_norm': vf_grad_norm.item(),
        }


def compute_rewards(trajectory):
    """
    Per-step rewards matching EXACT official cost formula (vectorized).
    
    Official: total_cost = (lat_accel_cost × 50) + jerk_cost
    where:
        lat_accel_cost = mean((target-pred)²) × 100
        jerk_cost = mean((diff(pred)/dt)²) × 100
        
    Per-step version: reward_t = -(lat_cost_t + jerk_cost_t)
    """
    from tinyphysics import LAT_ACCEL_COST_MULTIPLIER, DEL_T
    
    T = len(trajectory)
    
    # Vectorized: extract arrays once
    target_lat = np.array([t['target_lat'] for t in trajectory], dtype=np.float32)
    current_lat = np.array([t['current_lat'] for t in trajectory], dtype=np.float32)
    
    # Lataccel cost (vectorized)
    lat_error = target_lat - current_lat
    lat_cost = (lat_error ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
    
    # Jerk cost (vectorized with diff)
    lataccel_diff = np.diff(current_lat, prepend=current_lat[0])  # prepend makes first diff=0
    jerk = lataccel_diff / DEL_T
    jerk_cost = (jerk ** 2) * 100
    
    # Reward is negative cost (scaled down for stability)
    rewards = -(lat_cost + jerk_cost) / 100.0
    
    return rewards


def rollout_csv(controller, model, csv_path):
    """Collect one trajectory"""
    controller.trajectory = []
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
    cost_dict = sim.rollout()
    
    # Pre-allocate arrays instead of list comprehension
    T = len(controller.trajectory)
    obs = np.empty((T, 57), dtype=np.float32)
    actions = np.empty(T, dtype=np.float32)
    values = np.empty(T, dtype=np.float32)
    
    for i, t in enumerate(controller.trajectory):
        obs[i] = t['obs']
        actions[i] = t['action']
        values[i] = t['value']
    
    rewards = compute_rewards(controller.trajectory)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    
    return obs, actions, rewards, values, dones, cost_dict['total_cost']


def rollout_worker(args):
    """Worker for parallel collection - loads from disk"""
    csv_path, model_path_str, checkpoint_path = args
    
    model = TinyPhysicsModel(model_path_str, debug=False)
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers)
    # Faster loading: map_location avoids device transfers
    actor_critic.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location='cpu'))
    actor_critic.eval()
    controller = PPOController(actor_critic, deterministic=False)
    
    return rollout_csv(controller, model, csv_path)


class TrainingContext:
    """Training context - like beautiful_lander.py"""
    def __init__(self):
        data_dir = Path(__file__).parent.parent.parent / 'data'
        self.model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
        
        # Dataset split (configurable)
        all_files = sorted(list(data_dir.glob('*.csv')))
        self.train_files = all_files[:num_train_files]
        self.val_files = all_files[num_train_files:num_train_files + num_val_files]
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers)
        
        # Load BC weights if available (warm-start)
        bc_path = Path(__file__).parent / 'bc_init.pt'
        if bc_path.exists():
            print("Loading BC checkpoint for warm-start...")
            self.actor_critic.load_state_dict(torch.load(bc_path, weights_only=False))
            print("  ✓ Warm-started from BC")
            # Start with higher exploration after BC
            self.actor_critic.log_std.data.fill_(initial_log_std_bc)
        else:
            print("No BC checkpoint found, starting from scratch")
            # Lower exploration from scratch
            self.actor_critic.log_std.data.fill_(initial_log_std_scratch)
        
        self.ppo = PPO(self.actor_critic, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, vf_coef, entropy_coef)
        
        self.last_val_cost = float('inf')
        self.best_val_cost = float('inf')
        self.best_epoch = -1


def train_one_epoch(epoch, ctx):
    """One epoch: collect batch, update once. Returns False (no early stopping yet)."""
    t0 = time.time()
    
    # Save model once for workers to load
    torch.save(ctx.actor_critic.state_dict(), TMP_MODEL_PATH)
    
    # Collect trajectories in parallel (silent to avoid progress bar conflicts)
    csv_batch = random.sample(ctx.train_files, csvs_per_epoch)
    args = [(csv, str(ctx.model_path), str(TMP_MODEL_PATH)) for csv in csv_batch]
    
    print(f"Epoch {epoch}: Collecting {csvs_per_epoch} trajectories...", end='', flush=True)
    results = process_map(rollout_worker, args, max_workers=num_workers, chunksize=10, disable=True)
    t_collect = time.time() - t0
    
    all_obs = [r[0] for r in results]
    all_actions = [r[1] for r in results]
    all_rewards = [r[2] for r in results]
    all_values = [r[3] for r in results]
    all_dones = [r[4] for r in results]
    all_costs = [r[5] for r in results]
    
    print(f" done ({t_collect:.1f}s). Updating PPO...", end='', flush=True)
    
    # Diagnostics: Input feature statistics (every 10 epochs)
    if epoch % 10 == 0:
        obs_concat = np.concatenate(all_obs, axis=0)
        feature_names = ['target_lat', 'current_lat', 'error', 'v_ego', 'a_ego', 'roll', 'cur_κ', 'future_κ']
        feature_ranges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,57)]
        
        print("\n  [DIAGNOSTICS] Input Feature Statistics:")
        for name, (start, end) in zip(feature_names, feature_ranges):
            feat = obs_concat[:, start:end]
            print(f"    {name:12s}: mean={feat.mean():7.3f}  std={feat.std():6.3f}  max={np.abs(feat).max():7.3f}")
    
    # PPO update
    t_update_start = time.time()
    update_info = ctx.ppo.update(all_obs, all_actions, all_rewards, all_values, all_dones)
    t_update = time.time() - t_update_start
    
    print(f" done ({t_update:.1f}s).")
    
    # Evaluate and log
    mean_cost = np.mean(all_costs)
    std = ctx.actor_critic.log_std.exp().item()
    
    if epoch % eval_interval == 0:
        # Reuse checkpoint saved at start of epoch
        ctx.last_val_cost = evaluate_policy(ctx.actor_critic, ctx.model_path, ctx.val_files, num_files=50, save_checkpoint=False)
        
        # Track best validation and save checkpoint
        marker = ""
        if ctx.last_val_cost < ctx.best_val_cost:
            ctx.best_val_cost = ctx.last_val_cost
            ctx.best_epoch = epoch
            best_path = Path(__file__).parent / 'best_model.pt'
            torch.save(ctx.actor_critic.state_dict(), best_path)
            marker = " ★ NEW BEST!"
        
        # Print diagnostics every 10 epochs
        if epoch % 10 == 0:
            print(f"  → Train: {mean_cost:7.2f} | Val: {ctx.last_val_cost:7.2f} | Std: {std:.4f}{marker}")
            print(f"      [LOSS] actor={update_info['actor_loss']:.4f}  critic={update_info['critic_loss']:.2f}  entropy={update_info['entropy']:.4f}")
            print(f"      [GRADS] π_norm={update_info['pi_grad_norm']:.3f}  vf_norm={update_info['vf_grad_norm']:.3f}")
            
            # Weight diagnostics: which inputs is the network using?
            W = ctx.actor_critic.actor[0].weight.data  # [128, 57]
            w_current = W[:, :7].norm().item()          # Current state (7 features)
            w_future_k = W[:, 7:57].norm().item()       # Future κ (50 features)
            
            # Per-timestep future importance (which future timesteps matter most?)
            future_k_per_step = torch.stack([W[:, 7+i].norm() for i in range(50)])
            near_term = future_k_per_step[:10].mean().item()  # 0-1s
            mid_term = future_k_per_step[10:30].mean().item()  # 1-3s
            far_term = future_k_per_step[30:50].mean().item()  # 3-5s
            
            # Activation check (ReLU doesn't saturate, but track dead neurons)
            with torch.no_grad():
                sample_obs = torch.FloatTensor(all_obs[0][:100])  # 100 samples
                h1_post = ctx.actor_critic.actor[1](ctx.actor_critic.actor[0](sample_obs))  # Linear -> ReLU
                dead_neurons = (h1_post.max(0)[0] == 0).float().mean().item()
                mean_activation = h1_post.mean().item()
            
            print(f"      [WEIGHTS] current={w_current:.2f}  future_κ={w_future_k:.2f}")
            print(f"      [FUTURE_κ] near(0-1s)={near_term:.2f}  mid(1-3s)={mid_term:.2f}  far(3-5s)={far_term:.2f}")
            print(f"      [ACTIVATION] dead_neurons={dead_neurons:.1%}  mean={mean_activation:.3f}\n")
        else:
            print(f"  → Train: {mean_cost:7.2f} | Val: {ctx.last_val_cost:7.2f} | Std: {std:.4f}{marker}\n")
    else:
        print(f"  → Train: {mean_cost:7.2f} | Std: {std:.4f}\n")
    
    return False  # No early stopping


def eval_worker(args):
    """Worker for parallel evaluation - loads from disk"""
    csv_path, model_path_str, checkpoint_path = args
    
    model = TinyPhysicsModel(model_path_str, debug=False)
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, actor_layers, critic_layers)
    # Faster loading: map_location avoids device transfers
    actor_critic.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location='cpu'))
    actor_critic.eval()
    
    controller = PPOController(actor_critic, deterministic=True)
    _, _, _, _, _, cost = rollout_csv(controller, model, csv_path)
    return cost


def evaluate_policy(actor_critic, model_path, csv_files, num_files=10, save_checkpoint=True):
    """Evaluate deterministically in parallel"""
    # Save model once for workers (skip if already saved this epoch)
    if save_checkpoint:
        torch.save(actor_critic.state_dict(), TMP_MODEL_PATH)
    
    # Use FIXED validation set (first num_files) for consistent comparison
    eval_files = csv_files[:num_files]
    args = [(csv, str(model_path), str(TMP_MODEL_PATH)) for csv in eval_files]
    
    results = process_map(eval_worker, args, max_workers=num_workers, chunksize=5, disable=True)
    
    return float(np.mean(results))


def train():
    """Main training loop - exact structure from beautiful_lander.py"""
    ctx = TrainingContext()
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training files:   {len(ctx.train_files):,} (from {num_train_files:,} available)")
    print(f"Validation files: {len(ctx.val_files):,} (from {num_val_files:,} available)")
    print(f"Steps per epoch:  {steps_per_epoch:,}")
    print(f"CSVs per epoch:   {csvs_per_epoch}")
    print(f"Total epochs:     {max_epochs}")
    print(f"Parallel workers: {num_workers}")
    print(f"Initial std:      {ctx.actor_critic.log_std.exp().item():.4f} (log_std={ctx.actor_critic.log_std.item():.2f})")
    print("="*60)
    print("Starting training...\n")
    
    for epoch in range(max_epochs):
        if train_one_epoch(epoch, ctx):
            break
    
    print("="*60)
    print("Training complete!")
    final_cost = evaluate_policy(ctx.actor_critic, ctx.model_path, ctx.val_files, num_files=50)
    print(f"Final validation cost (50 files): {final_cost:.2f}")
    print(f"Best validation: {ctx.best_val_cost:.2f} at epoch {ctx.best_epoch}")
    print(f"  → Best model saved to: best_model.pt")
    
    torch.save(ctx.actor_critic.state_dict(), 'ppo_model.pt')
    print("  → Final model saved to: ppo_model.pt")
    
    # Cleanup temp file
    if TMP_MODEL_PATH.exists():
        TMP_MODEL_PATH.unlink()


if __name__ == '__main__':
    train()
