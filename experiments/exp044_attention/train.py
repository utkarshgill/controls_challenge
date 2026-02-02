"""
Exp044: Simple Cross-Attention for Feedforward Preview
Architecture: Current state queries future_κ to dynamically select relevant preview horizon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys
from collections import namedtuple
import time
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, LAT_ACCEL_COST_MULTIPLIER, DEL_T
from controllers import BaseController

# Hyperparameters (matching beautiful_lander.py + tuned for our problem)
state_dim, action_dim = 57, 1  # 7 current + 50 future_κ
hidden_dim = 128
actor_layers, critic_layers = 4, 4
pi_lr, vf_lr = 3e-4, 1e-3
gamma, gae_lambda = 0.99, 0.95
K_epochs = 20
eps_clip = 0.2
vf_coef, entropy_coef = 0.5, 0.002

# Attention hyperparameters
attention_dim = 32  # Dimension for Query/Key/Value projections

# Training
total_epochs = 200  # 200 epochs × 50 CSVs = 10K CSVs total
csvs_per_epoch = 50  # 50 CSVs × 400 steps = 20K steps/epoch (memory safe)
num_train_files = 5000
num_val_files = 100
batch_size = 2048  # Smaller batches for attention
reward_scale = 1.0

# Policy initialization
initial_log_std_scratch = -0.7
initial_log_std_bc = -1.6

# PID feedback baseline (from controllers/pid.py)
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053
pid_residual_weight = 0.1  # α: 10% PID baseline, 90% learned

# Paths
TMP_MODEL_PATH = Path(__file__).parent / 'tmp_checkpoint.pt'


class ActorCritic(nn.Module):
    """
    Minimal attention architecture: Query current state, attend to future_κ
    
    FB: 10% PID baseline (external)
    FF: Attention over future_κ (learned)
    """
    def __init__(self, state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers):
        super().__init__()
        self.attention_dim = attention_dim
        
        # Query: encode current state (includes error for context)
        self.query_net = nn.Linear(7, attention_dim)
        
        # Keys/Values: encode each future_κ
        self.key_net = nn.Linear(1, attention_dim)
        self.value_net = nn.Linear(1, attention_dim)
        
        # Actor: attended features → action (add small MLP for stability)
        self.actor = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic: attended features → value (add small MLP for stability)
        self.critic = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly (will be overwritten by BC checkpoint anyway)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Args:
            state: [batch, 57] - [current(7), future_κ(50)]
        Returns:
            action_mean, action_std, value
        """
        # Clamp inputs for stability
        state = torch.clamp(state, -100, 100)
        
        current = state[:, :7]
        future_κ = state[:, 7:57]
        
        # Query: "Given my current state (error, v, ...), what future matters?"
        query = self.query_net(current)  # [batch, attention_dim]
        query = torch.clamp(query, -10, 10)
        
        # Keys/Values: encode each future curvature
        keys = self.key_net(future_κ.unsqueeze(-1))  # [batch, 50, attention_dim]
        values = self.value_net(future_κ.unsqueeze(-1))  # [batch, 50, attention_dim]
        keys = torch.clamp(keys, -10, 10)
        values = torch.clamp(values, -10, 10)
        
        # Attention: which future timesteps are relevant?
        scores = torch.bmm(keys, query.unsqueeze(-1)) / np.sqrt(self.attention_dim)  # [batch, 50, 1]
        scores = torch.clamp(scores, -10, 10)  # Prevent extreme values
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)  # [batch, 50]
        
        # Weighted sum of future
        attended = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch, attention_dim]
        attended = torch.clamp(attended, -10, 10)  # Prevent extreme attended features
        
        # Output
        action_mean = self.actor(attended)
        action_mean = torch.clamp(action_mean, -10, 10)  # Prevent extreme actions
        action_std = self.log_std.exp().clamp(min=1e-6, max=2.0)  # Bound std
        value = self.critic(attended)
        
        return action_mean, action_std, value
    
    @torch.inference_mode()
    def act(self, obs, deterministic=False):
        """Action selection for rollouts"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, action_std, value = self(obs_t)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            
            if deterministic:
                raw_action = action_mean
            else:
                raw_action = dist.sample()
            
            action = torch.tanh(raw_action)  # Squash to [-1, 1]
            
            return action.item(), raw_action.item(), value.item()


def tanh_log_prob(raw_action, action_mean, action_std):
    """Compute log prob with tanh squashing correction"""
    normal_dist = torch.distributions.Normal(action_mean, action_std)
    log_prob = normal_dist.log_prob(raw_action)
    log_prob -= torch.log(1 - torch.tanh(raw_action)**2 + 1e-6)
    return log_prob.sum(dim=-1)


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
        
        # Actor parameters: attention + actor head
        actor_params = (
            list(actor_critic.query_net.parameters()) +
            list(actor_critic.key_net.parameters()) +
            list(actor_critic.value_net.parameters()) +
            list(actor_critic.actor.parameters()) + 
            [actor_critic.log_std]
        )
        self.pi_optimizer = optim.Adam(actor_params, lr=pi_lr)
        self.vf_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)
    
    def compute_advantages(self, rewards, values, dones):
        """GAE for multiple trajectories"""
        all_advantages = []
        all_returns = []
        
        for rew, val, done in zip(rewards, values, dones):
            T = len(rew)
            advantages = np.zeros(T, dtype=np.float32)
            returns = np.zeros(T, dtype=np.float32)
            
            next_value = 0.0
            next_advantage = 0.0
            
            for t in reversed(range(T)):
                if done[t]:
                    next_value = 0.0
                    next_advantage = 0.0
                
                delta = rew[t] + self.gamma * next_value - val[t]
                advantages[t] = delta + self.gamma * self.lamda * next_advantage
                returns[t] = advantages[t] + val[t]
                
                next_value = val[t]
                next_advantage = advantages[t]
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        return all_advantages, all_returns
    
    def update(self, trajectories):
        """PPO update on batch of trajectories"""
        # Flatten trajectories
        all_obs = [traj['obs'] for traj in trajectories]
        all_actions = [traj['actions'] for traj in trajectories]
        all_values = [traj['values'] for traj in trajectories]
        all_rewards = [traj['rewards'] for traj in trajectories]
        all_dones = [traj['dones'] for traj in trajectories]
        
        # Compute advantages
        all_advantages, all_returns = self.compute_advantages(all_rewards, all_values, all_dones)
        
        # Concatenate everything
        obs_batch = torch.FloatTensor(np.concatenate(all_obs))
        actions_batch = torch.FloatTensor(np.concatenate(all_actions))
        advantages_batch = torch.FloatTensor(np.concatenate(all_advantages))
        returns_batch = torch.FloatTensor(np.concatenate(all_returns))
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        # Compute old log probs
        with torch.no_grad():
            action_mean, action_std, _ = self.actor_critic(obs_batch)
            old_logprobs = tanh_log_prob(actions_batch, action_mean, action_std)
        
        # PPO update for K epochs
        for _ in range(self.K_epochs):
            # Shuffle data
            indices = torch.randperm(len(obs_batch))
            obs_batch = obs_batch[indices]
            actions_batch = actions_batch[indices]
            old_logprobs = old_logprobs[indices]
            advantages_batch = advantages_batch[indices]
            returns_batch = returns_batch[indices]
            
            # Forward pass
            action_mean, action_std, values = self.actor_critic(obs_batch)
            logprobs = tanh_log_prob(actions_batch, action_mean, action_std)
            
            # Actor loss: clipped surrogate
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss: MSE with normalized returns
            returns_norm = (returns_batch - returns_batch.mean()) / (returns_batch.std() + 1e-8)
            critic_loss = F.mse_loss(values.squeeze(), returns_norm)
            
            # Entropy bonus
            entropy = torch.distributions.Normal(action_mean, action_std).entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Update actor
            self.pi_optimizer.zero_grad()
            (actor_loss + entropy_loss).backward(retain_graph=True)  # Keep graph for critic
            # Clip gradients for all actor parameters (attention + actor head)
            actor_params = (
                list(self.actor_critic.query_net.parameters()) +
                list(self.actor_critic.key_net.parameters()) +
                list(self.actor_critic.value_net.parameters()) +
                list(self.actor_critic.actor.parameters()) + 
                [self.actor_critic.log_std]
            )
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
            self.pi_optimizer.step()
            
            # Update critic
            self.vf_optimizer.zero_grad()
            critic_loss.backward()  # Now can backward through shared attention layers
            torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_norm=0.5)
            self.vf_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }


def compute_rewards(lataccels, actions):
    """Convert costs to rewards (matching official cost function)"""
    lat_cost = LAT_ACCEL_COST_MULTIPLIER * np.abs(lataccels)
    jerk_cost = np.abs(np.diff(lataccels, prepend=lataccels[0])) / DEL_T
    costs = lat_cost + jerk_cost
    rewards = -costs * reward_scale
    return rewards


def rollout_worker(args):
    """Worker for parallel trajectory collection"""
    csv_path, checkpoint_path = args
    
    model = TinyPhysicsModel(str(Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'), debug=False)
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers)
    actor_critic.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    actor_critic.eval()
    
    controller = PPOController(actor_critic, deterministic=False)
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
    sim.rollout()
    
    traj = controller.trajectory
    T = len(traj)
    
    obs = np.array([t['obs'] for t in traj])
    actions = np.array([t['action'] for t in traj])
    values = np.array([t['value'] for t in traj])
    lataccels = np.array([t['current_lat'] for t in traj])
    
    rewards = compute_rewards(lataccels, actions)
    dones = np.zeros(T, dtype=bool)
    dones[-1] = True
    
    return {
        'obs': obs,
        'actions': actions,
        'values': values,
        'rewards': rewards,
        'dones': dones,
        'cost': sim.compute_cost()['total_cost']
    }


def eval_worker(args):
    """Worker for parallel evaluation"""
    csv_path, checkpoint_path = args
    
    model = TinyPhysicsModel(str(Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'), debug=False)
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers)
    actor_critic.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    actor_critic.eval()
    
    controller = PPOController(actor_critic, deterministic=True)
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
    sim.rollout()
    
    return sim.compute_cost()['total_cost']


def train_one_epoch(ctx):
    """Train for one epoch (collection + update)"""
    # Save temp checkpoint for workers
    torch.save(ctx.actor_critic.state_dict(), TMP_MODEL_PATH)
    
    # Collect trajectories in parallel
    train_files = np.random.choice(ctx.train_csv_files, size=csvs_per_epoch, replace=False)
    args = [(f, TMP_MODEL_PATH) for f in train_files]
    
    trajectories = process_map(rollout_worker, args, max_workers=8, chunksize=10, desc="Collecting", leave=False)
    
    # Compute train cost
    train_costs = [t['cost'] for t in trajectories]
    train_cost = np.mean(train_costs)
    
    # PPO update
    update_info = ctx.ppo.update(trajectories)
    
    # Evaluate on validation set
    eval_args = [(f, TMP_MODEL_PATH) for f in ctx.eval_files]
    val_costs = process_map(eval_worker, eval_args, max_workers=8, chunksize=10, desc="Evaluating", leave=False)
    val_cost = np.mean(val_costs)
    
    # Track best model
    if val_cost < ctx.best_val_cost:
        ctx.best_val_cost = val_cost
        torch.save(ctx.actor_critic.state_dict(), Path(__file__).parent / 'best_model.pt')
        return train_cost, val_cost, update_info, True
    
    return train_cost, val_cost, update_info, False


def main():
    print("=== Exp044: Simple Attention for Feedforward Preview ===\n")
    
    # Setup
    data_dir = Path(__file__).parent.parent.parent / 'data'
    csv_files = sorted(data_dir.glob('*.csv'))
    
    train_csv_files = csv_files[:num_train_files]
    eval_files = csv_files[num_train_files:num_train_files + num_val_files]
    
    print(f"Train files: {len(train_csv_files)}")
    print(f"Val files: {len(eval_files)}")
    print(f"Architecture: Minimal Attention (Query current, attend to future_κ)")
    print(f"PID residual: α={pid_residual_weight} (FB baseline)")
    print(f"CSVs per epoch: {csvs_per_epoch} (~20K steps/epoch)")
    print()
    
    # Initialize network
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, attention_dim, actor_layers, critic_layers)
    
    # Load BC checkpoint if available (only actor parts, critic trains from scratch)
    bc_checkpoint = Path(__file__).parent / 'bc_init.pt'
    if bc_checkpoint.exists():
        print(f"Loading BC warm-start from {bc_checkpoint.name}")
        checkpoint = torch.load(bc_checkpoint, weights_only=False)
        # Only load attention + actor weights, not critic (critic was never trained in BC)
        actor_critic.query_net.load_state_dict({k.replace('query_net.', ''): v for k, v in checkpoint.items() if k.startswith('query_net.')})
        actor_critic.key_net.load_state_dict({k.replace('key_net.', ''): v for k, v in checkpoint.items() if k.startswith('key_net.')})
        actor_critic.value_net.load_state_dict({k.replace('value_net.', ''): v for k, v in checkpoint.items() if k.startswith('value_net.')})
        actor_critic.actor.load_state_dict({k.replace('actor.', ''): v for k, v in checkpoint.items() if k.startswith('actor.')})
        actor_critic.log_std.data.copy_(checkpoint['log_std'])
        actor_critic.log_std.data.fill_(initial_log_std_bc)
        print("  ✓ Loaded attention + actor from BC, critic initialized randomly")
    else:
        print("Training from scratch (no BC checkpoint found)")
        actor_critic.log_std.data.fill_(initial_log_std_scratch)
    
    # PPO trainer
    ppo = PPO(actor_critic, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, vf_coef, entropy_coef)
    
    # Training context
    class Context:
        pass
    ctx = Context()
    ctx.actor_critic = actor_critic
    ctx.ppo = ppo
    ctx.train_csv_files = train_csv_files
    ctx.eval_files = eval_files
    ctx.best_val_cost = float('inf')
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        train_cost, val_cost, update_info, is_best = train_one_epoch(ctx)
        elapsed = time.time() - t0
        
        std = actor_critic.log_std.exp().item()
        
        best_marker = " ★ NEW BEST!" if is_best else ""
        print(f"Epoch {epoch}: Train: {train_cost:7.2f} | Val: {val_cost:7.2f} | "
              f"Std: {std:.4f} | Time: {elapsed:.1f}s{best_marker}")
    
    print(f"\n✓ Training complete! Best val cost: {ctx.best_val_cost:.2f}")
    TMP_MODEL_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
