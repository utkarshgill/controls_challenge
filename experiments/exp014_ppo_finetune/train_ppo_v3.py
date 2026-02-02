"""
Experiment 014: PPO Fine-Tuning from BC (Version 3 - FIXED REWARDS)

FIXES:
1. Proper reward scaling (not tiny values)
2. Only compute rewards in official cost range [100:500]
3. Lower LR for fine-tuning (1e-4 instead of 3e-4)
4. Fewer PPO epochs (4 instead of 10)
5. Detailed logging
"""
import sys
from pathlib import Path

# Add project root to path (works from any directory)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import random
from tqdm import trange, tqdm
from typing import List

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX, DEL_T

# ============================================================
# Hyperparameters
# ============================================================
STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3

# PPO Hyperparameters
NUM_PARALLEL_ROLLOUTS = 8       # Collect rollouts from N routes simultaneously
NUM_ITERATIONS = 200            # Total training iterations
STEPS_PER_ITERATION = 10        # Rollouts per iteration (adjust based on episode length)

BATCH_SIZE = 2048               # Mini-batch size for PPO update
NUM_EPOCHS_PPO = 4              # Fewer epochs to prevent overfitting

LEARNING_RATE = 1e-4            # MUCH lower LR for fine-tuning from good policy
GAMMA = 0.99                    # Discount factor
GAE_LAMBDA = 0.95               # GAE parameter
CLIP_EPSILON = 0.2              # PPO clip ratio
ENTROPY_COEF = 0.0              # NO entropy bonus (we want to stay near BC)
VALUE_LOSS_COEF = 0.5           # Value loss coefficient
MAX_GRAD_NORM = 0.5             # Gradient clipping

# Evaluation
EVAL_EVERY = 10                 # Evaluate every N iterations
NUM_EVAL_ROUTES = 20            # Routes for evaluation

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Match BC device

# ============================================================
# Actor-Critic Network
# ============================================================
class ActorCritic(nn.Module):
    """Actor-Critic with BC initialization"""
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, bc_checkpoint=None):
        super().__init__()
        
        # Shared trunk
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.trunk = nn.Sequential(*layers)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(0.1))
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Load BC weights if provided
        if bc_checkpoint is not None:
            self._load_bc_weights(bc_checkpoint)
            print("✅ Loaded BC weights for actor initialization")
    
    def _load_bc_weights(self, checkpoint):
        """Load trunk and actor head from BC checkpoint"""
        bc_state_dict = checkpoint['model_state_dict']
        our_state_dict = self.state_dict()
        
        for bc_name, bc_param in bc_state_dict.items():
            our_name = bc_name.replace('mean_head', 'actor_head')
            
            if our_name in our_state_dict:
                our_state_dict[our_name].copy_(bc_param)
                print(f"  ✓ Loaded: {bc_name} -> {our_name}")
            elif bc_name in our_state_dict:
                our_state_dict[bc_name].copy_(bc_param)
                print(f"  ✓ Loaded: {bc_name}")
    
    def forward(self, state):
        """Returns action_mean, action_std, value"""
        features = self.trunk(state)
        action_mean = self.actor_head(features) * 2.0  # Scale to [-2, 2]
        action_std = self.log_std.exp()
        value = self.critic_head(features)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action and compute log_prob"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, None, value
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, -2.0, 2.0)  # Ensure valid range
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        """Evaluate actions (for PPO update)"""
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ============================================================
# PPO Controller (Wraps Policy for Official Simulator)
# ============================================================
class PPOController:
    """
    Controller interface for TinyPhysicsSimulator
    Wraps actor-critic policy and tracks trajectory data
    """
    def __init__(self, actor_critic, state_mean, state_std, device, collect_data=True, deterministic=False):
        self.actor_critic = actor_critic
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device  # Store device explicitly
        self.collect_data = collect_data
        self.deterministic = deterministic  # NEW: control sampling vs deterministic
        
        # Internal state
        self.error_integral = 0.0
        self.step_idx = 0  # Track step for cost range
        
        # Trajectory data (if collecting)
        if collect_data:
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.values = []
            self.prev_lataccel = None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Controller interface (called by simulator)"""
        # Build state vector (same as BC training)
        error = target_lataccel - current_lataccel
        self.error_integral += error
        
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel
        
        # Calculate curvatures
        future_v_egos = np.array(future_plan.v_ego, dtype=np.float32)
        if len(future_v_egos) < 50:
            pad_mode = 'constant' if len(future_v_egos) == 0 else 'edge'
            future_v_egos = np.pad(future_v_egos, (0, 50 - len(future_v_egos)), mode=pad_mode)
        else:
            future_v_egos = future_v_egos[:50]
        
        future_lataccels = np.array(future_plan.lataccel, dtype=np.float32)
        if len(future_lataccels) < 50:
            pad_mode = 'constant' if len(future_lataccels) == 0 else 'edge'
            future_lataccels = np.pad(future_lataccels, (0, 50 - len(future_lataccels)), mode=pad_mode)
        else:
            future_lataccels = future_lataccels[:50]
        
        future_roll = np.array(future_plan.roll_lataccel, dtype=np.float32)
        if len(future_roll) < 50:
            pad_mode = 'constant' if len(future_roll) == 0 else 'edge'
            future_roll = np.pad(future_roll, (0, 50 - len(future_roll)), mode=pad_mode)
        else:
            future_roll = future_roll[:50]
        
        v_ego_sq = np.maximum(future_v_egos ** 2, 1.0)
        curvatures = (future_lataccels - future_roll) / v_ego_sq
        
        # Build state vector
        state_vec = np.array([
            error,
            self.error_integral,
            v_ego,
            a_ego,
            roll_lataccel,
            *curvatures
        ], dtype=np.float32)
        
        # Normalize (MUST match BC exactly - no + 1e-8!)
        state_norm = (state_vec - self.state_mean) / self.state_std
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            action, log_prob, value = self.actor_critic.get_action(state_tensor, deterministic=self.deterministic)
            action_np = action.cpu().numpy()[0, 0]
            
            # Store data if collecting
            if self.collect_data:
                self.states.append(state_norm)
                self.actions.append(action.cpu().numpy()[0])
                self.log_probs.append(log_prob.cpu().numpy()[0] if log_prob is not None else 0)
                self.values.append(value.cpu().numpy()[0, 0])
                
                # Compute step reward ONLY in official cost range [100:500]
                if self.prev_lataccel is not None and CONTROL_START_IDX <= self.step_idx < COST_END_IDX:
                    lat_error_sq = (target_lataccel - current_lataccel) ** 2
                    jerk_sq = ((current_lataccel - self.prev_lataccel) / DEL_T) ** 2
                    
                    # Match official cost function (but as negative reward per step)
                    lat_accel_cost = lat_error_sq * 100 * 50  # scaled up
                    jerk_cost = jerk_sq * 100
                    step_cost = lat_accel_cost + jerk_cost
                    
                    # Negative reward (want to minimize cost)
                    # Scale down by 100 to make rewards reasonable magnitude
                    reward = -step_cost / 100.0
                    self.rewards.append(reward)
                
                self.prev_lataccel = current_lataccel
                self.step_idx += 1
        
        return float(np.clip(action_np, -2.0, 2.0))
    
    def get_trajectory(self):
        """Return collected trajectory data"""
        if not self.collect_data:
            return None
        
        return {
            'states': np.array(self.states[:-1]),  # Exclude last state (no reward)
            'actions': np.array(self.actions[:-1]),
            'log_probs': np.array(self.log_probs[:-1]),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values[:-1]),
        }


# ============================================================
# PPO Algorithm
# ============================================================
class PPO:
    """PPO with GAE"""
    def __init__(self, actor_critic, lr, gamma, gae_lambda, clip_epsilon, 
                 entropy_coef, value_loss_coef, max_grad_norm):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_gae(self, rewards, values, next_value):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        values_list = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_list[t + 1] - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return np.array(advantages), np.array(returns)
    
    def update(self, trajectories, batch_size, num_epochs):
        """PPO update on collected trajectories"""
        # Combine all trajectories
        all_states = np.concatenate([traj['states'] for traj in trajectories])
        all_actions = np.concatenate([traj['actions'] for traj in trajectories])
        all_old_log_probs = np.concatenate([traj['log_probs'] for traj in trajectories])
        
        # Compute advantages for each trajectory separately (to handle episode boundaries)
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            # Get next value (bootstrap from last state)
            with torch.no_grad():
                last_state = torch.FloatTensor(traj['states'][-1:]).to(device)
                _, _, next_value = self.actor_critic.get_action(last_state, deterministic=False)
                next_value = next_value.cpu().numpy()[0, 0]
            
            advantages, returns = self.compute_gae(traj['rewards'], traj['values'], next_value)
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        all_advantages = np.concatenate(all_advantages)
        all_returns = np.concatenate(all_returns)
        
        # Convert to tensors
        states = torch.FloatTensor(all_states).to(device)
        actions = torch.FloatTensor(all_actions).to(device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(device)
        advantages = torch.FloatTensor(all_advantages).to(device)
        returns = torch.FloatTensor(all_returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(num_epochs):
            # Mini-batch updates
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
                
                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                
                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }


# ============================================================
# Training Loop
# ============================================================
def collect_rollouts(actor_critic, model, data_files, state_mean, state_std, device, num_rollouts):
    """Collect rollouts using official simulator"""
    trajectories = []
    costs = []
    
    for data_file in random.sample(data_files, num_rollouts):
        # Create controller for this rollout
        controller = PPOController(actor_critic, state_mean, state_std, device, collect_data=True)
        
        # Run official rollout
        sim = TinyPhysicsSimulator(model, data_file, controller=controller)
        cost_dict = sim.rollout()
        
        # Get trajectory data
        traj = controller.get_trajectory()
        if traj is not None and len(traj['rewards']) > 0:
            trajectories.append(traj)
            costs.append(cost_dict['total_cost'])
    
    return trajectories, costs


def evaluate(actor_critic, model, data_files, state_mean, state_std, device, num_routes):
    """Evaluate policy on test routes"""
    costs = []
    
    for data_file in data_files[:num_routes]:
        controller = PPOController(actor_critic, state_mean, state_std, device, collect_data=False, deterministic=True)
        sim = TinyPhysicsSimulator(model, data_file, controller=controller)
        cost_dict = sim.rollout()
        costs.append(cost_dict['total_cost'])
    
    return np.mean(costs)


def main():
    print("="*80)
    print("Experiment 014: PPO Fine-Tuning (V3 - FIXED REWARDS & HYPERPARAMS)")
    print("="*80)
    
    # Load BC checkpoint
    bc_checkpoint_path = Path(__file__).parent.parent / "exp013_bc_from_pid/bc_best.pth"
    if not bc_checkpoint_path.exists():
        print(f"❌ BC checkpoint not found: {bc_checkpoint_path}")
        return
    
    print(f"\n✓ Loading BC checkpoint: {bc_checkpoint_path}")
    bc_checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
    
    state_mean = bc_checkpoint['state_mean']
    state_std = bc_checkpoint['state_std']
    print(f"✓ Loaded normalization stats")
    
    # Create actor-critic
    actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS, 
                               bc_checkpoint=bc_checkpoint).to(device)
    
    print(f"\n✓ Actor-Critic created:")
    print(f"  State dim: {STATE_DIM}")
    print(f"  Parameters: {sum(p.numel() for p in actor_critic.parameters()):,}")
    
    # Create PPO
    ppo = PPO(actor_critic, LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_EPSILON,
              ENTROPY_COEF, VALUE_LOSS_COEF, MAX_GRAD_NORM)
    
    # Load model and data
    model_path = Path(__file__).parent.parent.parent / "models/tinyphysics.onnx"
    model = TinyPhysicsModel(str(model_path), debug=False)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_files = sorted([str(f) for f in data_dir.glob("*.csv")])
    
    random.seed(42)
    random.shuffle(data_files)
    train_files = data_files[:15000]
    test_files = data_files[15000:]
    
    print(f"\n✓ Data loaded:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Test files: {len(test_files)}")
    
    print(f"\n{'='*80}")
    print(f"Starting PPO Training (Official Simulator)")
    print(f"{'='*80}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Rollouts per iteration: {STEPS_PER_ITERATION}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"PPO epochs: {NUM_EPOCHS_PPO}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    best_cost = float('inf')
    
    for iteration in trange(NUM_ITERATIONS, desc="Training"):
        # Collect rollouts using official simulator
        trajectories, episode_costs = collect_rollouts(
            actor_critic, model, train_files, state_mean, state_std, device, STEPS_PER_ITERATION
        )
        
        if len(trajectories) == 0:
            tqdm.write(f"Iter {iteration}: No trajectories collected, skipping update")
            continue
        
        # PPO update
        stats = ppo.update(trajectories, BATCH_SIZE, NUM_EPOCHS_PPO)
        
        # Logging
        mean_cost = np.mean(episode_costs)
        min_cost = np.min(episode_costs)
        max_cost = np.max(episode_costs)
        
        tqdm.write(f"Iter {iteration:3d} | Cost: {mean_cost:6.2f} (min: {min_cost:6.2f}, max: {max_cost:6.2f}) | "
                  f"PLoss: {stats['policy_loss']:.4f} | VLoss: {stats['value_loss']:.4f}")
        
        # Save best
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save({
                'iteration': iteration,
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'cost': best_cost,
                'state_mean': state_mean,
                'state_std': state_std,
            }, 'ppo_best_v3.pth')
        
        # Evaluation
        if iteration % EVAL_EVERY == 0 and iteration > 0:
            eval_cost = evaluate(actor_critic, model, test_files, state_mean, state_std, device, NUM_EVAL_ROUTES)
            tqdm.write(f"  Eval Cost: {eval_cost:.2f}")
    
    # Save final
    torch.save({
        'iteration': NUM_ITERATIONS,
        'model_state_dict': actor_critic.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'cost': mean_cost,
        'state_mean': state_mean,
        'state_std': state_std,
    }, 'ppo_final_v3.pth')
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"{'='*80}")
    print(f"Best cost: {best_cost:.2f}")
    print(f"✅ Saved: ppo_best_v3.pth")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

