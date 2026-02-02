"""
Experiment 041: Split Feedback + Feedforward Architecture

Architecture:
- Feedback: EXACT same as exp040 - single linear neuron [error, ei, ed] → action
- Feedforward: 1D Conv + MLP on 50D curvatures (UNSCALED)
- Curvature = (lataccel - roll) / v² for EACH timestep (per-step v_ego)
- Range: ~[-0.0003, +0.0003] (raw, NO artificial scaling)
- Conv captures temporal patterns in curvature sequence
- Total action = FB + FF

Key Fixes (inspired by beautiful_lander.py + exp040):
1. SPARSE rewards: total_cost at episode end ONLY (like exp040 - proven to work!)
2. GAE bootstrap: from last state_value (not zero)
3. Hyperparameters: pi_lr=3e-4 (↑3x), K_epochs=10 (↑2x), entropy_coef=0.001 (↓10x)
4. Input: Curvature with per-timestep v_ego, NO artificial scaling
5. Code structure: TrainingContext + train_one_epoch pattern for clean iteration

Training:
Phase 1: BC trains FB only to clone PID (FF stays at zero)
Phase 2: PPO trains both FB + FF together

Hypothesis: Adding FF anticipation improves over exp040's 76.6
Expected: < 70 cost
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
import os
import pickle
from tqdm import trange, tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, run_rollout
from controllers import BaseController

# PID constants (baseline)
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053

# Training config
num_envs = int(os.getenv('NUM_ENVS', 8))
max_epochs_bc = 100
max_epochs_ppo = 200
log_interval = 10
eval_interval = 10

# PPO hyperparameters
batch_size = 2048
K_epochs = 10  # More updates per batch (was 5, beautiful_lander uses 20)
pi_lr = 3e-4  # Higher learning rate (was 1e-4)
vf_lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
vf_coef = 0.5
entropy_coef = 0.001  # Lower entropy = less forced exploration (was 0.01)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load data
all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

all_files = list(DATA_PATH.glob('*.csv'))
random.shuffle(all_files)
train_files = [f for f in all_files if f not in eval_files][:15000]

# Load simulator
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)


# ============================================================
# SPLIT ARCHITECTURE: FB (1 neuron) + FF (MLP)
# ============================================================

class FeedbackModule(nn.Module):
    """EXACT same as exp040: single linear neuron, no bias
    
    Input: [error, error_integral, error_diff] (3D)
    Output: feedback action (scalar)
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)
    
    def forward(self, fb_state):
        return self.linear(fb_state)


class FeedforwardModule(nn.Module):
    """1D Conv on future curvatures to capture temporal patterns
    
    Input: [curvature_0, curvature_1, ..., curvature_49] (50D)
    Output: feedforward action (scalar)
    """
    def __init__(self):
        super().__init__()
        # 1D Conv to extract temporal features
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)  # [batch, 16, 50]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2) # [batch, 32, 25]
        
        # Small MLP to decode
        self.fc1 = nn.Linear(32 * 25, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, curvatures):
        # curvatures: [batch, 50]
        x = curvatures.unsqueeze(1)  # [batch, 1, 50] - add channel dimension
        x = torch.relu(self.conv1(x))  # [batch, 16, 50]
        x = torch.relu(self.conv2(x))  # [batch, 32, 25]
        x = x.flatten(1)  # [batch, 800]
        x = torch.relu(self.fc1(x))  # [batch, 64]
        ff_action = self.fc2(x)  # [batch, 1]
        return ff_action


class SplitActor(nn.Module):
    """FB + FF with learnable log_std for PPO"""
    def __init__(self):
        super().__init__()
        self.fb = FeedbackModule()
        self.ff = FeedforwardModule()
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, fb_state, curvatures):
        """
        fb_state: [batch, 3] = [error, ei, ed]
        curvatures: [batch, 50] = future curvatures
        """
        fb_action = self.fb(fb_state)
        ff_action = self.ff(curvatures)
        action_mean = fb_action + ff_action
        action_std = self.log_std.exp()
        return action_mean, action_std, fb_action, ff_action
    
    @torch.inference_mode()
    def act(self, fb_state, curvatures, deterministic=False):
        """For rollouts - single step"""
        if not isinstance(fb_state, torch.Tensor):
            fb_state = torch.tensor(fb_state, dtype=torch.float32)
        if not isinstance(curvatures, torch.Tensor):
            curvatures = torch.tensor(curvatures, dtype=torch.float32)
        
        if fb_state.ndim == 1:
            fb_state = fb_state.unsqueeze(0)
        if curvatures.ndim == 1:
            curvatures = curvatures.unsqueeze(0)
        
        mean, std, fb, ff = self(fb_state, curvatures)
        
        if deterministic:
            action = mean
        else:
            action = torch.distributions.Normal(mean, std).sample()
        
        return action.cpu().numpy()[0, 0], fb.item(), ff.item()


class Critic(nn.Module):
    """Value function: sees FB state"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, fb_state):
        x = torch.relu(self.fc1(fb_state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class SplitController(BaseController):
    """Controller using split FB+FF"""
    def __init__(self, actor, deterministic=True):
        self.actor = actor
        self.deterministic = deterministic
        self.error_integral = 0
        self.prev_error = 0
    
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Feedback state (same as exp040)
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        fb_state = np.array([error, self.error_integral, error_diff], dtype=np.float32)
        
        # Feedforward: compute CURVATURES from future_plan (SCALED UP)
        # state is State namedtuple: .v_ego, .a_ego, .roll_lataccel
        # future_plan is FuturePlan namedtuple: .lataccel, .roll_lataccel, .v_ego, .a_ego
        
        # Pad to 50 elements (may be shorter at end of trajectory)
        future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
        future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
        future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
        
        # Curvature = (lataccel - roll) / v² for EACH timestep (per-step v_ego)
        # NO scaling (raw curvature ~0.0003)
        v_squared = np.maximum(future_v_ego ** 2, 1.0)
        curvatures = ((future_lataccel - future_roll) / v_squared).astype(np.float32)
        
        action, fb, ff = self.actor.act(fb_state, curvatures, deterministic=self.deterministic)
        return action


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_pid_demonstrations(data_files, num_files=1000, use_cache=True):
    """Collect PID demonstrations for BC (FB only, same as exp040)"""
    cache_file = Path(__file__).parent / f"pid_demonstrations_{num_files}.pkl"
    
    if use_cache and cache_file.exists():
        print(f"\n{'='*60}")
        print("Loading PID Demonstrations from Cache")
        print(f"{'='*60}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        states, actions = data['states'], data['actions']
        print(f"✓ Loaded {len(states):,} samples from {cache_file.name}")
        
        # Verify
        print("\nVerifying cached data (first 3 samples):")
        for i in range(min(3, len(states))):
            error, integral, diff = states[i]
            action = actions[i, 0]
            expected = PID_P * error + PID_I * integral + PID_D * diff
            print(f"  Sample {i}: action={action:.6f}, expected={expected:.6f}, diff={abs(action-expected):.8f}")
        
        return states, actions
    
    print(f"\n{'='*60}")
    print("Phase 1: Collecting PID Demonstrations")
    print(f"{'='*60}")
    
    from controllers.pid import Controller as PIDController
    
    states = []
    actions = []
    
    for data_file in tqdm(data_files[:num_files], desc="Collecting"):
        controller = PIDController()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
        
        original_update = controller.update
        
        def capture(target_lataccel, current_lataccel, state, future_plan):
            old_error_integral = controller.error_integral
            old_prev_error = controller.prev_error
            
            error = target_lataccel - current_lataccel
            new_error_integral = old_error_integral + error
            error_diff = error - old_prev_error
            
            # FB state only (same as exp040)
            state_vec = [error, new_error_integral, error_diff]
            action = original_update(target_lataccel, current_lataccel, state, future_plan)
            
            states.append(state_vec)
            actions.append(action)
            return action
        
        controller.update = capture
        sim.rollout()
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"✓ Collected {len(states):,} samples")
    
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump({'states': states, 'actions': actions}, f)
        print(f"✓ Saved to cache: {cache_file.name}")
    
    return states, actions


def train_bc(actor, states, actions):
    """Phase 1: Train FB module only (EXACT same as exp040)"""
    print(f"\n{'='*60}")
    print("Phase 1: Behavioral Cloning (Train FB Only)")
    print(f"{'='*60}")
    
    X = torch.FloatTensor(states).to(device)
    y = torch.FloatTensor(actions).to(device)
    
    # Train with Adam (same as exp040: lr=0.01, 500 epochs)
    print("\n  Training with Adam (lr=0.01, 500 epochs)...")
    optimizer = optim.Adam(actor.fb.linear.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        predictions = actor.fb.linear(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            weights = actor.fb.linear.weight.data.cpu().numpy()[0]
            print(f"  Epoch {epoch+1:3d}/500 | Loss: {loss.item():.8f} | P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")
    
    final_loss = loss.item()
    print(f"\n  Final BC Loss: {final_loss:.10f}")
    
    weights = actor.fb.linear.weight.data.cpu().numpy()[0]
    print(f"\n✓ BC Complete!")
    print(f"  Learned: P={weights[0]:.6f}, I={weights[1]:.6f}, D={weights[2]:.6f}")
    print(f"  PID:     P={PID_P:.6f}, I={PID_I:.6f}, D={PID_D:.6f}")
    print(f"  Error:   {abs(weights[0]-PID_P)+abs(weights[1]-PID_I)+abs(weights[2]-PID_D):.6f}")
    
    # Verify correct signs
    if np.sign(weights[2]) != np.sign(PID_D):
        print(f"  ⚠️  WARNING: D coefficient has WRONG SIGN!")
    
    return final_loss


# ============================================================
# EVALUATION
# ============================================================

def rollout_episode(data_file, controller):
    """Single episode rollout"""
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
    sim.rollout()
    return sim.compute_cost()


def evaluate_policy(actor, data_files, num_episodes=16, use_multiprocessing=False):
    """Evaluate policy on data files (includes FB + FF)"""
    if use_multiprocessing:
        # Use official multiprocessing evaluation (like exp039)
        temp_controller_path = Path(__file__).parent.parent.parent / 'controllers' / 'temp_eval_controller.py'
        temp_model_path = Path(__file__).parent / 'temp_eval_model.pth'
        torch.save(actor.state_dict(), temp_model_path)
        
        # Write temporary controller that loads full model (FB + FF)
        exp_dir = Path(__file__).parent.absolute()
        controller_code = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from controllers import BaseController
import importlib.util

# Load the experiment's train module
train_path = Path("{exp_dir}") / "train.py"
spec = importlib.util.spec_from_file_location("exp041_train", train_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

class Controller(BaseController):
    def __init__(self):
        device = torch.device('cpu')
        self.actor = train_module.SplitActor().to(device)
        model_path = Path("{exp_dir}") / "temp_eval_model.pth"
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()
        self.controller = train_module.SplitController(self.actor, deterministic=True)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.controller.update(target_lataccel, current_lataccel, state, future_plan)
    
    def reset(self):
        self.controller.reset()
'''
        temp_controller_path.write_text(controller_code)
        
        import shutil
        cache_dir = temp_controller_path.parent / '__pycache__'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        try:
            run_rollout_partial = partial(run_rollout, controller_type='temp_eval_controller',
                                         model_path=str(MODEL_PATH), debug=False)
            files_to_eval = data_files[:num_episodes]
            results = process_map(run_rollout_partial, files_to_eval, max_workers=16, chunksize=10, disable=True)
            costs = [result[0]['total_cost'] for result in results]
            return float(np.mean(costs))
        finally:
            if temp_controller_path.exists():
                temp_controller_path.unlink()
            if temp_model_path.exists():
                temp_model_path.unlink()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
    
    else:
        # Sequential evaluation (includes FF)
        controller = SplitController(actor, deterministic=True)
        costs = []
        for i in range(min(num_episodes, len(data_files))):
            controller.reset()
            cost_dict = rollout_episode(data_files[i], controller)
            costs.append(cost_dict['total_cost'])
        return float(np.mean(costs))


# ============================================================
# PPO TRAINING
# ============================================================

class PPO:
    def __init__(self, actor, critic, pi_lr, vf_lr):
        self.actor = actor
        self.critic = critic
        # Train ALL actor parameters (FB + FF + log_std)
        self.pi_optimizer = optim.Adam(actor.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(critic.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.lamda = gae_lambda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
    
    def compute_advantages(self, rewards, state_values, is_terminals):
        advantages = []
        gae = 0
        T = len(rewards)
        # Bootstrap from last state value, not zero
        state_values_pad = np.concatenate([state_values, [state_values[-1]]])
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + state_values
        return advantages, returns
    
    def update(self, fb_states, curvatures, actions, rewards, dones):
        """PPO update - trains both FB and FF"""
        fb_states_t = torch.FloatTensor(fb_states).to(device)
        curvatures_t = torch.FloatTensor(curvatures).to(device)
        actions_t = torch.FloatTensor(actions).unsqueeze(-1).to(device)
        
        # Compute values
        with torch.no_grad():
            state_values = self.critic(fb_states_t).cpu().numpy().flatten()
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, state_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        
        # Old log probs
        with torch.no_grad():
            action_mean, action_std, _, _ = self.actor(fb_states_t, curvatures_t)
            dist = torch.distributions.Normal(action_mean, action_std)
            old_logprobs = dist.log_prob(actions_t).sum(-1)
        
        # PPO epochs
        for _ in range(self.K_epochs):
            indices = torch.randperm(len(fb_states_t))
            
            for start_idx in range(0, len(fb_states_t), self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Forward
                action_mean, action_std, _, _ = self.actor(fb_states_t[idx], curvatures_t[idx])
                values = self.critic(fb_states_t[idx])
                
                # Policy loss
                dist = torch.distributions.Normal(action_mean, action_std)
                logprobs = dist.log_prob(actions_t[idx]).sum(-1)
                ratios = torch.exp(logprobs - old_logprobs[idx])
                
                surr1 = ratios * advantages_t[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = F.mse_loss(values.squeeze(), returns_t[idx])
                
                # Entropy
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                
                # Update
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()


def collect_episodes(actor, critic, data_files, num_episodes):
    """Collect episodes for PPO training"""
    all_fb_states = []
    all_curvatures = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for data_file in random.sample(data_files, num_episodes):
        controller = SplitController(actor, deterministic=False)
        
        fb_states = []
        curvatures_list = []
        actions = []
        
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
        
        original_update = controller.update
        
        def capture_rollout(target_lataccel, current_lataccel, state, future_plan):
            error = target_lataccel - current_lataccel
            controller.error_integral += error
            error_diff = error - controller.prev_error
            controller.prev_error = error
            
            fb_state = np.array([error, controller.error_integral, error_diff], dtype=np.float32)
            
            # Compute CURVATURES (pad to 50 elements, SCALED)
            future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
            future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
            future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
            
            # Curvature = (lataccel - roll) / v² for EACH timestep
            # Scale by 1000 to bring ~0.0003 → ~0.3 (reasonable for neural network)
            v_squared = np.maximum(future_v_ego ** 2, 1.0)
            curvatures = (((future_lataccel - future_roll) / v_squared) * 1000.0).astype(np.float32)
            
            # Sample action
            action, fb, ff = actor.act(fb_state, curvatures, deterministic=False)
            
            fb_states.append(fb_state)
            curvatures_list.append(curvatures)
            actions.append(action)
            
            return action
        
        controller.update = capture_rollout
        sim.rollout()
        cost_dict = sim.compute_cost()
        
        # SPARSE reward: all cost at episode end (like exp040 - proven to work!)
        rewards = np.zeros(len(fb_states))
        rewards[-1] = -cost_dict['total_cost']
        
        # Terminal flag at end
        dones = np.zeros(len(fb_states))
        dones[-1] = 1.0
        
        all_fb_states.extend(fb_states)
        all_curvatures.extend(curvatures_list)
        all_actions.extend(actions)
        all_rewards.extend(rewards)  # Dense rewards collected per step
        all_dones.extend(dones)
    
    return (np.array(all_fb_states), np.array(all_curvatures), 
            np.array(all_actions), np.array(all_rewards), np.array(all_dones))


# ============================================================
# TRAINING CONTEXT & EPOCH LOOP (beautiful_lander pattern)
# ============================================================

class TrainingContext:
    """Holds training state (like beautiful_lander.py)"""
    def __init__(self, actor, critic, train_files, eval_files, bc_cost, pid_baseline):
        self.actor = actor
        self.critic = critic
        self.train_files = train_files
        self.eval_files = eval_files
        self.ppo = PPO(actor, critic, pi_lr, vf_lr)
        
        self.bc_cost = bc_cost
        self.pid_baseline = pid_baseline
        self.best_cost = float('inf')
        self.last_eval = float('inf')  # Like beautiful_lander
        self.patience_counter = 0
        
        self.pbar = trange(max_epochs_ppo, desc="PPO", unit='epoch')
        self.rollout_times = []
        self.update_times = []
    
    def cleanup(self):
        self.pbar.close()


def train_one_epoch(epoch, ctx):
    """Train one PPO epoch (like beautiful_lander.py)"""
    import time
    
    # Rollout phase
    t0 = time.perf_counter()
    fb_states, curvatures, actions, rewards, dones = collect_episodes(
        ctx.actor, ctx.critic, ctx.train_files, num_episodes=num_envs
    )
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    # Update phase
    t0 = time.perf_counter()
    ctx.ppo.update(fb_states, curvatures, actions, rewards, dones)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    ctx.pbar.update(1)
    
    # Evaluate periodically (store in ctx like beautiful_lander does with ctx.last_eval)
    if epoch % eval_interval == 0:
        ctx.last_eval = evaluate_policy(ctx.actor, ctx.eval_files, num_episodes=100, use_multiprocessing=True)
        
        if ctx.last_eval < ctx.best_cost:
            ctx.best_cost = ctx.last_eval
            ctx.patience_counter = 0
            torch.save(ctx.actor.state_dict(), Path(__file__).parent / 'best_model.pth')
            
            if ctx.best_cost < ctx.pid_baseline:
                ctx.pbar.write(f"\n{'='*60}")
                ctx.pbar.write(f"🎯 BEAT PID! cost={ctx.best_cost:.2f} < {ctx.pid_baseline:.2f}")
                ctx.pbar.write(f"{'='*60}")
        else:
            ctx.patience_counter += 1
        
        # Early stopping
        if ctx.patience_counter > 30:
            ctx.pbar.write(f"\nEarly stopping at epoch {epoch} (no improvement for 30 evals)")
            return True  # Signal to stop
    
    # Log periodically (NOT nested in eval check - like beautiful_lander)
    if epoch % log_interval == 0:
        # Sample FB/FF contributions
        with torch.no_grad():
            sample_fb = fb_states[:100]
            sample_curv = curvatures[:100]
            _, _, fb_contrib, ff_contrib = ctx.actor(
                torch.FloatTensor(sample_fb).to(device),
                torch.FloatTensor(sample_curv).to(device)
            )
            fb_mean = fb_contrib.mean().item()
            ff_mean = ff_contrib.mean().item()
            fb_std = fb_contrib.std().item()
            ff_std = ff_contrib.std().item()
        
        fb_weights = ctx.actor.fb.linear.weight.data.cpu().numpy()[0]
        
        # Timing info
        rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
        update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
        
        ctx.pbar.write(f"Epoch {epoch:3d}  eval={ctx.last_eval:6.1f}  best={ctx.best_cost:6.1f}  "
                      f"P={fb_weights[0]:+.3f} I={fb_weights[1]:+.3f} D={fb_weights[2]:+.3f}  "
                      f"FB={fb_mean:+.3f}±{fb_std:.3f}  FF={ff_mean:+.3f}±{ff_std:.3f}  "
                      f"⏱ {rollout_ms:.0f}+{update_ms:.0f}ms")
    
    return False  # Continue training


def train_ppo(ctx):
    """PPO training loop"""
    print(f"\n{'='*60}")
    print("Phase 2: PPO Fine-Tuning (Train FB + FF)")
    print(f"{'='*60}")
    print(f"Starting from BC cost: {ctx.bc_cost:.2f}\n")
    
    for epoch in range(max_epochs_ppo):
        should_stop = train_one_epoch(epoch, ctx)
        if should_stop:
            break
    
    ctx.cleanup()


def final_evaluation(actor, bc_cost, pid_baseline, eval_files):
    """Final evaluation and reporting"""
    # Load best model
    best_model_path = Path(__file__).parent / 'best_model.pth'
    if best_model_path.exists():
        actor.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\n✓ Loaded best model from {best_model_path}")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    final_cost = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=False)
    
    fb_weights = actor.fb.linear.weight.data.cpu().numpy()[0]
    
    print(f"\n✓ BC cost (official): {bc_cost:.2f}")
    print(f"✓ Final PPO cost: {final_cost:.2f}")
    print(f"✓ PID baseline (official): {pid_baseline:.2f}")
    print(f"✓ Improvement over PID: {pid_baseline - final_cost:.2f} points")
    print(f"✓ Improvement over exp040 (76.6): {76.6 - final_cost:.2f} points")
    print(f"\n✓ Final FB weights: P={fb_weights[0]:.6f}, I={fb_weights[1]:.6f}, D={fb_weights[2]:.6f}")
    print(f"✓ PID weights:      P={PID_P:.6f}, I={PID_I:.6f}, D={PID_D:.6f}")
    print(f"{'='*60}")


# ============================================================
# MAIN TRAINING
# ============================================================

def train():
    print(f"\n{'='*60}")
    print("Experiment 041: Split FB/FF Architecture")
    print(f"{'='*60}")
    print(f"FF: PRIMARY - Proactive planner (1D Conv + MLP on future curvatures)")
    print(f"FB: SECONDARY - Reactive corrections (20% of PID gains)")
    print(f"Architecture: action = FF(future) + FB(error)")
    print(f"Using {device}")
    
    # Phase 1: BC on FB
    fb_states, actions = collect_pid_demonstrations(train_files, num_files=1000)
    
    actor = SplitActor().to(device)
    critic = Critic().to(device)
    
    # Initialize FF network to FULL STRENGTH (FF is PRIMARY - proactive control)
    nn.init.orthogonal_(actor.ff.fc2.weight, gain=1.0)  # Full strength, not tiny
    nn.init.zeros_(actor.ff.fc2.bias)
    print(f"✓ FF network initialized with gain=1.0 (FF is PRIMARY - proactive planner)")
    
    bc_mse = train_bc(actor, fb_states, actions)
    
    # SURGICAL FIX: Scale down FB to be small corrections (20% of PID)
    # FF = proactive (looks ahead), FB = reactive corrections (small adjustments)
    actor.fb.linear.weight.data *= 0.2
    print(f"✓ FB scaled to 20% of PID (FB is SECONDARY - reactive corrections)")
    
    # Evaluate BC (FB only, FF=0)
    print(f"\n⏳ Evaluating BC (FB only, FF=0)...")
    bc_cost_official = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=True)
    print(f"✓ BC cost (official): {bc_cost_official:.2f}")
    
    # PID baseline
    print(f"\n⏳ Computing PID baseline...")
    from controllers.pid import Controller as PIDController
    pid_costs = []
    for data_file in eval_files:
        controller = PIDController()
        cost_dict = rollout_episode(data_file, controller)
        pid_costs.append(cost_dict['total_cost'])
    pid_baseline = np.mean(pid_costs)
    print(f"✓ PID Baseline (official): {pid_baseline:.2f}")
    
    print(f"\n✓ BC vs PID error: {abs(bc_cost_official - pid_baseline):.2f}")
    
    # Phase 2: PPO with train_one_epoch pattern
    ctx = TrainingContext(actor, critic, train_files, eval_files, bc_cost_official, pid_baseline)
    train_ppo(ctx)
    
    # Final evaluation
    final_evaluation(actor, bc_cost_official, pid_baseline, eval_files)


if __name__ == '__main__':
    train()
