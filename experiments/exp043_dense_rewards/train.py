"""
Experiment 043: Dense Rewards + BC Warm Start

Architecture: ONE network
Input: [error, integral, derivative, 50 future curvatures*1000] = 53D
Output: action (1D)

Recipe:
1. BC warm start on PID (like exp040)
2. Scale curvatures 1000x (0.0003 → 0.3)
3. Dense per-step rewards: -abs(error) not -total_cost
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import trange
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, run_rollout
from controllers import BaseController

# Device
device = torch.device('cpu')

# Paths
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
DATA_PATH = Path(__file__).parent.parent.parent / 'data'

# Get data files
train_files = sorted([f for f in DATA_PATH.iterdir() if f.suffix == '.csv'])[:1500]
eval_files = sorted([f for f in DATA_PATH.iterdir() if f.suffix == '.csv'])[-100:]

# Global model (for multiprocessing)
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# Hyperparameters (from beautiful_lander + exp040)
hidden_dim = 128
num_envs = 8
max_epochs_ppo = 200
log_interval = 10
eval_interval = 10

pi_lr = 1e-4  # Lower for stability after BC
vf_lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
K_epochs = 20
eps_clip = 0.2
batch_size = 512
vf_coef = 0.5
entropy_coef = 0.01


# ============================================================
# SIMPLE ARCHITECTURE
# ============================================================

class SimpleActor(nn.Module):
    """ONE simple MLP - no tricks
    
    Input: [error, integral, derivative, curv_0, ..., curv_49] (53D)
    Output: action (1D)
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(53, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state):
        """state: [batch, 53]"""
        action_mean = self.net(state)
        action_std = self.log_std.exp()
        return action_mean, action_std
    
    @torch.inference_mode()
    def act(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        mean, std = self(state)
        if deterministic:
            action = mean
        else:
            action = torch.distributions.Normal(mean, std).sample()
        return action.cpu().numpy()[0, 0]


class Critic(nn.Module):
    """Value function"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(53, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class SimpleController(BaseController):
    """Controller using simple MLP"""
    def __init__(self, actor, deterministic=True):
        self.actor = actor
        self.deterministic = deterministic
        self.error_integral = 0
        self.prev_error = 0
    
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Error state
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Future curvatures
        future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
        future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
        future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
        
        v_squared = np.maximum(future_v_ego ** 2, 1.0)
        curvatures = (((future_lataccel - future_roll) / v_squared) * 1000.0).astype(np.float32)  # Scale 1000x
        
        # Concat all inputs
        full_state = np.concatenate([[error, self.error_integral, error_diff], curvatures])
        
        action = self.actor.act(full_state, deterministic=self.deterministic)
        return action


# ============================================================
# BEHAVIORAL CLONING (BC Warm Start)
# ============================================================

def collect_pid_demonstrations(data_files, num_files=1000):
    """Collect PID demonstrations for BC - EXACT copy of exp040 logic + curvatures"""
    from controllers.pid import Controller as PIDController
    
    print(f"\n{'='*60}")
    print(f"Collecting PID Demonstrations")
    print(f"{'='*60}\n")
    
    all_states = []
    all_actions = []
    
    for data_file in data_files[:num_files]:
        pid_controller = PIDController()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=pid_controller)
        
        original_update = pid_controller.update
        
        def capture_pid(target_lataccel, current_lataccel, state, future_plan):
            # Save OLD state BEFORE PID updates it (like exp040)
            old_error_integral = pid_controller.error_integral
            old_prev_error = pid_controller.prev_error
            
            # Compute what PID will compute
            error = target_lataccel - current_lataccel
            
            # NEW state that PID uses internally
            new_error_integral = old_error_integral + error
            error_diff = error - old_prev_error
            
            # Call PID (updates its internal state)
            pid_action = original_update(target_lataccel, current_lataccel, state, future_plan)
            
            # Build our full state: [error, integral, diff, 50 curvatures]
            future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
            future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
            future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
            
            v_squared = np.maximum(future_v_ego ** 2, 1.0)
            curvatures = (((future_lataccel - future_roll) / v_squared) * 1000.0).astype(np.float32)
            
            full_state = np.concatenate([[error, new_error_integral, error_diff], curvatures])
            
            all_states.append(full_state)
            all_actions.append(pid_action)
            return pid_action
        
        pid_controller.update = capture_pid
        sim.rollout()
    
    print(f"✓ Collected {len(all_states)} samples from {num_files} episodes\n")
    return np.array(all_states), np.array(all_actions)


def train_bc(actor, states, actions, epochs=500, lr=0.01):
    """Train actor to clone PID using BC"""
    print(f"{'='*60}")
    print(f"Phase 1: Behavioral Cloning (BC Warm Start)")
    print(f"{'='*60}\n")
    
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.FloatTensor(actions).unsqueeze(-1).to(device)
    
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_actions, _ = actor(states_t)
        loss = ((pred_actions - actions_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.8f}")
    
    print(f"\n✓ BC Complete! Final Loss: {loss.item():.8f}\n")
    return loss.item()


# ============================================================
# EVALUATION
# ============================================================

def rollout_episode(data_file, controller):
    """Single episode rollout"""
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
    sim.rollout()
    return sim.compute_cost()


def evaluate_policy(actor, data_files, num_episodes=16, use_multiprocessing=False):
    """Evaluate policy"""
    if use_multiprocessing:
        temp_controller_path = Path(__file__).parent.parent.parent / 'controllers' / 'temp_eval_controller.py'
        temp_model_path = Path(__file__).parent / 'temp_eval_model.pth'
        torch.save(actor.state_dict(), temp_model_path)
        
        exp_dir = Path(__file__).parent.absolute()
        controller_code = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from controllers import BaseController
import importlib.util

train_path = Path("{exp_dir}") / "train.py"
spec = importlib.util.spec_from_file_location("exp043_train", train_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

class Controller(BaseController):
    def __init__(self):
        device = torch.device('cpu')
        self.actor = train_module.SimpleActor().to(device)
        model_path = Path("{exp_dir}") / "temp_eval_model.pth"
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()
        self.controller = train_module.SimpleController(self.actor, deterministic=True)
    
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
        controller = SimpleController(actor, deterministic=True)
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
        state_values_pad = np.concatenate([state_values, [state_values[-1]]])
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + state_values
        return advantages, returns
    
    def update(self, states, actions, rewards, dones):
        """PPO update"""
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            state_values = self.critic(states_t).cpu().numpy().flatten()
        
        advantages, returns = self.compute_advantages(rewards, state_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        
        with torch.no_grad():
            action_mean, action_std = self.actor(states_t)
            dist = torch.distributions.Normal(action_mean, action_std)
            old_logprobs = dist.log_prob(actions_t).sum(-1)
        
        for _ in range(self.K_epochs):
            indices = torch.randperm(len(states_t))
            
            for start_idx in range(0, len(states_t), self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                action_mean, action_std = self.actor(states_t[idx])
                values = self.critic(states_t[idx])
                
                dist = torch.distributions.Normal(action_mean, action_std)
                logprobs = dist.log_prob(actions_t[idx]).sum(-1)
                ratios = torch.exp(logprobs - old_logprobs[idx])
                
                surr1 = ratios * advantages_t[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = ((values.squeeze(-1) - returns_t[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()


def collect_episodes(actor, critic, data_files, num_episodes):
    """Collect episodes"""
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for data_file in random.sample(data_files, num_episodes):
        controller = SimpleController(actor, deterministic=False)
        
        states = []
        actions = []
        errors = []  # Track errors for dense rewards
        
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
        
        original_update = controller.update
        
        def capture_rollout(target_lataccel, current_lataccel, state, future_plan):
            error = target_lataccel - current_lataccel
            controller.error_integral += error
            error_diff = error - controller.prev_error
            controller.prev_error = error
            
            future_lataccel = np.array(list(future_plan.lataccel) + [0.0] * 50)[:50]
            future_roll = np.array(list(future_plan.roll_lataccel) + [0.0] * 50)[:50]
            future_v_ego = np.array(list(future_plan.v_ego) + [state.v_ego] * 50)[:50]
            
            v_squared = np.maximum(future_v_ego ** 2, 1.0)
            curvatures = (((future_lataccel - future_roll) / v_squared) * 1000.0).astype(np.float32)  # Scale 1000x
            
            full_state = np.concatenate([[error, controller.error_integral, error_diff], curvatures])
            action = controller.actor.act(full_state, deterministic=False)
            
            states.append(full_state)
            actions.append(action)
            errors.append(error)  # Save error for dense reward
            return action
        
        controller.update = capture_rollout
        sim.rollout()
        cost_dict = sim.compute_cost()
        
        # DENSE reward: -abs(error) at each step (immediate feedback)
        rewards = -np.abs(np.array(errors))
        
        dones = np.zeros(len(states))
        dones[-1] = 1.0
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_dones.extend(dones)
    
    return (np.array(all_states), np.array(all_actions), 
            np.array(all_rewards), np.array(all_dones))


# ============================================================
# TRAINING LOOP
# ============================================================

class TrainingContext:
    def __init__(self, actor, critic, train_files, eval_files):
        self.actor = actor
        self.critic = critic
        self.train_files = train_files
        self.eval_files = eval_files
        self.ppo = PPO(actor, critic, pi_lr, vf_lr)
        
        self.best_cost = float('inf')
        self.last_eval = float('inf')
        self.patience_counter = 0
        
        self.pbar = trange(max_epochs_ppo, desc="PPO", unit='epoch')
        self.rollout_times = []
        self.update_times = []
    
    def cleanup(self):
        self.pbar.close()


def train_one_epoch(epoch, ctx):
    import time
    
    # Rollout
    t0 = time.perf_counter()
    states, actions, rewards, dones = collect_episodes(
        ctx.actor, ctx.critic, ctx.train_files, num_episodes=num_envs
    )
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    # Update
    t0 = time.perf_counter()
    ctx.ppo.update(states, actions, rewards, dones)
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    ctx.pbar.update(1)
    
    # Evaluate
    if epoch % eval_interval == 0:
        ctx.last_eval = evaluate_policy(ctx.actor, ctx.eval_files, num_episodes=100, use_multiprocessing=True)
        
        if ctx.last_eval < ctx.best_cost:
            ctx.best_cost = ctx.last_eval
            ctx.patience_counter = 0
            torch.save(ctx.actor.state_dict(), Path(__file__).parent / 'best_model.pth')
        else:
            ctx.patience_counter += 1
        
        if ctx.patience_counter > 30:
            ctx.pbar.write(f"\nEarly stopping at epoch {epoch}")
            return True
    
    # Log
    if epoch % log_interval == 0:
        rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
        update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
        
        ctx.pbar.write(f"Epoch {epoch:3d}  eval={ctx.last_eval:6.1f}  best={ctx.best_cost:6.1f}  "
                      f"⏱ {rollout_ms:.0f}+{update_ms:.0f}ms")
    
    return False


def train():
    print(f"\n{'='*60}")
    print("Experiment 043: Dense Rewards + BC Warm Start")
    print(f"{'='*60}")
    print(f"Architecture: ONE network, 53D → 128 → 128 → 1D")
    print(f"Input: [error, integral, derivative, 50 curvatures*1000]")
    print(f"Rewards: Dense per-step (-abs(error))")
    print(f"Using {device}\n")
    
    actor = SimpleActor(hidden_dim).to(device)
    critic = Critic(hidden_dim).to(device)
    
    # Phase 1: BC on PID
    states, actions = collect_pid_demonstrations(train_files, num_files=1000)
    train_bc(actor, states, actions, epochs=500, lr=0.01)
    
    # Evaluate BC baseline
    print(f"⏳ Evaluating BC baseline...")
    bc_cost = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=True)
    print(f"✓ BC cost: {bc_cost:.2f}\n")
    
    # PID baseline
    print(f"⏳ Computing PID baseline...")
    from controllers.pid import Controller as PIDController
    pid_costs = []
    for data_file in eval_files:
        controller = PIDController()
        cost_dict = rollout_episode(data_file, controller)
        pid_costs.append(cost_dict['total_cost'])
    pid_baseline = np.mean(pid_costs)
    print(f"✓ PID Baseline: {pid_baseline:.2f}")
    
    print(f"\n{'='*60}")
    print("Phase 2: PPO with Dense Rewards")
    print(f"{'='*60}\n")
    
    # PPO training - create pbar AFTER printing header
    ctx = TrainingContext(actor, critic, train_files, eval_files)
    
    for epoch in range(max_epochs_ppo):
        if train_one_epoch(epoch, ctx):
            break
    
    ctx.cleanup()
    
    # Final eval
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    best_model_path = Path(__file__).parent / 'best_model.pth'
    if best_model_path.exists():
        actor.load_state_dict(torch.load(best_model_path, map_location=device))
    
    final_cost = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=True)
    
    print(f"✓ Final PPO cost: {final_cost:.2f}")
    print(f"✓ PID baseline: {pid_baseline:.2f}")
    print(f"✓ Improvement: {pid_baseline - final_cost:.2f} points")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    train()
