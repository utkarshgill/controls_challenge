#!/usr/bin/env python3
"""
Unified BC → PPO Training Pipeline

Step 1: Train BC to clone PID controller
Step 2: Train PPO initialized from BC weights
Step 3: Evaluate both stages

This is the SINGLE SOURCE OF TRUTH for our training process.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import glob
import os
import gymnasium as gym
from multiprocessing import Pool, cpu_count

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, STEER_RANGE
from controllers import pid

# ============================================================
# CONFIGURATION
# ============================================================

# Architecture (shared by BC and PPO)
STATE_DIM, ACTION_DIM = 56, 1
HIDDEN_DIM = 128
TRUNK_LAYERS, HEAD_LAYERS = 1, 3

# State normalization
OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] +  # [error, error_diff, error_integral, lataccel, v_ego, curv]
    [1000.0] * 50,
    dtype=np.float32
)

# BC hyperparameters
BC_N_FILES = 1000          # Files for expert data collection
BC_N_EPOCHS = 50           # BC training epochs
BC_BATCH_SIZE = 512
BC_LR = 1e-3

# PPO hyperparameters
PPO_N_EPOCHS = 100         # PPO training epochs (increase to 500 later)
PPO_NUM_ENVS = 8           # Parallel environments
PPO_STEPS_PER_EPOCH = 10000
PPO_BATCH_SIZE = 2048
PPO_K_EPOCHS = 4
PPO_LR = 1e-5              # TODO: Increase to 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_EPS_CLIP = 0.1
PPO_ENTROPY_COEF = 0.0
PPO_LOG_STD_INIT = np.log(0.05)  # TODO: Increase to 0.1

# General
DEVICE = torch.device('cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("\n" + "="*60)
print("UNIFIED BC → PPO TRAINING PIPELINE")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")

# ============================================================
# SHARED FUNCTIONS
# ============================================================

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """Build 56D state vector (used by both BC and PPO)"""
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class ActorCritic(nn.Module):
    """Shared network architecture for BC and PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
        super(ActorCritic, self).__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * PPO_LOG_STD_INIT)
        
        # Critic head (only used for PPO)
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value
    
    @torch.no_grad()
    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(DEVICE)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action_mean, action_std, _ = self(state_tensor)
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action.squeeze().cpu().numpy()

# ============================================================
# STEP 1: BEHAVIORAL CLONING (BC)
# ============================================================

def collect_expert_data_single_file(args):
    """Collect PID expert data from a single file"""
    data_file, model = args
    
    # Create PID controller
    pid_controller = pid.Controller()
    
    # Run simulation
    sim = TinyPhysicsSimulator(model, data_file, controller=pid_controller, debug=False)
    
    states_actions = []
    prev_error = 0.0
    error_integral = 0.0
    
    for step_idx in range(CONTROL_START_IDX, len(sim.data)):
        state_obj, target_lataccel, future_plan = sim.get_state_target_futureplan(step_idx)
        current_lataccel = sim.current_lataccel
        
        # Build state
        state_vec = build_state(target_lataccel, current_lataccel, state_obj, future_plan, prev_error, error_integral)
        
        # Get PID action
        pid_action = pid_controller.update(target_lataccel, current_lataccel, state_obj, future_plan)
        
        states_actions.append((state_vec, pid_action))
        
        # Update state
        error = target_lataccel - current_lataccel
        error_integral += error
        prev_error = error
        
        # Step simulator
        sim.current_steer = pid_action
        sim.sim_step(step_idx)
        sim.step_idx = step_idx + 1
    
    return states_actions

def train_bc(network, train_files, val_files, n_expert_files):
    """Train BC to clone PID"""
    print("\n" + "="*60)
    print("STEP 1: BEHAVIORAL CLONING")
    print("="*60)
    
    # Collect expert data
    print(f"\nCollecting expert data from {n_expert_files} files...")
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    
    expert_files = np.random.choice(train_files, size=min(n_expert_files, len(train_files)), replace=False)
    
    # Sequential collection (avoid multiprocessing issues)
    all_data = []
    for data_file in tqdm(expert_files, desc="Collecting"):
        data = collect_expert_data_single_file((data_file, model))
        all_data.extend(data)
    
    states = np.array([s for s, a in all_data], dtype=np.float32)
    actions = np.array([a for s, a in all_data], dtype=np.float32).reshape(-1, 1)
    
    print(f"✅ Collected {len(states)} state-action pairs")
    
    # Normalize states
    states_normalized = states * OBS_SCALE
    
    # Create dataset
    dataset = TensorDataset(
        torch.from_numpy(states_normalized).float(),
        torch.from_numpy(actions).float()
    )
    dataloader = DataLoader(dataset, batch_size=BC_BATCH_SIZE, shuffle=True)
    
    # Train BC
    print(f"\nTraining BC for {BC_N_EPOCHS} epochs...")
    optimizer = optim.Adam(network.parameters(), lr=BC_LR)
    
    network.train()
    for epoch in trange(BC_N_EPOCHS, desc="BC Training"):
        epoch_loss = 0.0
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(DEVICE)
            actions_batch = actions_batch.to(DEVICE)
            
            # Forward
            action_pred, _, _ = network(states_batch)
            loss = F.mse_loss(action_pred, actions_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.6f}")
    
    print("✅ BC training complete")
    
    # Evaluate BC
    print("\nEvaluating BC...")
    bc_cost = evaluate_controller(network, val_files[:100], model)
    print(f"✅ BC validation cost: {bc_cost:.2f}")
    
    # Save BC checkpoint
    torch.save({
        'model_state_dict': network.state_dict(),
        'bc_val_cost': bc_cost,
    }, 'bc_checkpoint.pth')
    print("✅ Saved: bc_checkpoint.pth")
    
    return bc_cost

# ============================================================
# STEP 2: PPO TRAINING
# ============================================================

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class TinyPhysicsGymEnv(gym.Env):
    """Gymnasium wrapper for TinyPhysicsSimulator"""
    def __init__(self, model_path, data_files):
        super().__init__()
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data_files = data_files
        self.sim = None
        
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = 0.0
        self.episode_cost = 0.0
        
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        data_file = np.random.choice(self.data_files)
        self.sim = TinyPhysicsSimulator(self.model, data_file, controller=None, debug=False)
        
        # Fast-forward through warmup
        for _ in range(CONTROL_START_IDX):
            self.sim.current_steer = 0.0
            self.sim.sim_step(self.sim.step_idx)
            self.sim.step_idx += 1
        
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_lataccel = self.sim.current_lataccel
        self.episode_cost = 0.0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        current_lataccel = self.sim.current_lataccel
        
        action_value = float(np.clip(action[0], -2.0, 2.0))
        self.sim.current_steer = action_value
        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1
        
        # Compute reward
        lat_cost = (current_lataccel - target_lataccel) ** 2
        jerk_cost = (current_lataccel - self.prev_lataccel) ** 2
        reward = -(lat_cost + jerk_cost) / 100.0
        
        self.episode_cost += (lat_cost + jerk_cost)
        
        # Update state
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.prev_error = error
        self.prev_lataccel = current_lataccel
        
        done = self.sim.step_idx >= len(self.sim.data)
        truncated = False
        
        obs = self._get_observation() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        
        info = {'episode_cost': self.episode_cost} if done else {}
        
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        current_lataccel = self.sim.current_lataccel
        return build_state(target_lataccel, current_lataccel, state, future_plan,
                          self.prev_error, self.error_integral)

def make_env(model_path, data_files):
    def _init():
        return TinyPhysicsGymEnv(model_path, data_files)
    return _init

class PPO:
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=PPO_LR)
        self.states, self.actions = [], []

    def __call__(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(DEVICE)
            action_mean, action_std, _ = self.actor_critic(state_tensor)
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
            action = torch.tanh(raw_action) * STEER_RANGE[1]
            
            self.states.append(state_tensor)
            self.actions.append(raw_action)
            
            return action.cpu().numpy()

    def compute_advantages(self, rewards, state_values, is_terminals):
        T, N = rewards.shape
        advantages, gae = torch.zeros_like(rewards), torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        for t in reversed(range(T)):
            delta = rewards[t] + PPO_GAMMA * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, 
                                torch.clamp(ratios, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().sum(-1).mean()
        return actor_loss + critic_loss - PPO_ENTROPY_COEF * entropy
    
    def update(self, rewards, dones):
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(DEVICE)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(DEVICE)
            old_states, old_actions = torch.cat(self.states), torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1).view(-1, rewards.size(1))
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(PPO_K_EPOCHS):
            for batch in DataLoader(dataset, batch_size=PPO_BATCH_SIZE, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.1)
                self.optimizer.step()
        
        self.states, self.actions = [], []

def rollout(env, policy, num_steps):
    states, _ = env.reset()
    traj_rewards, traj_dones = [], []
    episode_costs = []
    step_count = 0
    
    while step_count < num_steps:
        actions = policy(states)
        states, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        traj_rewards.append(rewards)
        traj_dones.append(dones)
        step_count += env.num_envs
        
        # Extract episode costs
        if 'final_info' in infos:
            for info_dict in infos['final_info']:
                if info_dict is not None and 'episode_cost' in info_dict:
                    episode_costs.append(info_dict['episode_cost'])
    
    return traj_rewards, traj_dones, episode_costs

def train_ppo(network, train_files, val_files):
    """Train PPO from BC initialization"""
    print("\n" + "="*60)
    print("STEP 2: PPO TRAINING")
    print("="*60)
    
    # Create parallel environments
    model_path = "./models/tinyphysics.onnx"
    env = gym.vector.AsyncVectorEnv([
        make_env(model_path, train_files) for _ in range(PPO_NUM_ENVS)
    ])
    
    print(f"Parallel envs: {PPO_NUM_ENVS}")
    print(f"Steps per epoch: {PPO_STEPS_PER_EPOCH}")
    print(f"Total epochs: {PPO_N_EPOCHS}")
    
    ppo = PPO(network)
    network.train()
    
    best_cost = float('inf')
    
    pbar = trange(PPO_N_EPOCHS, desc="PPO Training")
    for epoch in pbar:
        # Collect trajectories
        rewards, dones, episode_costs = rollout(env, ppo, num_steps=PPO_STEPS_PER_EPOCH)
        
        # PPO update
        ppo.update(rewards, dones)
        
        # Track costs
        if episode_costs:
            mean_cost = np.mean(episode_costs)
            if mean_cost < best_cost:
                best_cost = mean_cost
                torch.save(network.state_dict(), 'ppo_best.pth')
            
            if epoch % 5 == 0:
                pbar.write(f"Epoch {epoch:3d} | Cost: {mean_cost:6.2f} | Best: {best_cost:6.2f} | Episodes: {len(episode_costs)}")
            
            if mean_cost > 1000:
                pbar.write(f"\n⚠️  Cost exploded to {mean_cost:.2f}. Stopping early.")
                break
    
    env.close()
    pbar.close()
    
    print(f"\n✅ PPO training complete! Best cost: {best_cost:.2f}")
    print("✅ Saved: ppo_best.pth")
    
    # Evaluate PPO
    print("\nEvaluating PPO...")
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    ppo_cost = evaluate_controller(network, val_files[:100], model)
    print(f"✅ PPO validation cost: {ppo_cost:.2f}")
    
    return ppo_cost

# ============================================================
# EVALUATION
# ============================================================

class EvalController:
    """Wrapper for evaluation"""
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        action = self.network.act(obs, deterministic=True)
        
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.prev_error = error
        
        return float(action) if isinstance(action, np.ndarray) else action

def evaluate_controller(network, data_files, model):
    """Evaluate controller on validation files"""
    costs = []
    for data_file in tqdm(data_files, desc="Evaluating", leave=False):
        controller = EvalController(network)
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()['total_cost']
        costs.append(cost)
    
    mean_cost = np.mean(costs)
    return mean_cost

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # Load data
    print("\nLoading dataset...")
    all_files = sorted(glob.glob("./data/*.csv"))
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    print(f"✅ Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create network
    print("\nCreating ActorCritic network...")
    network = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS).to(DEVICE)
    print(f"✅ Network created: {sum(p.numel() for p in network.parameters())} parameters")
    
    # Step 1: BC
    bc_cost = train_bc(network, train_files, val_files, BC_N_FILES)
    
    # Step 2: PPO
    ppo_cost = train_ppo(network, train_files, val_files)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"BC cost:  {bc_cost:.2f}")
    print(f"PPO cost: {ppo_cost:.2f}")
    print(f"Target:   <45.00")
    print("="*60)

if __name__ == '__main__':
    main()

