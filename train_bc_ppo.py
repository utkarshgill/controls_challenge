#!/usr/bin/env python3
"""
Two-stage training: Behavioral Cloning → PPO fine-tuning

Stage 1: BC learns from expert data (100 CSV files)
Stage 2: PPO fine-tunes BC policy with RL

Inspired by beautiful_lander.py's battle-tested structure.
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

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, LAT_ACCEL_COST_MULTIPLIER, CONTROL_START_IDX
from controllers import BaseController, pid

# Hyperparameters
state_dim, action_dim = 55, 1  # 55D: 5 current + 50 future curvatures
hidden_dim = 128
trunk_layers, head_layers = 1, 3

# BC hyperparameters
bc_lr = 1e-3
bc_epochs = 50
bc_batch_size = 256

# PPO hyperparameters
ppo_lr = 3e-4  # Lower LR for fine-tuning
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
batch_size, K_epochs = 2048, 4  # Fewer K_epochs to prevent overfitting
entropy_coef = 0.001
initial_log_std = -2.3  # σ ≈ 0.1
ppo_epochs = 300
episodes_per_epoch = 8
STEER_RANGE = (-2.0, 2.0)

# State normalization (like beautiful_lander's OBS_SCALE)
OBS_SCALE = np.array(
    [10.0, 2.0, 0.03, 20.0, 1000.0] +  # 5 current features
    [1000.0] * 50,                      # 50 future curvatures
    dtype=np.float32
)

device = torch.device('cpu')

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers, initial_log_std=0.0):
        super().__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * initial_log_std)
        
        # Critic head
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
        # Normalize state (like beautiful_lander)
        state_normalized = state * OBS_SCALE
        state_tensor = torch.as_tensor(state_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action.squeeze(0).cpu().numpy(), state_tensor, raw_action

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef
        self.states, self.actions = [], []

    def select_action(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, 
                                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().sum(-1).mean()
        return actor_loss + critic_loss - self.entropy_coef * entropy
    
    def update(self, rewards, dones):
        with torch.no_grad():
            rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.array(dones), dtype=torch.float32).to(device)
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1)
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states, self.actions = [], []

def build_state(target_lataccel, current_lataccel, state, future_plan):
    """
    Full 55D state: 5 current + 50 future curvatures
    """
    eps = 1e-6
    error = target_lataccel - current_lataccel
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    # Compute ALL 50 future curvatures
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    # Pad if needed
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, current_lataccel, state.v_ego, state.a_ego, curv_now] + future_curvs
    
    return np.array(state_vec, dtype=np.float32)

def collect_expert_data(model, data_files):
    """
    Collect (state, action) pairs from PID expert
    Run PID on each file and collect its actions as "expert demonstrations"
    """
    print("Collecting expert demonstrations from PID...")
    states_list = []
    actions_list = []
    
    for data_file in tqdm(data_files, desc="Running PID on data"):
        # Create a custom controller that logs state-action pairs
        pid_controller = pid.Controller()
        
        class LoggingController(BaseController):
            def __init__(self, pid_ctrl, states_list, actions_list):
                self.pid_ctrl = pid_ctrl
                self.states_list = states_list
                self.actions_list = actions_list
            
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                # Get PID action
                action = self.pid_ctrl.update(target_lataccel, current_lataccel, state, future_plan)
                
                # Build and log state
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
                self.states_list.append(state_vec)
                self.actions_list.append(action)
                
                return action
        
        logging_controller = LoggingController(pid_controller, states_list, actions_list)
        sim = TinyPhysicsSimulator(model, data_file, controller=logging_controller, debug=False)
        
        # Run full episode
        sim.rollout()
    
    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32).reshape(-1, 1)
    
    # Sanity check
    assert not np.any(np.isnan(states)), "States contain NaN!"
    assert not np.any(np.isnan(actions)), "Actions contain NaN!"
    assert not np.any(np.isinf(states)), "States contain inf!"
    assert not np.any(np.isinf(actions)), "Actions contain inf!"
    
    print(f"Collected {len(states)} PID expert transitions")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    return states, actions

def train_bc(actor_critic, states, actions):
    """
    Stage 1: Behavioral Cloning (supervised learning)
    Train actor to predict expert actions given states
    """
    print("\n" + "="*60)
    print("STAGE 1: Behavioral Cloning")
    print("="*60)
    
    # Normalize states
    states_normalized = states * OBS_SCALE
    
    # Keep actions as-is (don't transform to raw space for BC)
    # BC trains with direct MSE in action space
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states_normalized).to(device)
    actions_tensor = torch.FloatTensor(actions).to(device)
    
    # Create dataset
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=bc_batch_size, shuffle=True)
    
    # Optimizer for BC (only train actor, not critic)
    actor_params = list(actor_critic.trunk.parameters()) + \
                   list(actor_critic.actor_layers.parameters()) + \
                   list(actor_critic.actor_mean.parameters())
    optimizer = optim.Adam(actor_params, lr=bc_lr)
    
    pbar = trange(bc_epochs, desc="BC Training")
    for epoch in range(bc_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_states, batch_actions in dataloader:
            # Forward pass: predict action directly (apply tanh to bound output)
            trunk_feat = actor_critic.trunk(batch_states)
            actor_feat = actor_critic.actor_layers(trunk_feat)
            predicted_raw = actor_critic.actor_mean(actor_feat)
            predicted_action = torch.tanh(predicted_raw) * STEER_RANGE[1]  # Bound to [-2, 2]
            
            # MSE loss in action space
            loss = F.mse_loss(predicted_action, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        pbar.update(1)
        
        if epoch % 10 == 0:
            pbar.write(f"Epoch {epoch:3d} | BC Loss: {avg_loss:.6f}")
    
    pbar.close()
    print(f"BC training complete. Final loss: {avg_loss:.6f}")

def evaluate_policy(actor_critic, model, data_files, n_eval=20):
    """Evaluate policy on random files"""
    costs = []
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        class EvalController(BaseController):
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
                action, _, _ = actor_critic.act(state_vec, deterministic=True)
                return float(action[0]) if len(action.shape) > 0 else float(action)
        
        sim = TinyPhysicsSimulator(model, data_file, controller=EvalController(), debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)

class BCController(BaseController):
    """Controller using BC/PPO policy"""
    def __init__(self, ppo):
        self.ppo = ppo
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
        action = self.ppo.select_action(state_vec)
        return float(action[0]) if len(action.shape) > 0 else float(action)

def rollout_episode(model, data_file, ppo):
    """Run one episode with PPO policy"""
    sim = TinyPhysicsSimulator(model, data_file, controller=BCController(ppo), debug=False)
    
    episode_rewards = []
    episode_dones = []
    
    while sim.step_idx < len(sim.data):
        sim.step()
        
        if sim.step_idx >= CONTROL_START_IDX:
            error = sim.target_lataccel_history[-1] - sim.current_lataccel_history[-1]
            jerk = 0.0
            if len(sim.current_lataccel_history) >= 2:
                jerk = (sim.current_lataccel_history[-1] - sim.current_lataccel_history[-2]) / 0.1
            
            lat_cost = (error ** 2) * LAT_ACCEL_COST_MULTIPLIER
            jerk_cost = (jerk ** 2)
            reward = -(lat_cost + jerk_cost)
        else:
            reward = 0.0
        
        episode_rewards.append(reward)
        episode_dones.append(0.0)
    
    episode_dones[-1] = 1.0
    cost = sim.compute_cost()['total_cost']
    
    return episode_rewards, episode_dones, cost

def train_ppo(actor_critic, model, train_files):
    """
    Stage 2: PPO fine-tuning on BC policy
    """
    print("\n" + "="*60)
    print("STAGE 2: PPO Fine-tuning")
    print("="*60)
    
    ppo = PPO(actor_critic, ppo_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    
    print(f"Files: {len(train_files)}, Epochs: {ppo_epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Initial σ: {np.exp(initial_log_std):.3f}, Target: <45")
    print("="*60)
    
    all_costs = []
    best_cost = float('inf')
    pbar = trange(ppo_epochs, desc="PPO Training", unit='epoch')
    
    for epoch in range(ppo_epochs):
        # Collect episodes
        all_rewards = []
        all_dones = []
        epoch_costs = []
        
        for _ in range(episodes_per_epoch):
            data_file = np.random.choice(train_files)
            ep_rewards, ep_dones, cost = rollout_episode(model, data_file, ppo)
            all_rewards.extend(ep_rewards)
            all_dones.extend(ep_dones)
            epoch_costs.append(cost)
        
        # PPO update
        ppo.update(all_rewards, all_dones)
        
        # Track
        all_costs.extend(epoch_costs)
        mean_cost = np.mean(epoch_costs)
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save(actor_critic.state_dict(), "bc_ppo_best.pth")
        
        pbar.update(1)
        
        if epoch % 5 == 0:
            recent_cost = np.mean(all_costs[-50:]) if len(all_costs) >= 50 else mean_cost
            std_val = actor_critic.log_std.exp().item()
            eval_cost = evaluate_policy(actor_critic, model, train_files, n_eval=10) if epoch % 10 == 0 else mean_cost
            pbar.write(f"Epoch {epoch:3d} | Train: {mean_cost:.2f} | Recent50: {recent_cost:.2f} | "
                      f"Eval: {eval_cost:.2f} | σ: {std_val:.3f}")
    
    pbar.close()
    print("="*60)
    print(f"Final (last 50): {np.mean(all_costs[-50:]):.2f}")
    print(f"Best: {best_cost:.2f}")
    
    torch.save(actor_critic.state_dict(), "bc_ppo_final.pth")
    print("Models saved: bc_ppo_best.pth, bc_ppo_final.pth")
    
    return actor_critic

def main():
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    train_files = sorted(glob.glob("./data/*.csv"))[:100]
    
    print("Two-Stage Training: BC → PPO")
    print(f"Data files: {len(train_files)}")
    print(f"State dim: {state_dim}D (5 current + 50 future curvatures)")
    print()
    
    # Initialize network
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers, 
                                initial_log_std=initial_log_std).to(device)
    
    # Stage 1: Behavioral Cloning
    states, actions = collect_expert_data(model, train_files)
    train_bc(actor_critic, states, actions)
    
    # Evaluate BC with debug
    print("\nEvaluating BC policy...")
    
    # Quick sanity check on one file
    class DebugController(BaseController):
        def __init__(self):
            self.actions = []
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            state_vec = build_state(target_lataccel, current_lataccel, state, future_plan)
            action, _, raw = actor_critic.act(state_vec, deterministic=True)
            action_val = float(action[0]) if len(action.shape) > 0 else float(action)
            self.actions.append((action_val, float(raw[0,0])))
            return action_val
    
    debug_ctrl = DebugController()
    sim = TinyPhysicsSimulator(model, train_files[0], controller=debug_ctrl, debug=False)
    cost = sim.rollout()
    actions = np.array([a[0] for a in debug_ctrl.actions])
    raws = np.array([a[1] for a in debug_ctrl.actions])
    print(f"Debug on {train_files[0].split('/')[-1]}:")
    print(f"  Cost: {cost['total_cost']:.2f}")
    print(f"  Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}, std={actions.std():.3f}")
    print(f"  Raw actions: min={raws.min():.3f}, max={raws.max():.3f}, mean={raws.mean():.3f}, std={raws.std():.3f}")
    
    bc_cost = evaluate_policy(actor_critic, model, train_files, n_eval=20)
    print(f"\nBC Baseline (20 files): {bc_cost:.2f}")
    
    # Stage 2: PPO Fine-tuning
    train_ppo(actor_critic, model, train_files)

if __name__ == '__main__':
    main()

