"""
exp011: Proper PPO with multi-file training (using beautiful_lander.py scaffolding)
State: [error, error_integral, error_derivative, v_ego] (4 features)
Network: Small (32 hidden units, trunk + heads like beautiful_lander)
Training: Cycles through 100 files
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel

# ===== Config (inspired by beautiful_lander.py) =====
STATE_DIM, ACTION_DIM = 4, 1
NUM_FILES = 100  # Rotate through these
HIDDEN_DIM = 32
TRUNK_LAYERS, HEAD_LAYERS = 1, 1  # Keep it simple
LR = 1e-3
GAMMA, GAE_LAMBDA, EPS_CLIP = 0.99, 0.95, 0.2
ENTROPY_COEF = 0.001
BATCH_SIZE, K_EPOCHS = 2048, 10
STEPS_PER_EPOCH = 20000  # ~10 episodes per epoch
MAX_EPOCHS = 500
LOG_INTERVAL, EVAL_INTERVAL = 5, 25

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/Users/engelbart/Desktop/stuff/controls_challenge/models/tinyphysics.onnx'
DATA_FOLDER = '/Users/engelbart/Desktop/stuff/controls_challenge/data'

def build_state(target_lataccel, current_lataccel, state, prev_error, error_integral):
    """4 features: error, error_integral, error_derivative, v_ego"""
    error = target_lataccel - current_lataccel
    error_derivative = error - prev_error
    v_ego = state.v_ego
    return np.array([error, error_integral, error_derivative, v_ego], dtype=np.float32)

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing (from beautiful_lander.py)"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

# ===== ActorCritic (from beautiful_lander.py) =====
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
        super(ActorCritic, self).__init__()
        
        # Shared trunk
        trunk = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(trunk_layers - 1):
            trunk.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.trunk = nn.Sequential(*trunk)
        
        # Actor head
        self.actor_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                           for layer in [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]])
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic_layers = nn.Sequential(*[layer for _ in range(head_layers)
                                            for layer in [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]])
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
    def act(self, state, deterministic=False, return_internals=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(DEVICE)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action_mean, action_std, _ = self(state_tensor)
        raw_action = action_mean if deterministic else torch.distributions.Normal(action_mean, action_std).sample()
        action = torch.tanh(raw_action)
        if return_internals:
            return action.cpu().numpy(), state_tensor, raw_action
        return action.cpu().numpy()

# ===== PPO (from beautiful_lander.py) =====
class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.states, self.actions = [], []
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.entropy_coef = eps_clip, batch_size, entropy_coef

    def __call__(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(state, deterministic=False, return_internals=True)
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        """GAE computation (from beautiful_lander.py)"""
        T = len(rewards)
        advantages, gae = [], 0
        state_values_list = state_values + [state_values[-1]]
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_list[t + 1] * (1 - is_terminals[t]) - state_values_list[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = advantages + torch.tensor(state_values[:-1] if len(state_values) > 1 else state_values, dtype=torch.float32, device=DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        """Loss computation (from beautiful_lander.py)"""
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
        """PPO update (from beautiful_lander.py)"""
        with torch.no_grad():
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            state_values_list = old_state_values.squeeze(-1).tolist()
            advantages, returns = self.compute_advantages(rewards, state_values_list, dones)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states, self.actions = [], []

# ===== Multi-file Environment (adapted from exp004) =====
class TinyPhysicsMultiFileEnv:
    """Cycles through multiple files for training"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.current_file_idx = 0
        self.sim = None
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_cost = 0.0
    
    def reset(self):
        file_path = self.file_paths[self.current_file_idx % len(self.file_paths)]
        self.current_file_idx += 1
        
        model = TinyPhysicsModel(MODEL_PATH, debug=False)
        self.sim = TinyPhysicsSimulator(model, file_path, controller=None, debug=False)
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_cost = 0.0
        
        state, target, _ = self.sim.get_state_target_futureplan(self.sim.step_idx)
        obs = build_state(target, self.sim.current_lataccel, state, self.prev_error, self.error_integral)
        return obs
    
    def step(self, action):
        # Apply action (copied from exp004)
        action_clipped = np.clip(action[0], -2.0, 2.0)
        self.sim.action_history.append(action_clipped)
        
        # Simulate
        self.sim.sim_step(self.sim.step_idx)
        self.sim.step_idx += 1
        
        # Check if done
        done = self.sim.step_idx >= len(self.sim.data) - 50
        
        # Get next observation
        if not done:
            state, target, _ = self.sim.get_state_target_futureplan(self.sim.step_idx)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            current_lataccel = self.sim.current_lataccel
            
            error = target - current_lataccel
            self.error_integral += error
            self.error_integral = np.clip(self.error_integral, -14, 14)
            
            obs = build_state(target, current_lataccel, state, self.prev_error, self.error_integral)
            self.prev_error = error
        else:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
        
        # Compute reward (from exp004)
        if len(self.sim.current_lataccel_history) > 1:
            lat_cost = (self.sim.target_lataccel_history[-1] - self.sim.current_lataccel) ** 2
            jerk = (self.sim.current_lataccel_history[-1] - self.sim.current_lataccel_history[-2]) / 0.1
            jerk_cost = jerk ** 2
            step_cost = (50 * lat_cost + jerk_cost)
            self.episode_cost += step_cost
            reward = -step_cost / 100.0
        else:
            reward = 0.0
        
        return obs, reward, done

# ===== Multi-file rollout =====
def rollout(file_paths, policy, num_steps):
    """Rollout across multiple files (like beautiful_lander.py)"""
    rewards, dones, ep_returns = [], [], []
    step_count = 0
    
    env = TinyPhysicsMultiFileEnv(file_paths)
    state = env.reset()
    
    while step_count < num_steps:
        action = policy(state)
        next_state, reward, done = env.step(action)
        
        rewards.append(reward)
        dones.append(float(done))
        step_count += 1
        
        if done:
            ep_returns.append(env.episode_cost)
            state = env.reset()  # Environment auto-cycles to next file
        else:
            state = next_state
    
    return rewards, dones, ep_returns

def train_one_epoch(file_paths, ppo):
    """Train for one epoch (like beautiful_lander.py)"""
    rewards, dones, ep_rets = rollout(file_paths, ppo, num_steps=STEPS_PER_EPOCH)
    ppo.update(rewards, dones)
    return ep_rets

def evaluate_policy(actor_critic, eval_files, n=20):
    """Evaluate on held-out files"""
    costs = []
    for i, file_path in enumerate(eval_files[:n]):
        model = TinyPhysicsModel(MODEL_PATH, debug=False)
        sim = TinyPhysicsSimulator(model, file_path, controller=None, debug=False)
        prev_error, error_integral = 0.0, 0.0
        
        while sim.step_idx < len(sim.data) - 50:
            state, target, _ = sim.get_state_target_futureplan(sim.step_idx)
            obs = build_state(target, sim.current_lataccel, state, prev_error, error_integral)
            
            action = actor_critic.act(obs, deterministic=True)
            action_clipped = np.clip(action[0], -2.0, 2.0)
            sim.action_history.append(action_clipped)
            sim.sim_step(sim.step_idx)
            sim.step_idx += 1
            
            if sim.step_idx < len(sim.data) - 50:
                state, target, _ = sim.get_state_target_futureplan(sim.step_idx)
                sim.state_history.append(state)
                sim.target_lataccel_history.append(target)
                error = target - sim.current_lataccel
                error_integral += error
                error_integral = np.clip(error_integral, -14, 14)
                prev_error = error
        
        cost_dict = sim.compute_cost()
        costs.append(cost_dict['total_cost'])
    
    return np.mean(costs)

def train():
    print("=" * 60)
    print("EXP011: Multi-file PPO (beautiful_lander scaffolding)")
    print(f"State: 4 features, Network: {HIDDEN_DIM} hidden units")
    print(f"Training on {NUM_FILES} files, {MAX_EPOCHS} epochs")
    print("=" * 60)
    
    # Load files
    all_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')])
    train_files = [os.path.join(DATA_FOLDER, f) for f in all_files[:NUM_FILES]]
    eval_files = [os.path.join(DATA_FOLDER, f) for f in all_files[NUM_FILES:NUM_FILES+20]]
    
    # Initialize
    actor_critic = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TRUNK_LAYERS, HEAD_LAYERS).to(DEVICE)
    ppo = PPO(actor_critic, LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, ENTROPY_COEF)
    
    all_episode_costs = []
    best_eval = float('inf')
    os.makedirs('results', exist_ok=True)
    
    pbar = trange(MAX_EPOCHS, desc="Training", unit='epoch')
    for epoch in range(MAX_EPOCHS):
        ep_costs = train_one_epoch(train_files, ppo)
        all_episode_costs.extend(ep_costs)
        pbar.update(1)
        
        train_100 = np.mean(all_episode_costs[-100:]) if len(all_episode_costs) >= 100 else np.mean(all_episode_costs)
        
        # Evaluate
        if epoch % EVAL_INTERVAL == 0:
            eval_cost = evaluate_policy(actor_critic, eval_files, n=20)
            if eval_cost < best_eval:
                best_eval = eval_cost
                torch.save(actor_critic.state_dict(), 'results/ppo_best.pth')
            
            if epoch % LOG_INTERVAL == 0:
                s = actor_critic.log_std.exp().detach().cpu().numpy()
                pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  train={np.mean(ep_costs):7.1f}±{np.std(ep_costs):5.1f}  train_100={train_100:6.1f}  eval={eval_cost:6.1f}  best={best_eval:6.1f}  σ={s[0]:.3f}")
        elif epoch % LOG_INTERVAL == 0:
            s = actor_critic.log_std.exp().detach().cpu().numpy()
            pbar.write(f"Epoch {epoch:3d}  n_ep={len(ep_costs):3d}  train={np.mean(ep_costs):7.1f}±{np.std(ep_costs):5.1f}  train_100={train_100:6.1f}  σ={s[0]:.3f}")
        
        # Early stopping if diverging
        if epoch > 50 and train_100 > 50000:
            pbar.write("Training diverged, stopping early")
            break
        
        # Success!
        if best_eval < 45:
            pbar.write(f"\n{'='*60}\nSOLVED at epoch {epoch}! eval={best_eval:.1f} < 45\n{'='*60}")
            break
    
    pbar.close()
    print(f"\nTraining complete! Best eval cost: {best_eval:.1f}")

if __name__ == '__main__':
    print(f"Using {DEVICE} device")
    train()
