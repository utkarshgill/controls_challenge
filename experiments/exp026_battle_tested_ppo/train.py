"""
Battle-tested PPO from beautiful_lander.py, adapted for controls challenge
Key changes from our broken PPO:
1. Parallel route loading (like AsyncVectorEnv)
2. Proper tanh_log_prob (change of variables)
3. Larger batch collection before update
4. Learnable per-action σ
5. Exactly follow beautiful_lander.py structure
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
import multiprocessing as mp
from functools import partial

device = torch.device('cpu')

# Normalization (from exp025)
BASE_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333, 0.5], dtype=np.float32)
CURV_SCALE = 0.1573

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files, val_files, test_files = all_files[:15000], all_files[15000:17500], all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Hyperparams from beautiful_lander.py
max_epochs = 100
steps_per_epoch = 50_000  # Collect 50k steps before each update
batch_size, K_epochs = 10_000, 10
hidden_dim = 128
lr = 1e-3
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
entropy_coef = 0.001

def tanh_log_prob(raw_action, dist):
    """Change of variables for tanh squashing - CRITICAL for correct gradients!"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv on future curvatures
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # Shared trunk (5 + 128 = 133)
        self.trunk = nn.Sequential(
            nn.Linear(5 + 16*8, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (3 layers like beautiful_lander)
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # Learnable per action
        
        # Critic head
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.critic_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, base, curv):
        # base: (batch, 5), curv: (batch, 49)
        conv_out = self.conv(curv.unsqueeze(1))
        conv_flat = conv_out.reshape(conv_out.size(0), -1)
        combined = torch.cat([base, conv_flat], dim=1)
        
        trunk_feat = self.trunk(combined)
        
        actor_feat = self.actor_layers(trunk_feat)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        
        critic_feat = self.critic_layers(trunk_feat)
        value = self.critic_out(critic_feat)
        
        return action_mean, action_std, value
    
    @torch.no_grad()
    def act(self, base, curv, deterministic=False, return_internals=False):
        base_t = torch.as_tensor(base, dtype=torch.float32).to(device)
        curv_t = torch.as_tensor(curv, dtype=torch.float32).to(device)
        
        action_mean, action_std, _ = self(base_t, curv_t)
        
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
        
        action = torch.tanh(raw_action)
        
        if return_internals:
            return action.cpu().numpy(), base_t, curv_t, raw_action
        return action.cpu().numpy()

class PPO:
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        # Storage for rollout data
        self.base_states = []
        self.curv_states = []
        self.actions = []
    
    def __call__(self, base, curv):
        """Collect action during rollout"""
        action_np, base_t, curv_t, raw_action = self.actor_critic.act(
            base, curv, deterministic=False, return_internals=True
        )
        
        self.base_states.append(base_t)
        self.curv_states.append(curv_t)
        self.actions.append(raw_action)
        
        return action_np
    
    def compute_advantages(self, rewards, state_values, is_terminals):
        """GAE - exactly from beautiful_lander.py"""
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + gamma * gae_lambda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_base, batch_curv, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        """Loss computation - exactly from beautiful_lander.py"""
        action_means, action_stds, state_values = self.actor_critic(batch_base, batch_curv)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(
            ratios * batch_advantages,
            torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages
        ).mean()
        
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        entropy = dist.entropy().sum(-1).mean()
        
        return actor_loss + critic_loss - entropy_coef * entropy
    
    def update(self, rewards, dones):
        """Update - exactly from beautiful_lander.py"""
        with torch.no_grad():
            rewards_t = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(device)
            
            old_base = torch.cat(self.base_states)
            old_curv = torch.cat(self.curv_states)
            old_actions = torch.cat(self.actions)
            
            action_means, action_stds, old_state_values = self.actor_critic(old_base, old_curv)
            old_logprobs = tanh_log_prob(old_actions, torch.distributions.Normal(action_means, action_stds))
            old_state_values = old_state_values.squeeze(-1).view(-1, rewards_t.size(1))
            
            advantages, returns = self.compute_advantages(rewards_t, old_state_values, is_terms)
        
        dataset = TensorDataset(old_base, old_curv, old_actions, old_logprobs, advantages, returns)
        
        for _ in range(K_epochs):
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                self.optimizer.zero_grad()
                self.compute_losses(*batch).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # Clear storage
        self.base_states, self.curv_states, self.actions = [], [], []

class Controller:
    """Wraps PPO for tinyphysics simulator"""
    def __init__(self, ppo):
        self.ppo = ppo
        self.ei, self.pe, self.pa = 0.0, 0.0, 0.0
        
        # Storage for episode trajectory
        self.traj_base = []
        self.traj_curv = []
        self.traj_rewards = []
    
    def reset(self):
        self.ei, self.pe, self.pa = 0.0, 0.0, 0.0
        self.traj_base, self.traj_curv, self.traj_rewards = [], [], []
    
    def update(self, target, current, state, future_plan):
        from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER
        
        e = target - current
        self.ei += e
        ed = e - self.pe
        
        # State: [error, error_i, error_d, v_ego, prev_action]
        base = np.array([e, self.ei, ed, state.v_ego, self.pa], dtype=np.float32)
        base_norm = base / BASE_SCALE
        
        # Future curvatures
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                lat = future_plan.lataccel[i]
                curv = (lat - future_roll) / max(future_v ** 2, 1.0)
                curvs.append(curv)
            else:
                curvs.append(0.0)
        curv_norm = np.array(curvs, dtype=np.float32) / CURV_SCALE
        
        # Get action from PPO
        action = self.ppo(base_norm, curv_norm).item()
        
        # Compute reward (negative cost, like LunarLander has positive/negative rewards)
        step_idx = len(self.traj_rewards)
        if step_idx >= CONTROL_START_IDX and step_idx < COST_END_IDX:
            lat_err = (target - current) ** 2 * 100
            if len(self.traj_rewards) > 0:
                prev_lataccel = current  # Approximation
                jerk = ((current - prev_lataccel) / DEL_T) ** 2 * 100
            else:
                jerk = 0
            cost = lat_err * LAT_ACCEL_COST_MULTIPLIER + jerk
            reward = -cost / 100.0  # Scale down
        else:
            reward = 0.0
        
        self.traj_base.append(base_norm)
        self.traj_curv.append(curv_norm)
        self.traj_rewards.append(reward)
        
        self.pe = e
        self.pa = action
        
        return float(np.clip(action, -2.0, 2.0))
    
    def get_trajectory(self):
        return self.traj_base, self.traj_curv, self.traj_rewards

def rollout_parallel(file_paths, ppo):
    """Rollout multiple routes - simulates AsyncVectorEnv"""
    all_rewards, all_dones = [], []
    n_files = len(file_paths)
    
    controllers = [Controller(ppo) for _ in range(n_files)]
    sims = [TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl) 
            for f, ctrl in zip(file_paths, controllers)]
    
    # Run all simulations
    for sim in sims:
        sim.rollout()
    
    # Collect trajectories
    max_len = max(len(ctrl.traj_rewards) for ctrl in controllers)
    
    for t in range(max_len):
        step_rewards = [ctrl.traj_rewards[t] if t < len(ctrl.traj_rewards) else 0.0 
                       for ctrl in controllers]
        step_dones = [float(t == len(ctrl.traj_rewards) - 1) if t < len(ctrl.traj_rewards) else 1.0
                      for ctrl in controllers]
        all_rewards.append(step_rewards)
        all_dones.append(step_dones)
    
    return all_rewards, all_dones

def evaluate(actor_critic, files, n=20):
    """Evaluate deterministic policy"""
    costs = []
    for f in files[:n]:
        class EvalCtrl:
            def __init__(self):
                self.ei, self.pe, self.pa = 0.0, 0.0, 0.0
            
            def update(self, target, current, state, future_plan):
                e = target - current
                self.ei += e
                ed = e - self.pe
                base = np.array([e, self.ei, ed, state.v_ego, self.pa], dtype=np.float32) / BASE_SCALE
                
                curvs = []
                for i in range(49):
                    if i < len(future_plan.lataccel):
                        future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                        future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                        lat = future_plan.lataccel[i]
                        curv = (lat - future_roll) / max(future_v ** 2, 1.0)
                        curvs.append(curv)
                    else:
                        curvs.append(0.0)
                curv = np.array(curvs, dtype=np.float32) / CURV_SCALE
                
                action = actor_critic.act(base, curv, deterministic=True).item()
                
                self.pe = e
                self.pa = action
                return float(np.clip(action, -2.0, 2.0))
        
        ctrl = EvalCtrl()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    
    return np.mean(costs)

def train():
    print("Battle-tested PPO from beautiful_lander.py")
    print("=" * 60)
    
    # Load BC weights
    actor_critic = ActorCritic().to(device)
    bc_ckpt = torch.load('experiments/exp025_with_history/model.pth', map_location='cpu', weights_only=False)
    
    # Map BC weights
    ac_state = actor_critic.state_dict()
    for k, v in bc_ckpt['model_state_dict'].items():
        if k.startswith('conv.'):
            ac_state[k] = v
        elif k.startswith('mlp.'):
            # Map mlp to actor
            actor_key = 'actor_layers.' + k[4:] if 'mlp.0' in k or 'mlp.2' in k or 'mlp.4' in k else None
            if actor_key and actor_key in ac_state:
                ac_state[actor_key] = v
            elif k == 'mlp.4.weight':  # Final layer
                ac_state['actor_mean.weight'] = v
            elif k == 'mlp.4.bias':
                ac_state['actor_mean.bias'] = v
    
    actor_critic.load_state_dict(ac_state)
    print("✅ Loaded BC weights")
    
    bc_cost = evaluate(actor_critic, test_files, 20)
    print(f"BC baseline: {bc_cost:.2f}")
    print(f"Target: <45 (need {bc_cost - 45:.1f} improvement)")
    print(f"σ: {actor_critic.log_std.exp().item():.4f}\n")
    
    ppo = PPO(actor_critic)
    best_cost = bc_cost
    
    for epoch in trange(max_epochs):
        # Sample routes for this epoch (batch of 10 = ~10k steps)
        epoch_files = random.sample(train_files, 10)
        
        # Rollout in parallel
        rewards, dones = rollout_parallel(epoch_files, ppo)
        
        # Update
        ppo.update(rewards, dones)
        
        # Evaluate every 2 epochs
        if epoch % 2 == 0:
            test_cost = evaluate(actor_critic, test_files, 20)
            if test_cost < best_cost:
                best_cost = test_cost
                torch.save({'model_state_dict': actor_critic.state_dict()}, 
                          'experiments/exp026_battle_tested_ppo/best.pth')
            
            print(f"E{epoch:2d}  test={test_cost:5.1f}  best={best_cost:5.1f}  σ={actor_critic.log_std.exp().item():.4f}")
            
            if best_cost < 45:
                print(f"\n🎯 SOLVED! best={best_cost:.2f} < 45")
                break
    
    print(f"\n✅ Final best: {best_cost:.2f}")

if __name__ == '__main__':
    train()

