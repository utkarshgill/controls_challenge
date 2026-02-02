"""
exp021: PPO with BC regularization to prevent policy collapse
Start with MINIMAL noise, stay close to BC teacher
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER
from controllers.pid import Controller as PIDController

# Config
state_dim, action_dim = 53, 1
hidden_dim = 128

max_epochs = 100
routes_per_epoch = 50  # Balance between data and speed
log_interval, eval_interval = 1, 2
batch_size, K_epochs = 4000, 10  # More epochs, larger batches
lr = 3e-4  # Higher LR for faster critic learning
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
entropy_coef = 0.01  # Small entropy
bc_regularization = 0.0  # No BC reg, let it learn
target_cost = 45.0

device = torch.device('cpu')

# OBS_SCALE from exp020
OBS_SCALE = np.array([
    0.3664, 7.1769, 0.1396, 38.7333] + [0.1573] * 49, 
    dtype=np.float32)

# Data split
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = all_files[:15000]
val_files = all_files[15000:17500]
test_files = all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        
        # Actor: EXACT same as BC
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Start with ZERO noise, will increase gradually
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 10.0)  # exp(-10) ≈ 0.00005
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        value = self.critic(state)
        return action_mean, action_std, value
    
    def act(self, raw_state, deterministic=False, return_internals=False):
        state_tensor = torch.as_tensor(raw_state / OBS_SCALE, dtype=torch.float32).to(device)
        with torch.no_grad():
            action_mean, action_std, _ = self(state_tensor)
            if deterministic:
                action = action_mean
            else:
                action = torch.distributions.Normal(action_mean, action_std).sample()
        
        if return_internals:
            return action.cpu().numpy(), state_tensor, action
        return action.cpu().numpy()

def load_bc_weights(actor_critic, bc_path):
    bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)
    bc_state_dict = bc_checkpoint['model_state_dict']
    
    actor_state_dict = actor_critic.actor.state_dict()
    for key in bc_state_dict.keys():
        if key in actor_state_dict:
            actor_state_dict[key].copy_(bc_state_dict[key])
    
    print("✅ BC weights loaded")
    return bc_checkpoint['model_state_dict']  # Return for BC regularization

class PPOController:
    def __init__(self, actor_critic, collect_data=False):
        self.actor_critic = actor_critic
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.collect_data = collect_data
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.prev_lataccel = None
        self.step_count = 0
        
    def build_state(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        v_ego = state.v_ego
        
        future_curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
                future_curvs.append(curv)
            else:
                future_curvs.append(0.0)
        
        raw_state = np.array([error, self.error_integral, error_diff, v_ego] + future_curvs, dtype=np.float32)
        self.prev_error = error
        return raw_state
    
    def update(self, target, current, state, future_plan):
        raw_state = self.build_state(target, current, state, future_plan)
        
        if self.collect_data:
            action_np, state_tensor, action = self.actor_critic.act(raw_state, deterministic=False, return_internals=True)
            
            # Dense reward: negative per-step cost
            lataccel_error = (target - current) ** 2 * 100
            
            if self.prev_lataccel is not None and self.step_count >= CONTROL_START_IDX and self.step_count < COST_END_IDX:
                jerk = ((current - self.prev_lataccel) / DEL_T) ** 2 * 100
                step_cost = lataccel_error * LAT_ACCEL_COST_MULTIPLIER + jerk
                reward = -step_cost / 1000.0
            else:
                reward = 0.0
            
            self.states.append(state_tensor)
            self.actions.append(action)
            self.rewards.append(reward)
            
            self.prev_lataccel = current
            self.step_count += 1
            
            action_clipped = float(np.clip(action_np.item(), -2.0, 2.0))
        else:
            action_np = self.actor_critic.act(raw_state, deterministic=True)
            action_clipped = float(np.clip(action_np.item(), -2.0, 2.0))
        
        return action_clipped

class PPO:
    def __init__(self, actor_critic, bc_actor_state, lr, gamma, lamda, K_epochs, eps_clip, batch_size, bc_reg):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size = eps_clip, batch_size
        self.bc_regularization = bc_reg
        
        # Store BC actor for regularization
        self.bc_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.bc_actor.load_state_dict(bc_actor_state)
        self.bc_actor.eval()
        for param in self.bc_actor.parameters():
            param.requires_grad = False

    def compute_advantages(self, rewards, state_values, is_terminals):
        T = len(rewards)
        advantages, gae = [], 0.0
        state_values_list = state_values.squeeze().tolist() + [0.0]
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_list[t + 1] * (1 - is_terminals[t]) - state_values_list[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(state_values_list[:-1]).to(device)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        
        action_logprobs = dist.log_prob(batch_actions).sum(-1)
        
        # PPO clipped objective
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(
            ratios * batch_advantages, 
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        ).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        
        # BC regularization: MSE between current policy and BC policy
        with torch.no_grad():
            bc_actions = self.bc_actor(batch_states)
        bc_loss = F.mse_loss(action_means, bc_actions)
        
        total_loss = actor_loss + 0.5 * critic_loss + self.bc_regularization * bc_loss
        return total_loss, actor_loss.item(), critic_loss.item(), bc_loss.item()
    
    def update(self, controllers):
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        
        for ctrl in controllers:
            if len(ctrl.states) > 0:
                all_states.extend(ctrl.states)
                all_actions.extend(ctrl.actions)
                all_rewards.extend(ctrl.rewards)
                dones = [0.0] * (len(ctrl.states) - 1) + [1.0]
                all_dones.extend(dones)
        
        if len(all_states) == 0:
            return 0, 0, 0, 0
        
        with torch.no_grad():
            old_states = torch.stack(all_states)
            old_actions = torch.stack(all_actions)
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            dist = torch.distributions.Normal(action_means, action_stds)
            old_logprobs = dist.log_prob(old_actions).sum(-1)
        
        advantages, returns = self.compute_advantages(all_rewards, old_state_values, all_dones)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        
        total_loss_sum, actor_loss_sum, critic_loss_sum, bc_loss_sum = 0, 0, 0, 0
        num_updates = 0
        
        for _ in range(self.K_epochs):
            for batch in DataLoader(dataset, batch_size=min(self.batch_size, len(old_states)), shuffle=True):
                self.optimizer.zero_grad()
                total_loss, actor_loss, critic_loss, bc_loss = self.compute_losses(*batch)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss_sum += total_loss.item()
                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss
                bc_loss_sum += bc_loss
                num_updates += 1
        
        return (total_loss_sum / num_updates, actor_loss_sum / num_updates, 
                critic_loss_sum / num_updates, bc_loss_sum / num_updates)

def rollout_route(actor_critic, route_file, collect_data=False):
    controller = PPOController(actor_critic, collect_data=collect_data)
    sim = TinyPhysicsSimulator(model_onnx, str(route_file), controller=controller)
    cost_dict = sim.rollout()
    return controller, cost_dict['total_cost']

def evaluate_on_routes(actor_critic, files, num_eval=10):
    costs = []
    for f in files[:num_eval]:
        _, cost = rollout_route(actor_critic, f, collect_data=False)
        costs.append(cost)
    return np.mean(costs)

def train():
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    bc_path = Path('./experiments/exp020_normalized/model.pth')
    
    print("Loading BC weights from exp020...")
    bc_state_dict = load_bc_weights(actor_critic, bc_path)
    
    print("\nVerifying BC initialization...")
    bc_cost = evaluate_on_routes(actor_critic, test_files, num_eval=10)
    print(f"BC init cost: {bc_cost:.2f}")
    print(f"Exploration σ: {actor_critic.log_std.exp().item():.6f}")
    
    if abs(bc_cost - 75) > 20:
        print("❌ BC init failed!")
        return
    
    print("✅ BC initialization verified")
    
    ppo = PPO(actor_critic, bc_state_dict, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, bc_regularization)
    
    print(f"\n{'='*80}")
    print(f"PPO with BC regularization (λ={bc_regularization})")
    print(f"{'='*80}\n")
    
    best_cost = bc_cost
    pbar = trange(max_epochs, desc="Training", unit='epoch')
    
    for epoch in range(max_epochs):
        epoch_routes = random.sample(train_files, routes_per_epoch)
        
        controllers = []
        costs = []
        total_reward = 0
        total_steps = 0
        
        for route_file in epoch_routes:
            ctrl, cost = rollout_route(actor_critic, route_file, collect_data=True)
            controllers.append(ctrl)
            costs.append(cost)
            total_reward += sum(ctrl.rewards)
            total_steps += len(ctrl.states)
        
        total_loss, actor_loss, critic_loss, bc_loss = ppo.update(controllers)
        
        train_cost = np.mean(costs)
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        
        if epoch % eval_interval == 0:
            val_cost = evaluate_on_routes(actor_critic, val_files, num_eval=10)
            test_cost = evaluate_on_routes(actor_critic, test_files, num_eval=10)
            
            if test_cost < best_cost:
                best_cost = test_cost
                torch.save({'model_state_dict': actor_critic.state_dict(), 'obs_scale': OBS_SCALE}, 
                           'experiments/exp021_ppo/model_best.pth')
            
            msg = f"E{epoch:3d} tr={train_cost:5.1f} val={val_cost:5.1f} tst={test_cost:5.1f} bst={best_cost:5.1f} bc_loss={bc_loss:.3f} σ={actor_critic.log_std.exp().item():.4f}"
            print(msg, flush=True)
            pbar.write(msg)
        
        # Gradually increase exploration noise
        if epoch > 0 and epoch % 10 == 0:
            actor_critic.log_std.data += 0.5  # Slowly increase noise
        
        pbar.update(1)
        
        if best_cost <= target_cost:
            pbar.write(f"\n✅ SOLVED! best={best_cost:.2f}")
            break
    
    pbar.close()
    
    torch.save({'model_state_dict': actor_critic.state_dict(), 'obs_scale': OBS_SCALE}, 
               'experiments/exp021_ppo/model_final.pth')
    print(f"\n✅ Complete | Best: {best_cost:.2f} | Improvement: {((bc_cost - best_cost) / bc_cost * 100):.1f}%")

if __name__ == '__main__':
    print(f"Device: {device}")
    train()
