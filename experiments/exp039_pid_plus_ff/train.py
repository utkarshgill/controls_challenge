"""
Experiment 039: PID + Learned Feedforward

Surgical approach inspired by beautiful_lander.py:
- PID controller (fixed, proven stable baseline)
- Feedforward network (learned via PPO)
- FF takes future plan, outputs anticipatory steering
- Total action = PID(error) + FF(future_plan)

Architecture:
- FF: 1D conv over future [lataccel, v_ego, a_ego, roll_lataccel] × 50 timesteps
- PPO trains only FF network, PID stays frozen
- Start from ~75 cost (PID), learn to beat it

Key insight: SUBTLE exploration
- Controls are tight - large noise breaks everything
- Start with deterministic warmup (10 epochs)
- Then explore with σ=0.05 (not σ=1.0)
- This lets PPO discover improvements without destroying baseline
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
from tqdm import trange
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, State, FuturePlan, run_rollout
from controllers import BaseController
from multiprocessing import Pool
from functools import partial

# Environment config
num_envs = int(os.getenv('NUM_ENVS', 8))  # Reduced for speed
max_epochs = 200
steps_per_epoch = 100_000
log_interval = 5
eval_interval = 10

# PPO hyperparameters (adjusted for controls)
batch_size = 2048
K_epochs = 5  # Reduced from 20 for speed
hidden_dim = 128
pi_lr = 1e-4  # Reduced from 3e-4 to prevent oscillation
vf_lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
vf_coef = 0.5
entropy_coef = 0.001  # Very low - we want subtle exploration for tight control

# Device setup
METAL = bool(int(os.getenv('METAL', '0')))
device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

# Load data
# Use FIRST 100 files for eval (same as official tinyphysics.py --num_segs 100)
all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]  # Official eval files (00000.csv to 00099.csv)

# For training, use random subset excluding eval files
all_files = list(DATA_PATH.glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files = [f for f in all_files if f not in eval_files][:15000]
test_files = all_files_sorted[100:200]  # Next 100 for test

# Load simulator
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# PID baseline from controllers/pid.py
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053


def prepare_future_plan(future_plan, state):
    """Convert FuturePlan to tensor [4, 50]"""
    # Handle edge case: empty or short future plan at end of episode
    def pad_to_50(arr, default=0.0):
        if len(arr) == 0:
            return np.full(50, default, dtype=np.float32)
        elif len(arr) >= 50:
            return np.array(arr[:50], dtype=np.float32)
        else:
            # Pad with last value
            return np.array(list(arr) + [arr[-1]] * (50 - len(arr)), dtype=np.float32)
    
    lataccel = pad_to_50(future_plan.lataccel, 0.0)
    v_ego = pad_to_50(future_plan.v_ego, state.v_ego if hasattr(state, 'v_ego') else 20.0)
    a_ego = pad_to_50(future_plan.a_ego, 0.0)
    roll_lataccel = pad_to_50(future_plan.roll_lataccel, 0.0)
    
    return np.stack([lataccel, v_ego, a_ego, roll_lataccel], axis=0).astype(np.float32)


class FeedforwardNetwork(nn.Module):
    """1D conv over future plan to predict anticipatory steering"""
    def __init__(self):
        super(FeedforwardNetwork, self).__init__()
        # Input: 4 channels (lataccel, v_ego, a_ego, roll_lataccel) × 50 timesteps
        self.conv1 = nn.Conv1d(4, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Keep 50 timesteps
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)  # Keep 50 timesteps
        
        self.fc1 = nn.Linear(64 * 50, 128)  # 50 timesteps preserved
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, future_plan):
        # future_plan: [batch, 4, 50]
        x = torch.relu(self.conv1(future_plan))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        ff_action = self.fc3(x)
        return ff_action


class ActorCritic(nn.Module):
    """Actor outputs FF action (mean + std), Critic estimates value"""
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        # Actor: FF network + log_std parameter
        self.actor = FeedforwardNetwork()
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # Critic: separate value network (also sees future plan)
        self.critic_conv1 = nn.Conv1d(4, 16, kernel_size=5, padding=2)
        self.critic_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)  # Keep 50 timesteps
        self.critic_fc1 = nn.Linear(32 * 50, 128)  # 50 timesteps preserved
        self.critic_fc2 = nn.Linear(128, 64)
        self.critic_fc3 = nn.Linear(64, 1)
    
    def forward(self, future_plan):
        # Actor
        action_mean = self.actor(future_plan)
        action_std = self.log_std.exp()
        
        # Critic
        x = torch.relu(self.critic_conv1(future_plan))
        x = torch.relu(self.critic_conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.critic_fc1(x))
        x = torch.relu(self.critic_fc2(x))
        value = self.critic_fc3(x)
        
        return action_mean, action_std, value
    
    @torch.inference_mode()
    def act(self, future_plan_np, deterministic=False):
        """Sample action for rollout"""
        dev = next(self.parameters()).device
        future_plan = torch.from_numpy(future_plan_np).to(device=dev, dtype=torch.float32)
        if future_plan.ndim == 2:
            future_plan = future_plan.unsqueeze(0)
        
        action_mean, action_std, _ = self(future_plan)
        
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = torch.distributions.Normal(action_mean, action_std).sample()
        
        # Output is FF term (will be added to PID)
        action = torch.tanh(raw_action)  # Bound to [-1, 1]
        return action.cpu().numpy()[0, 0], raw_action.cpu().numpy()[0, 0]


class HybridController(BaseController):
    """Controller: PID (fixed) + FF (from policy network)"""
    def __init__(self, actor_critic, deterministic=True):
        self.actor_critic = actor_critic
        self.deterministic = deterministic
        self.error_integral = 0
        self.prev_error = 0
        self.presampled_ff = None  # For training: use pre-sampled action
        
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0
        self.presampled_ff = None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID term (fixed)
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        pid_action = PID_P * error + PID_I * self.error_integral + PID_D * error_diff
        
        # FF term: use pre-sampled if available (training), otherwise sample (eval)
        if self.presampled_ff is not None:
            ff_action = self.presampled_ff
            self.presampled_ff = None  # Consume it
        else:
            future_plan_array = prepare_future_plan(future_plan, state)
            ff_action, _ = self.actor_critic.act(future_plan_array, deterministic=self.deterministic)
        
        return pid_action + ff_action


class PPO:
    def __init__(self, actor_critic, pi_lr, vf_lr, gamma, lamda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.pi_optimizer = optim.Adam(list(actor_critic.actor.parameters()) + [actor_critic.log_std], lr=pi_lr)
        self.vf_optimizer = optim.Adam(list(actor_critic.critic_conv1.parameters()) + 
                                       list(actor_critic.critic_conv2.parameters()) +
                                       list(actor_critic.critic_fc1.parameters()) +
                                       list(actor_critic.critic_fc2.parameters()) +
                                       list(actor_critic.critic_fc3.parameters()), lr=vf_lr)
        self.gamma, self.lamda, self.K_epochs = gamma, lamda, K_epochs
        self.eps_clip, self.batch_size, self.vf_coef, self.entropy_coef = eps_clip, batch_size, vf_coef, entropy_coef

    def compute_advantages(self, rewards, state_values, is_terminals):
        T, N = rewards.shape
        advantages, gae = torch.zeros_like(rewards), torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def tanh_log_prob(self, raw_action, dist):
        action = torch.tanh(raw_action)
        logp_gaussian = dist.log_prob(raw_action).sum(-1)
        return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)
    
    def compute_loss(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = self.tanh_log_prob(batch_actions, dist)
        ratios = torch.exp(action_logprobs - batch_logprobs)
        actor_loss = -torch.min(ratios * batch_advantages, torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages).mean()
        critic_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)
        
        # Entropy of tanh-transformed distribution
        gaussian_entropy = dist.entropy().sum(-1)
        actions_squashed = torch.tanh(batch_actions)
        jacobian_correction = torch.log(1 - actions_squashed**2 + 1e-6).sum(-1)
        entropy = (gaussian_entropy - jacobian_correction).mean()
        
        return actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
    
    def update(self, future_plans, raw_actions, rewards, dones, debug=False):
        dev = next(self.actor_critic.parameters()).device
        T, N = rewards.shape
        
        # Convert to tensors
        future_plans_t = torch.from_numpy(future_plans).to(device=dev, dtype=torch.float32)  # [T, N, 4, 50]
        raw_actions_t = torch.from_numpy(raw_actions).to(device=dev, dtype=torch.float32)  # [T, N, 1]
        rewards_t = torch.from_numpy(rewards).to(device=dev, dtype=torch.float32)
        dones_t = torch.from_numpy(dones).to(device=dev, dtype=torch.float32)
        
        # Flatten for network forward
        B = T * N
        future_plans_flat = future_plans_t.reshape(B, 4, 50)
        raw_actions_flat = raw_actions_t.reshape(B, 1)
        
        with torch.no_grad():
            mean, std, val = self.actor_critic(future_plans_flat)
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = self.tanh_log_prob(raw_actions_flat, dist)
            old_values = val.squeeze(-1).view(T, N)
            advantages, returns = self.compute_advantages(rewards_t, old_values, dones_t)
        
        if debug:
            print(f"   Rewards: mean={rewards_t.mean():.2f}, std={rewards_t.std():.2f}")
            print(f"   Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"   Returns: mean={returns.mean():.2f}, std={returns.std():.2f}")
        
        # PPO epochs
        total_loss = 0
        for epoch_i in range(self.K_epochs):
            perm = torch.randperm(B, device=dev)
            for start in range(0, B, self.batch_size):
                idx = perm[start:start + self.batch_size]
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss = self.compute_loss(future_plans_flat[idx], raw_actions_flat[idx], 
                                        old_logprobs[idx], advantages[idx], returns[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std], max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(list(self.actor_critic.critic_conv1.parameters()) + 
                                              list(self.actor_critic.critic_conv2.parameters()) +
                                              list(self.actor_critic.critic_fc1.parameters()) +
                                              list(self.actor_critic.critic_fc2.parameters()) +
                                              list(self.actor_critic.critic_fc3.parameters()), max_norm=0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()
                total_loss += loss.item()
        
        if debug:
            print(f"   Avg loss: {total_loss / (self.K_epochs * (B // self.batch_size)):.4f}")


def rollout_episode(data_file, controller):
    """Run one episode and return cost"""
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
    cost = sim.rollout()
    return cost


def evaluate_policy(actor_critic, data_files, num_episodes=16, debug=False, use_multiprocessing=False):
    """Evaluate policy on given files - deterministic
    
    Args:
        actor_critic: The ActorCritic network
        data_files: List of data files to evaluate on
        num_episodes: Number of episodes to evaluate
        debug: If True, print FF action stats
        use_multiprocessing: If True, use official multiprocessing eval (matches tinyphysics.py)
    """
    if use_multiprocessing:
        # Use official multiprocessing evaluation
        # Write temp controller to controllers/ directory and save model
        temp_controller_path = Path(__file__).parent.parent.parent / 'controllers' / 'temp_eval_controller.py'
        temp_model_path = Path(__file__).parent / 'temp_eval_model.pth'
        torch.save(actor_critic.state_dict(), temp_model_path)
        
        # Write temporary controller file
        # Note: Path references need to be absolute since controller runs from controllers/ directory
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
spec = importlib.util.spec_from_file_location("exp039_train", train_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

class Controller(BaseController):
    def __init__(self):
        device = torch.device('cpu')
        self.actor_critic = train_module.ActorCritic().to(device)
        model_path = Path("{exp_dir}") / "temp_eval_model.pth"
        self.actor_critic.load_state_dict(torch.load(model_path, map_location=device))
        self.actor_critic.eval()
        self.hybrid = train_module.HybridController(self.actor_critic, deterministic=True)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.hybrid.update(target_lataccel, current_lataccel, state, future_plan)
    
    def reset(self):
        self.hybrid.reset()
'''
        temp_controller_path.write_text(controller_code)
        
        try:
            # Use official run_rollout with multiprocessing
            run_rollout_partial = partial(run_rollout, controller_type='temp_eval_controller', 
                                          model_path=str(MODEL_PATH), debug=False)
            files_to_eval = data_files[:num_episodes]
            results = process_map(run_rollout_partial, files_to_eval, max_workers=16, chunksize=10, disable=True)
            costs = [result[0]['total_cost'] for result in results]
            return float(np.mean(costs))
        finally:
            # Cleanup temp files
            if temp_controller_path.exists():
                temp_controller_path.unlink()
            if temp_model_path.exists():
                temp_model_path.unlink()
    else:
        # Fast sequential evaluation for training
        controller = HybridController(actor_critic, deterministic=True)
        costs = []
        ff_actions = []
        
        for i in range(min(num_episodes, len(data_files))):
            controller.reset()
            
            # For first episode, track FF actions if debug
            if debug and i == 0:
                # Monkey patch to collect FF actions
                original_update = controller.update
                episode_ff = []
                def debug_update(target, current, state, future_plan):
                    result = original_update(target, current, state, future_plan)
                    future_plan_array = prepare_future_plan(future_plan, state)
                    ff, _ = actor_critic.act(future_plan_array, deterministic=True)
                    episode_ff.append(ff)
                    return result
                controller.update = debug_update
            
            cost_dict = rollout_episode(data_files[i], controller)
            costs.append(cost_dict['total_cost'])
            
            if debug and i == 0:
                ff_actions = episode_ff
                print(f"   FF actions: mean={np.mean(ff_actions):.4f}, std={np.std(ff_actions):.4f}, range=[{np.min(ff_actions):.4f}, {np.max(ff_actions):.4f}]")
        
        return float(np.mean(costs))


class TrainingContext:
    def __init__(self):
        # Dual models: cpu for rollout, device for update
        self.ac_cpu = ActorCritic().to('cpu')
        self.ac_device = ActorCritic().to(device)
        
        # Initialize FF network: normal scale internally, small output
        for m in self.ac_device.actor.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                # Use normal gain for ReLU (sqrt(2)) for internal layers
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Only the FINAL layer should be small (so FF starts near zero)
        nn.init.orthogonal_(self.ac_device.actor.fc3.weight, gain=0.01)
        nn.init.zeros_(self.ac_device.actor.fc3.bias)
        
        # Initialize critic: normal scale for internal layers, SMALL for output
        critic_internal = [
            self.ac_device.critic_conv1, self.ac_device.critic_conv2,
            self.ac_device.critic_fc1, self.ac_device.critic_fc2
        ]
        for layer in critic_internal:
            if hasattr(layer, 'weight'):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Critic output layer: SMALL scale so values start near zero (not 100+)
        nn.init.orthogonal_(self.ac_device.critic_fc3.weight, gain=0.01)
        nn.init.zeros_(self.ac_device.critic_fc3.bias)
        
        # Initialize log_std to VERY SMALL exploration
        # Controls are tight - large noise breaks everything
        self.ac_device.log_std.data.fill_(-3.0)  # std = exp(-3) ≈ 0.05 (subtle!)
        
        self.ppo = PPO(self.ac_device, pi_lr, vf_lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, vf_coef, entropy_coef)
        
        self.best_eval = float('inf')
        self.pbar = trange(max_epochs, desc="Training", unit='epoch')
        self.rollout_times = []
        self.update_times = []
        
    def cleanup(self):
        self.pbar.close()


def collect_episodes(actor_critic, data_files, num_episodes, warmup=False):
    """Collect episode trajectories for PPO - Monte Carlo with GAE
    
    warmup: If True, use deterministic policy (no exploration noise)
    """
    all_future_plans = []
    all_raw_actions = []
    all_episode_costs = []
    
    for _ in range(num_episodes):
        # Random file
        data_file = random.choice(data_files)
        
        # Use controller that samples from policy
        # During warmup, be deterministic to avoid destructive exploration
        controller = HybridController(actor_critic, deterministic=warmup)
        
        # Collect trajectory
        episode_plans = []
        episode_actions = []
        
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
        
        while sim.step_idx < len(sim.data) - 1:
            # Get future plan BEFORE stepping
            state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
            future_plan_array = prepare_future_plan(futureplan, state)
            
            # Sample action ONCE
            ff_action, raw_ff = actor_critic.act(future_plan_array, deterministic=warmup)
            
            # Store for training
            episode_plans.append(future_plan_array)
            episode_actions.append(raw_ff)
            
            # Give controller the pre-sampled action to use
            controller.presampled_ff = ff_action
            
            # Step (controller will use our pre-sampled action)
            sim.step()
        
        # Get episode cost
        cost_dict = sim.compute_cost()
        episode_cost = cost_dict['total_cost']
        
        all_future_plans.append(np.stack(episode_plans))
        all_raw_actions.append(np.array(episode_actions))
        all_episode_costs.append(episode_cost)
    
    # Pad episodes to same length and create batch
    max_len = max(len(ep) for ep in all_future_plans)
    T = max_len
    N = num_episodes
    
    future_plans = np.zeros((T, N, 4, 50), dtype=np.float32)
    raw_actions = np.zeros((T, N, 1), dtype=np.float32)
    rewards = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    
    for i in range(num_episodes):
        ep_len = len(all_future_plans[i])
        future_plans[:ep_len, i] = all_future_plans[i]
        raw_actions[:ep_len, i, 0] = all_raw_actions[i]
        
        # Sparse reward: all cost at episode end (Monte Carlo)
        # GAE will handle credit assignment via value function bootstrapping
        rewards[:ep_len, i] = 0.0
        rewards[ep_len-1, i] = -all_episode_costs[i]
        dones[ep_len-1, i] = 1.0
    
    return future_plans, raw_actions, rewards, dones


def train_one_epoch(epoch, ctx):
    ctx.ac_cpu.load_state_dict(ctx.ac_device.state_dict())
    
    # Warmup: deterministic for first 10 epochs to learn without destructive noise
    # Then subtle exploration (std=0.05) to discover improvements
    warmup = epoch < 10
    
    # Collect episodes - sample fresh routes each epoch
    t0 = time.perf_counter()
    future_plans, raw_actions, rewards, dones = collect_episodes(
        ctx.ac_cpu, train_files, num_episodes=num_envs, warmup=warmup
    )
    t1 = time.perf_counter()
    ctx.rollout_times.append(t1 - t0)
    
    # PPO update
    t0 = time.perf_counter()
    ctx.ppo.update(future_plans, raw_actions, rewards, dones, debug=(epoch==0))
    t1 = time.perf_counter()
    ctx.update_times.append(t1 - t0)
    
    # Evaluate
    if epoch % eval_interval == 0:
        eval_cost = evaluate_policy(ctx.ac_cpu, eval_files, num_episodes=100, debug=(epoch==0))
        if eval_cost < ctx.best_eval:
            ctx.best_eval = eval_cost
            torch.save(ctx.ac_cpu.state_dict(), Path(__file__).parent / 'best_model.pth')
        
        if epoch % log_interval == 0:
            s = ctx.ac_device.log_std.exp().detach().cpu().numpy()
            rollout_ms = np.mean(ctx.rollout_times[-log_interval:]) * 1000
            update_ms = np.mean(ctx.update_times[-log_interval:]) * 1000
            mode = "warmup" if warmup else "explore"
            ctx.pbar.write(f"Epoch {epoch:3d}  eval={eval_cost:6.1f}  best={ctx.best_eval:6.1f}  σ={s[0]:.3f}  [{mode}]  ⏱{rollout_ms+update_ms:.0f}ms")
        
        if eval_cost < 75:
            ctx.pbar.write(f"\n{'='*60}\n🎯 BEAT PID! eval={eval_cost:.1f} < 75\n{'='*60}")
    
    ctx.pbar.update(1)
    return False


def train():
    ctx = TrainingContext()
    
    print(f"\nStarting training on {device}")
    print(f"Target: beat PID baseline (~75-85 cost)")
    
    for epoch in range(max_epochs):
        if train_one_epoch(epoch, ctx):
            break
    
    ctx.cleanup()
    
    # Training complete
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"✓ Best eval cost: {ctx.best_eval:.1f}")
    print(f"\n{'='*60}")
    print(f"To run official evaluation:")
    print(f"  1. Save best model to a controller file")
    print(f"  2. Run: python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller <your_controller>")
    print(f"  3. Compare to PID baseline (~84.85 with official eval)")
    print(f"{'='*60}")


if __name__ == '__main__':
    print(f"Experiment 039: PID + Learned Feedforward")
    print(f"Using {device} device")
    print(f"Num envs: {num_envs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    train()
