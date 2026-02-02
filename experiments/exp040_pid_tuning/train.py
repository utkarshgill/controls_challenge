"""
Experiment 040: BC + PPO Fine-tuning of Single Neuron

Phase 1: Behavioral clone PID with 1 neuron (3 weights, no bias)
Phase 2: Fine-tune with PPO to beat PID on actual cost function

Hypothesis: PPO can discover better gains than hand-tuned PID
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

# PID constants (what we're trying to beat)
PID_P = 0.195
PID_I = 0.100
PID_D = -0.053

# Training config
num_envs = int(os.getenv('NUM_ENVS', 8))
max_epochs_bc = 100  # SGD converges fast
max_epochs_ppo = 200
log_interval = 5
eval_interval = 10

# PPO hyperparameters (from beautiful_lander)
batch_size = 2048
K_epochs = 5
pi_lr = 1e-4  # Small LR for fine-tuning
vf_lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
vf_coef = 0.5
entropy_coef = 0.01

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

# Set seeds FIRST before any data operations
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load data
# Use FIRST 100 files for eval (same as official tinyphysics.py --num_segs 100)
all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]  # Official eval files (00000.csv to 00099.csv)

# For training, use random subset excluding eval files
all_files = list(DATA_PATH.glob('*.csv'))
random.shuffle(all_files)
train_files = [f for f in all_files if f not in eval_files][:15000]
test_files = all_files_sorted[100:200]  # Next 100 for test

# Load simulator (after seeds are set)
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)


class OneNeuronActor(nn.Module):
    """Single neuron: [error, error_integral, error_diff] → action"""
    def __init__(self):
        super(OneNeuronActor, self).__init__()
        self.linear = nn.Linear(3, 1, bias=False)
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state):
        action_mean = self.linear(state)
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
            action = mean  # No tanh - PID is linear
        else:
            action = torch.distributions.Normal(mean, std).sample()
        return action.cpu().numpy()[0, 0]


class Critic(nn.Module):
    """Value function: same input as actor"""
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class OneNeuronController(BaseController):
    """Controller that uses the one-neuron network"""
    def __init__(self, actor, deterministic=True):
        self.actor = actor
        self.deterministic = deterministic
        self.error_integral = 0
        self.prev_error = 0
    
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # State for network: [error, error_integral, error_diff]
        network_state = np.array([error, self.error_integral, error_diff], dtype=np.float32)
        
        action = self.actor.act(network_state, deterministic=self.deterministic)
        return action


def collect_pid_demonstrations(data_files, num_files=1000, use_cache=True):
    """Phase 1: Collect PID demonstrations for BC (with caching)"""
    cache_file = Path(__file__).parent / f"pid_demonstrations_{num_files}.pkl"
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"\n{'='*60}")
        print("Loading PID Demonstrations from Cache")
        print(f"{'='*60}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        states, actions = data['states'], data['actions']
        print(f"✓ Loaded {len(states):,} samples from {cache_file.name}")
        
        # VERIFY cached data
        print("\nVerifying cached data (first 5 samples):")
        for i in range(min(5, len(states))):
            error, integral, diff = states[i]
            action = actions[i, 0]
            expected = PID_P * error + PID_I * integral + PID_D * diff
            print(f"  Sample {i}: action={action:.6f}, expected={expected:.6f}, diff={abs(action-expected):.6f}")
        
        return states, actions
    
    # Otherwise collect fresh
    print(f"\n{'='*60}")
    print("Phase 1: Collecting PID Demonstrations")
    print(f"{'='*60}")
    
    from controllers.pid import Controller as PIDController
    
    states = []
    actions = []
    
    debug_first_file = True
    
    for data_file in tqdm(data_files[:num_files], desc="Collecting"):
        controller = PIDController()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller)
        
        original_update = controller.update
        step_count = [0]  # Mutable counter
        
        def capture(target_lataccel, current_lataccel, state, future_plan):
            # Save OLD state BEFORE PID updates it
            old_error_integral = controller.error_integral
            old_prev_error = controller.prev_error
            
            # Compute what PID will compute
            error = target_lataccel - current_lataccel
            
            # NEW state that PID uses internally
            new_error_integral = old_error_integral + error
            error_diff = error - old_prev_error
            
            state_vec = [error, new_error_integral, error_diff]
            
            # Call PID (updates its internal state)
            action = original_update(target_lataccel, current_lataccel, state, future_plan)
            
            # Debug first 3 steps of first file
            if debug_first_file and step_count[0] < 3:
                expected = PID_P * error + PID_I * new_error_integral + PID_D * error_diff
                print(f"\n  Step {step_count[0]}: error={error:.6f}, integral={new_error_integral:.6f}, diff={error_diff:.6f}")
                print(f"    PID action={action:.6f}, expected={expected:.6f}, match={abs(action-expected)<1e-6}")
            step_count[0] += 1
            
            states.append(state_vec)
            actions.append(action)
            return action
        
        controller.update = capture
        sim.rollout()
        debug_first_file = False
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"✓ Collected {len(states):,} samples")
    
    # VERIFY: Check if data matches PID equation
    print("\nVerifying data (first 5 samples):")
    for i in range(min(5, len(states))):
        error, integral, diff = states[i]
        action = actions[i, 0]
        expected = PID_P * error + PID_I * integral + PID_D * diff
        print(f"  Sample {i}: action={action:.6f}, expected={expected:.6f}, diff={abs(action-expected):.6f}")
    
    # Save to cache
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump({'states': states, 'actions': actions}, f)
        print(f"✓ Saved to cache: {cache_file.name}")
    
    return states, actions


def train_bc(actor, states, actions):
    """Phase 1: Train with BC to recover PID"""
    print(f"\n{'='*60}")
    print("Phase 1: Behavioral Cloning (Recover PID)")
    print(f"{'='*60}")
    
    # Final verification before training
    print("\nFinal verification (3 random samples):")
    for i in [0, len(states)//2, len(states)-1]:
        error, integral, diff = states[i]
        action = actions[i, 0]
        expected = PID_P * error + PID_I * integral + PID_D * diff
        print(f"  idx={i}: action={action:.6f}, expected={expected:.6f}, match={abs(action-expected)<1e-5}")
    
    X = torch.FloatTensor(states)
    y = torch.FloatTensor(actions)
    
    # Train with ADAM (works perfectly for linear problems with unbalanced features)
    print("\n  Training with Adam (lr=0.01, 500 epochs)...")
    optimizer = optim.Adam(actor.linear.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        predictions = actor.linear(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            weights = actor.linear.weight.data.numpy()[0]
            print(f"  Epoch {epoch+1:3d}/500 | Loss: {loss.item():.8f} | P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")
    
    final_loss = loss.item()
    print(f"\n  Final BC Loss: {final_loss:.10f}")
    
    for epoch in range(0):  # Skip loop
        optimizer.zero_grad()
        # Use actor.linear directly, not actor() which also returns log_std
        predictions = actor.linear(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) == 1:
            # Debug first epoch
            weights = actor.linear.weight.data.numpy()[0]
            print(f"\nEpoch 1 DEBUG:")
            print(f"  Weight tensor shape: {actor.linear.weight.shape}")
            print(f"  Weight tensor: {actor.linear.weight.data}")
            print(f"  Weights: P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")
            print(f"  First 3 predictions: {predictions[:3, 0].detach().numpy()}")
            print(f"  First 3 targets: {y[:3, 0].numpy()}")
            
            # Direct computation
            direct = actor.linear(X[:3])
            print(f"  Direct actor.linear(X[:3]): {direct[:, 0].detach().numpy()}")
            
            print(f"  First 3 inputs:")
            for i in range(3):
                e, integ, diff = X[i].numpy()
                pred = predictions[i, 0].item()
                target = y[i, 0].item()
                manual = weights[0]*e + weights[1]*integ + weights[2]*diff
                print(f"    [{e:.6f}, {integ:.6f}, {diff:.6f}] → pred={pred:.6f}, target={target:.6f}, manual={manual:.6f}")
        
        if (epoch + 1) % 10 == 0:
            weights = actor.linear.weight.data.numpy()[0]
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f} | "
                  f"P={weights[0]:+.4f} I={weights[1]:+.4f} D={weights[2]:+.4f}")
    
    weights = actor.linear.weight.data.numpy()[0]
    print(f"\n✓ BC Complete!")
    print(f"  Learned: P={weights[0]:.4f}, I={weights[1]:.4f}, D={weights[2]:.4f}")
    print(f"  PID:     P={PID_P:.4f}, I={PID_I:.4f}, D={PID_D:.4f}")
    print(f"  Error:   {abs(weights[0]-PID_P)+abs(weights[1]-PID_I)+abs(weights[2]-PID_D):.6f}")
    
    # Verify D coefficient has correct sign (most critical)
    if np.sign(weights[2]) != np.sign(PID_D):
        print(f"  ⚠️  WARNING: D coefficient has WRONG SIGN!")
        print(f"  BC failed to converge properly. Consider rerunning.")


def rollout_episode(data_file, controller):
    """Run episode and return cost"""
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
    cost = sim.rollout()
    return cost


def evaluate_policy(actor, data_files, num_episodes=16, use_multiprocessing=False):
    """Evaluate current policy
    
    Args:
        actor: The neural network actor
        data_files: List of data files to evaluate on
        num_episodes: Number of episodes to evaluate
        use_multiprocessing: If True, use official multiprocessing eval (slower but matches official metrics)
    """
    if use_multiprocessing:
        # Use official multiprocessing evaluation (matches tinyphysics.py --num_segs)
        # Write temp controller to controllers/ directory so it can be imported
        temp_controller_path = Path(__file__).parent.parent.parent / 'controllers' / 'temp_eval_controller.py'
        weights = actor.linear.weight.data.cpu().numpy().flatten()
        
        # Write temporary controller file
        controller_code = f'''
from controllers import BaseController
class Controller(BaseController):
    def __init__(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.p = {weights[0]:.10f}
        self.i = {weights[1]:.10f}
        self.d = {weights[2]:.10f}
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        return action
    
    def reset(self):
        self.error_integral = 0.0
        self.prev_error = 0.0
'''
        temp_controller_path.write_text(controller_code)
        
        # Clear any cached bytecode and verify file exists
        import shutil
        cache_dir = temp_controller_path.parent / '__pycache__'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        if not temp_controller_path.exists():
            raise FileNotFoundError(f"Failed to write {temp_controller_path}")
        
        # Small delay to ensure filesystem sync
        import time as time_module
        time_module.sleep(0.1)
        
        try:
            # Use official run_rollout with multiprocessing
            run_rollout_partial = partial(run_rollout, controller_type='temp_eval_controller', 
                                          model_path=str(MODEL_PATH), debug=False)
            files_to_eval = data_files[:num_episodes]
            results = process_map(run_rollout_partial, files_to_eval, max_workers=16, chunksize=10, disable=True)
            costs = [result[0]['total_cost'] for result in results]
            return float(np.mean(costs))
        finally:
            # Cleanup temp controller file and cache
            if temp_controller_path.exists():
                temp_controller_path.unlink()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
    else:
        # Fast sequential evaluation for training
        controller = OneNeuronController(actor, deterministic=True)
        costs = []
        for i in range(min(num_episodes, len(data_files))):
            controller.reset()
            cost_dict = rollout_episode(data_files[i], controller)
            costs.append(cost_dict['total_cost'])
        return float(np.mean(costs))


def collect_episodes(actor, data_files, num_episodes):
    """Phase 2: Collect episodes with current policy for PPO"""
    all_states = []
    all_actions = []
    all_costs = []
    
    for _ in range(num_episodes):
        data_file = random.choice(data_files)
        controller = OneNeuronController(actor, deterministic=False)
        
        episode_states = []
        episode_actions = []
        
        sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
        
        while sim.step_idx < len(sim.data) - 1:
            # Capture state before step
            _, target, _ = sim.get_state_target_futureplan(sim.step_idx)
            error = target - sim.current_lataccel
            state_vec = np.array([error, controller.error_integral, error - controller.prev_error], dtype=np.float32)
            
            episode_states.append(state_vec)
            
            # Step (controller will sample)
            sim.step()
            
            # Store action that was taken (from controller's internal state after step)
            # We approximate by sampling again (slightly suboptimal but OK)
            with torch.no_grad():
                state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                mean, std = actor(state_t)
                action_dist = torch.distributions.Normal(mean, std)
                raw_action = action_dist.sample()
                episode_actions.append(raw_action.numpy()[0, 0])
        
        cost_dict = sim.compute_cost()
        all_states.append(np.stack(episode_states))
        all_actions.append(np.array(episode_actions))
        all_costs.append(cost_dict['total_cost'])
    
    # Pad and batch
    max_len = max(len(ep) for ep in all_states)
    T, N = max_len, num_episodes
    
    states = np.zeros((T, N, 3), dtype=np.float32)
    actions = np.zeros((T, N, 1), dtype=np.float32)
    rewards = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    
    for i in range(num_episodes):
        ep_len = len(all_states[i])
        states[:ep_len, i] = all_states[i]
        actions[:ep_len, i, 0] = all_actions[i]
        # Sparse reward: all cost at episode end (Monte Carlo)
        rewards[:ep_len, i] = 0.0
        rewards[ep_len-1, i] = -all_costs[i]
        dones[ep_len-1, i] = 1.0
    
    return states, actions, rewards, dones


class PPO:
    """PPO for fine-tuning"""
    def __init__(self, actor, critic, pi_lr, vf_lr):
        self.actor = actor
        self.critic = critic
        self.pi_optimizer = optim.Adam(list(actor.parameters()), lr=pi_lr)
        self.vf_optimizer = optim.Adam(critic.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.lamda = gae_lambda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
    
    def compute_advantages(self, rewards, state_values, is_terminals):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=rewards.device)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t+1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_log_prob(self, action, dist):
        # No tanh for PID - just raw Gaussian log prob
        return dist.log_prob(action).sum(-1)
    
    def update(self, states, actions, rewards, dones):
        dev = next(self.actor.parameters()).device
        T, N = rewards.shape
        B = T * N
        
        states_t = torch.from_numpy(states).to(device=dev, dtype=torch.float32).reshape(B, 3)
        actions_t = torch.from_numpy(actions).to(device=dev, dtype=torch.float32).reshape(B, 1)
        rewards_t = torch.from_numpy(rewards).to(device=dev, dtype=torch.float32)
        dones_t = torch.from_numpy(dones).to(device=dev, dtype=torch.float32)
        
        with torch.no_grad():
            mean, std = self.actor(states_t)
            dist = torch.distributions.Normal(mean, std)
            old_logprobs = self.compute_log_prob(actions_t, dist)
            old_values = self.critic(states_t).squeeze(-1).view(T, N)
            advantages, returns = self.compute_advantages(rewards_t, old_values, dones_t)
        
        for _ in range(self.K_epochs):
            perm = torch.randperm(B, device=dev)
            for start in range(0, B, self.batch_size):
                idx = perm[start:start + self.batch_size]
                
                mean, std = self.actor(states_t[idx])
                values = self.critic(states_t[idx]).squeeze(-1)
                
                dist = torch.distributions.Normal(mean, std)
                logprobs = self.compute_log_prob(actions_t[idx], dist)
                
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(values, returns[idx])
                
                entropy = dist.entropy().mean()
                
                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                
                self.pi_optimizer.zero_grad()
                self.vf_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.pi_optimizer.step()
                self.vf_optimizer.step()


def train():
    print("="*60)
    print("Experiment 040: BC + PPO for PID Tuning")
    print("="*60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: BC
    states, actions = collect_pid_demonstrations(train_files, num_files=1000)
    actor = OneNeuronActor().to(device)
    train_bc(actor, states, actions)
    
    # Evaluate BC with official multiprocessing (like tinyphysics.py)
    print(f"\n⏳ Evaluating BC with official metrics (multiprocessing on {len(eval_files)} files)...")
    bc_cost_official = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=True)
    
    # Verify PID baseline with official multiprocessing
    print(f"⏳ Evaluating PID baseline with official metrics...")
    run_rollout_partial = partial(run_rollout, controller_type='pid', model_path=str(MODEL_PATH), debug=False)
    pid_results = process_map(run_rollout_partial, eval_files[:100], max_workers=16, chunksize=10, disable=True)
    pid_baseline_official = np.mean([r[0]['total_cost'] for r in pid_results])
    
    print(f"\n✓ BC Evaluation (official): {bc_cost_official:.2f}")
    print(f"✓ PID Baseline (official): {pid_baseline_official:.2f}")
    print(f"✓ BC vs PID error: {abs(bc_cost_official - pid_baseline_official):.2f}")
    
    # Phase 2: PPO
    print(f"\n{'='*60}")
    print("Phase 2: PPO Fine-Tuning")
    print(f"{'='*60}")
    
    critic = Critic().to(device)
    ppo = PPO(actor, critic, pi_lr, vf_lr)
    
    # Initialize best_cost with BC
    best_cost = bc_cost_official
    print(f"Starting PPO from cost: {best_cost:.2f} (official eval)")
    
    pbar = trange(max_epochs_ppo, desc="PPO", unit="epoch")
    
    for epoch in range(max_epochs_ppo):
        # Collect episodes
        states_batch, actions_batch, rewards_batch, dones_batch = collect_episodes(actor, train_files, num_envs)
        
        # PPO update
        ppo.update(states_batch, actions_batch, rewards_batch, dones_batch)
        
        # Evaluate (fast sequential for training speed)
        if epoch % eval_interval == 0:
            eval_cost = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=False)
            if eval_cost < best_cost:
                best_cost = eval_cost
                torch.save(actor.state_dict(), Path(__file__).parent / 'best_model.pth')
            
            if epoch % log_interval == 0:
                weights = actor.linear.weight.data.cpu().numpy()[0]
                pbar.write(f"Epoch {epoch:3d}  eval={eval_cost:6.1f}  best={best_cost:6.1f}  "
                          f"P={weights[0]:+.3f} I={weights[1]:+.3f} D={weights[2]:+.3f}")
            
            if eval_cost < pid_baseline_official:
                pbar.write(f"\n{'='*60}\n🎯 BEAT PID! cost={eval_cost:.2f} < {pid_baseline_official:.2f}\n{'='*60}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Load best model for final evaluation
    best_model_path = Path(__file__).parent / 'best_model.pth'
    if best_model_path.exists():
        actor.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\n✓ Loaded best model from {best_model_path}")
    
    # Final evaluation with official multiprocessing
    print(f"\n{'='*60}")
    print(f"Final Evaluation (official multiprocessing metrics)")
    print(f"{'='*60}")
    final_cost_official = evaluate_policy(actor, eval_files, num_episodes=100, use_multiprocessing=True)
    
    print(f"\n✓ BC cost (official): {bc_cost_official:.2f}")
    print(f"✓ Final PPO cost (official): {final_cost_official:.2f}")
    print(f"✓ PID baseline (official): {pid_baseline_official:.2f}")
    print(f"✓ Improvement over PID: {pid_baseline_official - final_cost_official:.2f} points")
    
    final_weights = actor.linear.weight.data.cpu().numpy()[0]
    print(f"\n✓ Final weights: P={final_weights[0]:.6f}, I={final_weights[1]:.6f}, D={final_weights[2]:.6f}")
    print(f"✓ PID weights:   P=0.195000, I=0.100000, D=-0.053000")
    print(f"{'='*60}")


if __name__ == '__main__':
    print(f"Using {device}")
    train()

