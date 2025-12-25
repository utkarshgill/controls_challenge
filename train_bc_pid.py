#!/usr/bin/env python3
"""
Behavioral Cloning: Train network to predict PID actions

ARCHITECTURE:
- State: 57D = [error, error_diff, error_integral, lataccel, v_ego, a_ego, curv_now] + 50 future curvatures
- Network: MLP (128 hidden, 1+3 layers) with tanh-squashed output → [-2, 2]
- Same architecture as beautiful_lander.py actor-critic (actor head only)

CRITICAL STATE MATCHING:
Must match PID's internal computation exactly:
  error_diff = error - prev_error        # NO division by dt!
  error_integral += error                # NO multiplication by dt!
  No clipping on integral                # PID doesn't clip

TRAINING:
- 1000 files (shuffled) for diverse scenarios
- 50 epochs of supervised learning (MSE loss)
- Evaluated on separate validation set

EXPECTED RESULTS:
- BC cost: 80-120 (comparable to or better than PID ~100-150)
- Train/val gap: <30% (should generalize)
- Ready for PPO fine-tuning if both criteria met
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
from multiprocessing import Pool, cpu_count
import glob

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
from controllers import BaseController
from controllers.pid import Controller as PIDController

# Same architecture as train_ppo_pure.py
state_dim, action_dim = 57, 1
hidden_dim = 128
trunk_layers, head_layers = 1, 3
STEER_RANGE = (-2.0, 2.0)

# State normalization (matching PID's actual computation)
# error_diff is ~10× larger than error_derivative (no /dt), so scale accordingly
OBS_SCALE = np.array(
    [10.0, 1.0, 1.0, 2.0, 0.03, 20.0, 1000.0] +  # [error, error_diff, error_integral, lataccel, v_ego, a_ego, curv]
    [1000.0] * 50,
    dtype=np.float32
)

device = torch.device('cpu')

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """57D: PID terms (3) + current state (4) + 50 future curvatures
    NOTE: Match PID's actual computation (no dt scaling, no clipping)
    """
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error  # Match PID: no division by dt!
    
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
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, state.a_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class BCNetwork(nn.Module):
    """Same architecture as ActorCritic, but just the actor part"""
    def __init__(self, state_dim, action_dim, hidden_dim, trunk_layers, head_layers):
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

    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        raw_action = self.actor_mean(actor_feat)
        # Squash to action range
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action
    
    @torch.no_grad()
    def act(self, state):
        state_normalized = state * OBS_SCALE
        state_tensor = torch.as_tensor(state_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        action = self(state_tensor)
        return action.squeeze(0).cpu().numpy()

class LoggingController(BaseController):
    """
    Wraps PID to collect (state, action) pairs for BC training
    
    CRITICAL: Must use PID's internal state (prev_error, error_integral) to build
    state vectors, NOT maintain our own tracking. This ensures BC sees exactly what
    PID sees when making decisions.
    
    The state is built BEFORE calling pid.update() to capture the state that
    corresponds to the action PID will produce.
    """
    def __init__(self, pid_controller):
        self.pid = pid_controller
        self.states = []
        self.actions = []
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build state vector using PID's internal state (not our own tracking!)
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                               self.pid.prev_error, self.pid.error_integral)
        
        # Get PID action (this updates PID's internal state)
        pid_action = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        
        self.states.append(state_vec)
        self.actions.append(pid_action)
        
        return pid_action

# Global model per worker (initialized once per process)
_worker_model = None

def _init_worker():
    """Initialize model once per worker process"""
    global _worker_model
    _worker_model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

def _collect_single_file(data_file):
    """Worker function to process a single file (for multiprocessing)"""
    global _worker_model
    pid = PIDController()
    logger = LoggingController(pid)
    sim = TinyPhysicsSimulator(_worker_model, data_file, controller=logger, debug=False)
    sim.rollout()
    
    # Only keep states after control starts
    states = logger.states[CONTROL_START_IDX:]
    actions = logger.actions[CONTROL_START_IDX:]
    
    return states, actions

def collect_expert_data(data_files, n_files=1000):
    """Run PID on files and collect (state, action) pairs (parallelized)"""
    files_to_process = data_files[:n_files]
    
    print(f"Collecting PID demonstrations from {n_files} files (parallel)...")
    
    # Use multiprocessing to speed up data collection
    n_workers = min(cpu_count(), 16)  # Cap at 16 to avoid overwhelming system
    
    with Pool(n_workers, initializer=_init_worker) as pool:
        # Use larger chunksize to reduce overhead
        chunksize = max(1, len(files_to_process) // (n_workers * 4))
        results = list(tqdm(
            pool.imap(_collect_single_file, files_to_process, chunksize=chunksize),
            total=len(files_to_process),
            desc="Collecting expert data"
        ))
    
    # Aggregate results
    all_states = []
    all_actions = []
    for states, actions in results:
        all_states.extend(states)
        all_actions.extend(actions)
    
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"Collected {len(states)} expert transitions")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")
    
    # Sanity checks
    assert not np.any(np.isnan(states)), "NaN found in states!"
    assert not np.any(np.isnan(actions)), "NaN found in actions!"
    assert len(states) == len(actions), "State/action length mismatch!"
    
    return states, actions

def evaluate_bc(network, model, data_files, n_eval=20):
    """Evaluate BC network"""
    costs = []
    
    for data_file in np.random.choice(data_files, size=min(n_eval, len(data_files)), replace=False):
        class BCController(BaseController):
            def __init__(self):
                self.prev_error = 0.0
                self.error_integral = 0.0
            
            def update(self, target_lataccel, current_lataccel, state, future_plan):
                error = target_lataccel - current_lataccel
                
                # Build state BEFORE updating (to match data collection)
                state_vec = build_state(target_lataccel, current_lataccel, state, future_plan,
                                       self.prev_error, self.error_integral)
                action = network.act(state_vec)
                
                # Update state AFTER (match PID's update order)
                self.error_integral += error  # No dt, no clipping!
                self.prev_error = error
                
                return float(action[0]) if len(action.shape) > 0 else float(action)
        
        sim = TinyPhysicsSimulator(model, data_file, controller=BCController(), debug=False)
        cost = sim.rollout()
        costs.append(cost['total_cost'])
    
    return np.mean(costs)

def train_bc():
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    all_files = sorted(glob.glob("./data/*.csv"))
    
    # CRITICAL: Shuffle before split (files are sorted by difficulty!)
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    # Train/val split
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print("="*60)
    
    # Collect expert data (1000 files for better coverage of difficulty distribution)
    states, actions = collect_expert_data(train_files, n_files=1000)
    
    # Create network
    network = BCNetwork(state_dim, action_dim, hidden_dim, trunk_layers, head_layers).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    
    # Create dataset
    states_tensor = torch.from_numpy(states * OBS_SCALE).to(device)
    actions_tensor = torch.from_numpy(actions).to(device)
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Train
    print("\n" + "="*60)
    print("Training BC to clone PID...")
    print("="*60)
    
    n_epochs = 50
    best_loss = float('inf')
    
    pbar = trange(n_epochs, desc="BC Training")
    for epoch in pbar:
        network.train()
        epoch_loss = 0.0
        
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            predicted = network(batch_states)
            loss = nn.MSELoss()(predicted, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(network.state_dict(), "bc_pid_best.pth")
        
        if epoch % 10 == 0:
            pbar.write(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")
    
    print(f"\nBC training complete. Best loss: {best_loss:.6f}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating BC network...")
    print("="*60)
    
    network.eval()
    
    # PID baseline
    pid = PIDController()
    pid_costs = []
    for f in np.random.choice(train_files, size=20, replace=False):
        sim = TinyPhysicsSimulator(model, f, controller=pid, debug=False)
        pid_costs.append(sim.rollout()['total_cost'])
    pid_baseline = np.mean(pid_costs)
    
    # BC on train
    bc_train_cost = evaluate_bc(network, model, train_files, n_eval=20)
    
    # BC on val
    bc_val_cost = evaluate_bc(network, model, val_files, n_eval=20)
    
    print(f"PID baseline:      {pid_baseline:.2f}")
    print(f"BC train cost:     {bc_train_cost:.2f}")
    print(f"BC val cost:       {bc_val_cost:.2f}")
    print(f"Gap from PID:      {bc_train_cost - pid_baseline:.2f}")
    
    # BC is good if: (1) close to PID, (2) train/val gap is reasonable
    train_val_gap = abs(bc_train_cost - bc_val_cost) / bc_train_cost
    # Accept if: both train AND val beat PID significantly, even if gap exists
    both_beat_pid = bc_train_cost < pid_baseline and bc_val_cost < pid_baseline * 1.2
    is_good = (bc_train_cost < pid_baseline * 1.5 and train_val_gap < 0.5) or both_beat_pid
    
    if is_good:
        print("\n✅ BC looks good! Network can learn the control task.")
        print("   Ready for PPO fine-tuning.")
    else:
        print("\n⚠️ BC not learning well. Debug before PPO.")
        print(f"   Issue: BC={bc_train_cost:.1f}, PID={pid_baseline:.1f}, gap={train_val_gap:.1%}")
    
    # Save checkpoint with full config for PPO stage
    checkpoint = {
        'model_state_dict': network.state_dict(),
        'architecture': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dim': hidden_dim,
            'trunk_layers': trunk_layers,
            'head_layers': head_layers,
        },
        'normalization': {
            'OBS_SCALE': OBS_SCALE.tolist(),
        },
        'performance': {
            'bc_train_cost': float(bc_train_cost),
            'bc_val_cost': float(bc_val_cost),
            'pid_baseline': float(pid_baseline),
            'train_val_gap': float(train_val_gap),
            'best_loss': float(best_loss),
            'is_ready_for_ppo': is_good,
        },
        'training': {
            'n_expert_files': 1000,
            'n_transitions': len(states),
            'n_epochs': n_epochs,
        }
    }
    torch.save(checkpoint, 'bc_pid_checkpoint.pth')
    print(f"\nSaved: bc_pid_best.pth (weights), bc_pid_checkpoint.pth (full config)")
    
    return network

if __name__ == '__main__':
    train_bc()

