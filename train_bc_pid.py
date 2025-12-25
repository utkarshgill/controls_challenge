#!/usr/bin/env python3
"""
Behavioral Cloning: Train network to predict PID actions
Goal: Validate network architecture can learn the control task before adding PPO
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
    """Wraps PID to collect (state, action) pairs - MUST match PID's internal state exactly!"""
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
    
    # Train/val split
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print("="*60)
    
    # Collect expert data (500 files = ~200k transitions, plenty for BC)
    states, actions = collect_expert_data(train_files, n_files=500)
    
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
    
    if bc_train_cost < 100:
        print("\n✅ BC looks good! Network can learn the control task.")
        print("   Ready for PPO fine-tuning.")
    else:
        print("\n⚠️ BC not learning well. Debug before PPO.")
    
    return network

if __name__ == '__main__':
    train_bc()

