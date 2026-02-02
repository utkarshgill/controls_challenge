#!/usr/bin/env python3
"""
Experiment 001: BC with a_ego

Changes from baseline:
1. Add state.a_ego to state vector (dim 56 → 57)
2. Add 20.0 to OBS_SCALE for a_ego normalization
3. Update network input_dim to 57
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController
from multiprocessing import Pool, cpu_count, set_start_method

# ============================================================================
# EXPERIMENT 001 CHANGES
# ============================================================================

# Change 1: Add a_ego to OBS_SCALE (position 5, value 20.0)
OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 20.0, 1000.0] +  # [error, error_diff, error_integral, lataccel, v_ego, a_ego, curv]
    [1000.0] * 50,
    dtype=np.float32
)

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """
    57D state: PID terms (3) + current state (4) + 50 future curvatures
    
    CHANGE: Added state.a_ego at position 5
    """
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
    
    # CHANGE: Added state.a_ego here
    state_vec = [error, error_diff, error_integral, current_lataccel, 
                 state.v_ego, state.a_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

# Change 2: Network with 57D input (MATCH BASELINE ARCHITECTURE)
STEER_RANGE = [-2.0, 2.0]

class BCNetwork(nn.Module):
    def __init__(self, state_dim=57, action_dim=1, hidden_dim=128, trunk_layers=1, head_layers=3):
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
        # Squash to action range (CRITICAL!)
        action = torch.tanh(raw_action) * STEER_RANGE[1]
        return action

# ============================================================================
# Training Logic (unchanged from baseline)
# ============================================================================

def collect_from_file(file_path):
    """Collect (state, action) pairs from one file using PID expert"""
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    pid = PIDController()
    sim = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
    
    states, actions = [], []
    prev_error = 0.0
    error_integral = 0.0
    
    # Stop before we run out of future plan (need 50 steps)
    max_steps = len(sim.data) - 50
    for _ in range(max_steps):
        state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
        current_lataccel = sim.current_lataccel
        
        # PID action
        action = pid.update(target, current_lataccel, state, futureplan)
        
        # Build state
        error = target - current_lataccel
        error_integral = np.clip(error_integral + error, -14, 14)  # Anti-windup
        state_vec = build_state(target, current_lataccel, state, futureplan, prev_error, error_integral)
        
        states.append(state_vec)
        actions.append(action)
        
        prev_error = error
        sim.current_steer = action
        sim.step()
    
    return np.array(states), np.array(actions)

def train():
    print("\n" + "="*60)
    print("EXPERIMENT 001: BC with a_ego")
    print("="*60)
    print("State dim: 56 → 57 (added a_ego)")
    print("OBS_SCALE: added 20.0 for a_ego")
    print("="*60 + "\n")
    
    # Collect data (1000 files = ~400k samples, plenty for BC)
    print("[1/3] Collecting expert data...")
    all_files = sorted(glob.glob("../../data/*.csv"))[:1000]
    
    set_start_method("spawn", force=True)
    with Pool(min(cpu_count(), 16)) as pool:
        results = list(tqdm(pool.imap(collect_from_file, all_files), total=len(all_files)))
    
    states = np.concatenate([r[0] for r in results])
    actions = np.concatenate([r[1] for r in results])
    
    print(f"✅ Collected {len(states)} samples")
    print(f"   State shape: {states[0].shape} (should be 57)")
    print(f"   a_ego stats: mean={states[:, 5].mean():.3f}, std={states[:, 5].std():.3f}")
    
    # Normalize (MULTIPLY, not divide!)
    states = states * OBS_SCALE
    
    # Train
    print("\n[2/3] Training BC...")
    device = torch.device("cpu")
    model = BCNetwork(state_dim=57).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).float()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(30):
        total_loss = 0
        for batch_states, batch_actions in loader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
            pred = model(batch_states).squeeze(-1)  # Remove action dim
            loss = nn.MSELoss()(pred, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/30: Loss = {total_loss/len(loader):.6f}")
    
    # Save
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "results/checkpoints/bc_with_a_ego.pt")
    print("✅ Model saved")
    
    print("\n[3/3] Run evaluation:")
    print("  python evaluate.py")

if __name__ == '__main__':
    train()
