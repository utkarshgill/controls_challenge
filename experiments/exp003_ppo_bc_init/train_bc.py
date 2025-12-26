#!/usr/bin/env python3
"""
Experiment 003: Clean BC from PID
State: 55D = [error, roll_lataccel, v_ego, a_ego, current_lataccel, future_lataccel×50]
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, STEER_RANGE
from controllers.pid import Controller as PIDController

# Hyperparameters
STATE_DIM = 55
ACTION_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 3
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
NUM_FILES = 1000

# Normalization constants (from data statistics)
NORM_SCALE = np.array([
    2.0,   # error (-2 to 2)
    5.0,   # roll_lataccel (-5 to 5)
    0.05,  # v_ego (0 to 40 m/s)
    5.0,   # a_ego (-5 to 5)
    2.0,   # current_lataccel (-2 to 2)
    *[2.0]*50  # future_lataccel (-2 to 2 each)
], dtype=np.float32)

def build_state(target_lataccel, current_lataccel, state, future_plan):
    """Build 55-dim state vector with normalization"""
    error = target_lataccel - current_lataccel
    
    # Pad future plan if needed
    future_lataccel = np.array(future_plan.lataccel)
    if len(future_lataccel) == 0:
        future_lataccel = np.zeros(50, dtype=np.float32)
    elif len(future_lataccel) < 50:
        future_lataccel = np.pad(future_lataccel, (0, 50 - len(future_lataccel)), 'edge')
    else:
        future_lataccel = future_lataccel[:50]
    
    state_vec = np.array([
        error,
        state.roll_lataccel,
        state.v_ego,
        state.a_ego,
        current_lataccel,
        *future_lataccel
    ], dtype=np.float32)
    
    # Normalize
    state_vec = state_vec * NORM_SCALE
    
    assert state_vec.shape == (55,), f"State shape mismatch: {state_vec.shape}"
    return state_vec

class BCNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1], scale to [-2, 2]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x) * STEER_RANGE[1]  # Scale to [-2, 2]

def collect_data_from_file(file_path, model_path):
    """Collect state-action pairs from one file using PID"""
    model = TinyPhysicsModel(model_path, debug=False)
    pid = PIDController()
    sim = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
    
    states = []
    actions = []
    
    max_steps = len(sim.data) - 55  # Leave room for future plan
    
    for _ in range(max_steps):
        if sim.step_idx >= len(sim.data) - 50:
            break
        
        state, target, future_plan = sim.get_state_target_futureplan(sim.step_idx)
        current_lataccel = sim.current_lataccel
        
        # Build state vector
        state_vec = build_state(target, current_lataccel, state, future_plan)
        
        # Get PID action
        action = pid.update(target, current_lataccel, state, future_plan)
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        
        states.append(state_vec)
        actions.append(action)
        
        sim.step()
    
    return np.array(states), np.array(actions)

def main():
    print("\n" + "="*60)
    print("Experiment 003: BC from PID")
    print("="*60)
    print(f"State: 55D [error, roll, v_ego, a_ego, current, future×50]")
    print(f"Action: 1D steer in [-2, 2]")
    print(f"Network: {NUM_LAYERS} layers × {HIDDEN_DIM} hidden")
    print(f"Files: {NUM_FILES}, Epochs: {NUM_EPOCHS}")
    print("="*60 + "\n")
    
    # Collect data
    model_path = "../../models/tinyphysics.onnx"
    all_files = sorted(glob.glob("../../data/*.csv"))[:NUM_FILES]
    
    print(f"[1/3] Collecting data from {len(all_files)} files...")
    all_states = []
    all_actions = []
    
    for file_path in tqdm(all_files, desc="Collecting"):
        try:
            states, actions = collect_data_from_file(file_path, model_path)
            all_states.append(states)
            all_actions.append(actions)
        except Exception as e:
            print(f"⚠️  Skipped {file_path}: {e}")
            continue
    
    X = np.vstack(all_states)
    y = np.hstack(all_actions)
    
    print(f"✅ Collected {len(X):,} samples")
    print(f"   State shape: {X.shape}")
    print(f"   Action shape: {y.shape}")
    print(f"   State range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Action range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create dataset
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create network
    print(f"\n[2/3] Training BC network...")
    network = BCNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_LAYERS)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            pred = network(batch_states)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(network.state_dict(), 'results/checkpoints/bc_best.pth')
        
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    
    # Save final
    torch.save({
        'state_dict': network.state_dict(),
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
    }, 'results/checkpoints/bc_final.pth')
    
    print(f"\n[3/3] Creating controller...")
    print(f"✅ Saved checkpoints to results/checkpoints/")
    print(f"✅ Next: Create controllers/bc.py and evaluate")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

