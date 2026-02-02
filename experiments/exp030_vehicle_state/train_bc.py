"""
BC with VEHICLE-CENTRIC state (like LunarLander).

LunarLander philosophy:
- State = vehicle properties (position, velocity, angle)
- NOT controller constructs (PID terms, errors)

Vehicle state for lateral control:
- current_lataccel (where am I?)
- target_lataccel (where should I be?)
- v_ego (how fast?)
- a_ego (accelerating/braking?)
- roll_lataccel (road bank)
- prev_action (momentum)

+ Future trajectory (environment):
- future_lataccels (where will I need to be?)
- future_v_egos (at what speeds?)

NO PID terms! The network learns control from vehicle state directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

device = torch.device('cpu')

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files, val_files, test_files = all_files[:15000], all_files[15000:17500], all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Normalization scales (from data statistics)
LATACCEL_SCALE = 3.0
V_SCALE = 40.0
A_SCALE = 3.0
ACTION_SCALE = 0.5

class VehicleStateNetwork(nn.Module):
    """Network operating on vehicle state, not controller constructs"""
    def __init__(self):
        super().__init__()
        
        # Process future trajectory with Conv1D
        # Input: 2 channels (future_lataccel, future_v_ego)
        self.future_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # Main network: vehicle state (6D) + future features (128D)
        self.net = nn.Sequential(
            nn.Linear(6 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, vehicle_state, future_trajectory):
        # vehicle_state: (batch, 6) - current vehicle properties
        # future_trajectory: (batch, 2, 49) - [lataccel, v_ego] over time
        
        future_features = self.future_conv(future_trajectory)  # (batch, 16, 8)
        future_flat = future_features.reshape(future_features.size(0), -1)
        
        combined = torch.cat([vehicle_state, future_flat], dim=1)
        return self.net(combined)

class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.vehicle_states = []
        self.future_trajectories = []
        self.actions = []
        self.prev_action = 0.0
    
    def update(self, target, current, state, future_plan):
        # VEHICLE STATE (what the car "knows")
        vehicle_state = np.array([
            current,              # Where am I laterally?
            target,               # Where should I be?
            state.v_ego,          # How fast am I going?
            state.a_ego,          # Am I accelerating/braking?
            state.roll_lataccel,  # Road bank effect
            self.prev_action      # What did I just do?
        ], dtype=np.float32)
        
        # FUTURE TRAJECTORY (environment/task)
        future_lataccels = []
        future_v_egos = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_lataccels.append(future_plan.lataccel[i])
                future_v_egos.append(future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego)
            else:
                future_lataccels.append(target)  # Assume constant target
                future_v_egos.append(state.v_ego)  # Assume constant speed
        
        future_traj = np.stack([future_lataccels, future_v_egos], axis=0).astype(np.float32)
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.vehicle_states.append(vehicle_state)
        self.future_trajectories.append(future_traj)
        self.actions.append(action)
        self.prev_action = action
        
        return action

def collect_data(files, num_samples):
    X_state, X_future, y = [], [], []
    
    for f in files:
        if len(X_state) >= num_samples:
            break
        
        collector = PIDDataCollector()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=collector)
        sim.rollout()
        
        X_state.extend(collector.vehicle_states)
        X_future.extend(collector.future_trajectories)
        y.extend(collector.actions)
    
    X_state = torch.FloatTensor(X_state[:num_samples])
    X_future = torch.FloatTensor(X_future[:num_samples])
    y = torch.FloatTensor(y[:num_samples]).unsqueeze(1)
    
    return X_state, X_future, y

def train():
    print("Collecting PID with VEHICLE-CENTRIC state...")
    X_state, X_future, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X_state)} samples")
    print(f"Vehicle state dim: {X_state.shape[1]}")
    print(f"Future trajectory: {X_future.shape[1]} channels × {X_future.shape[2]} steps")
    
    # Normalize vehicle state
    scales = torch.FloatTensor([LATACCEL_SCALE, LATACCEL_SCALE, V_SCALE, A_SCALE, LATACCEL_SCALE, ACTION_SCALE])
    X_state = X_state / scales
    
    # Normalize future trajectory
    X_future[:, 0, :] = X_future[:, 0, :] / LATACCEL_SCALE  # lataccel channel
    X_future[:, 1, :] = X_future[:, 1, :] / V_SCALE         # v_ego channel
    
    print("\nVehicle state stats (normalized):")
    names = ['current_lat', 'target_lat', 'v_ego', 'a_ego', 'roll_lat', 'prev_action']
    for i, name in enumerate(names):
        print(f"  {name:15s}: mean={X_state[:, i].mean():7.4f}, std={X_state[:, i].std():7.4f}")
    
    model = VehicleStateNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    batch_size = 128
    num_epochs = 50
    
    for epoch in trange(num_epochs):
        indices = torch.randperm(len(X_state))
        epoch_loss = 0
        
        for i in range(0, len(X_state), batch_size):
            batch_idx = indices[i:i+batch_size]
            state_batch = X_state[batch_idx].to(device)
            future_batch = X_future[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(state_batch, future_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss/(len(X_state)//batch_size):.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'lataccel_scale': LATACCEL_SCALE,
        'v_scale': V_SCALE,
        'a_scale': A_SCALE,
        'action_scale': ACTION_SCALE
    }, 'experiments/exp030_vehicle_state/model.pth')
    print("✅ Model saved")

if __name__ == '__main__':
    train()



