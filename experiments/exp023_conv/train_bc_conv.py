"""
BC with 1D convolutions on future curvature sequence
Learn trajectory patterns, not just individual points
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

device = torch.device('cpu')

# Data split
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
train_files, val_files, test_files = all_files[:15000], all_files[15000:17500], all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# Compute OBS_SCALE for base features (PID state + v_ego)
BASE_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333], dtype=np.float32)
CURV_SCALE = 0.1573  # Curvature scale

class ConvBCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1D Conv on future curvatures (49 points)
        # Input: (batch, 1, 49) - treat as 1-channel sequence
        # Extract temporal patterns with multiple conv layers
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # 49 → 49
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),  # 49 → 49
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # 49 → 49
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # 49 → 8 (compress to fixed size)
        )
        
        # MLP on combined features
        # Input: 4 (PID state) + 16*8 (conv features) = 132
        self.mlp = nn.Sequential(
            nn.Linear(4 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, curv_sequence):
        # base_features: (batch, 4)
        # curv_sequence: (batch, 49)
        
        # Conv expects (batch, channels, length)
        curv_input = curv_sequence.unsqueeze(1)  # (batch, 1, 49)
        conv_out = self.conv(curv_input)  # (batch, 16, 8)
        conv_flat = conv_out.reshape(conv_out.size(0), -1)  # (batch, 128)
        
        # Concatenate
        combined = torch.cat([base_features, conv_flat], dim=1)
        
        return self.mlp(combined)

class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.base_states = []
        self.curv_sequences = []
        self.actions = []
    
    def update(self, target, current, state, future_plan):
        error = target - current
        
        # Base features (PID state)
        base = np.array([
            error,
            self.pid.error_integral + error,
            error - self.pid.prev_error,
            state.v_ego
        ], dtype=np.float32)
        
        # Future curvature sequence
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                curv = (lat - state.roll_lataccel) / max(state.v_ego ** 2, 1.0)
                curvs.append(curv)
            else:
                curvs.append(0.0)
        curv_seq = np.array(curvs, dtype=np.float32)
        
        # Get PID action
        action = self.pid.update(target, current, state, future_plan)
        
        self.base_states.append(base)
        self.curv_sequences.append(curv_seq)
        self.actions.append(action)
        
        return action

def collect_data(files, num_samples):
    X_base, X_curv, y = [], [], []
    
    for f in files:
        if len(X_base) >= num_samples:
            break
        
        collector = PIDDataCollector()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=collector)
        sim.rollout()
        
        X_base.extend(collector.base_states)
        X_curv.extend(collector.curv_sequences)
        y.extend(collector.actions)
    
    X_base = torch.FloatTensor(X_base[:num_samples])
    X_curv = torch.FloatTensor(X_curv[:num_samples])
    y = torch.FloatTensor(y[:num_samples]).unsqueeze(1)
    
    return X_base, X_curv, y

def train():
    print("Collecting PID demonstrations...")
    X_base, X_curv, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X_base)} samples")
    
    # Normalize
    X_base = X_base / torch.FloatTensor(BASE_SCALE)
    X_curv = X_curv / CURV_SCALE
    
    # Train
    model = ConvBCNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    batch_size = 128
    num_epochs = 50
    
    for epoch in trange(num_epochs):
        indices = torch.randperm(len(X_base))
        epoch_loss = 0
        
        for i in range(0, len(X_base), batch_size):
            batch_idx = indices[i:i+batch_size]
            base_batch = X_base[batch_idx].to(device)
            curv_batch = X_curv[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(base_batch, curv_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss/(len(X_base)//batch_size):.6f}")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'base_scale': BASE_SCALE,
        'curv_scale': CURV_SCALE
    }, 'experiments/exp023_conv/model.pth')
    print("✅ Model saved")
    
    # Evaluate
    print("\nEvaluating...")
    from experiments.exp023_conv.controller import Controller as ConvController
    
    costs = []
    for f in test_files[:20]:
        ctrl = ConvController()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    
    print(f"Test cost: {np.mean(costs):.2f}")
    print(f"BC baseline was: 74.88")

if __name__ == '__main__':
    train()



