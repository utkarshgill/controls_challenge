"""
BC with CORRECT future curvature calculation
Use future v_ego for each future point, not current v_ego
ALSO: Add future a_ego as additional input
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

BASE_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333], dtype=np.float32)
CURV_SCALE = 0.1573
ACCEL_SCALE = 2.0  # a_ego scale

class ConvBCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv on TWO sequences: curvature (49) + a_ego (49)
        # Stack as 2-channel input
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # MLP: 4 (base) + 16*8 (conv) = 132
        self.mlp = nn.Sequential(
            nn.Linear(4 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, future_sequences):
        # base_features: (batch, 4)
        # future_sequences: (batch, 2, 49) - [curvature, a_ego]
        
        conv_out = self.conv(future_sequences)  # (batch, 16, 8)
        conv_flat = conv_out.reshape(conv_out.size(0), -1)
        combined = torch.cat([base_features, conv_flat], dim=1)
        return self.mlp(combined)

class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.base_states = []
        self.future_seqs = []
        self.actions = []
    
    def update(self, target, current, state, future_plan):
        error = target - current
        
        base = np.array([
            error,
            self.pid.error_integral + error,
            error - self.pid.prev_error,
            state.v_ego
        ], dtype=np.float32)
        
        # CORRECT future curvature + a_ego sequence
        curvs = []
        accels = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                # Use FUTURE v_ego, not current!
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                future_lat = future_plan.lataccel[i]
                future_a = future_plan.a_ego[i] if i < len(future_plan.a_ego) else state.a_ego
                
                curv = (future_lat - future_roll) / max(future_v ** 2, 1.0)
                curvs.append(curv)
                accels.append(future_a)
            else:
                curvs.append(0.0)
                accels.append(0.0)
        
        # Stack as (2, 49)
        future_seq = np.stack([curvs, accels], axis=0).astype(np.float32)
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.base_states.append(base)
        self.future_seqs.append(future_seq)
        self.actions.append(action)
        
        return action

def collect_data(files, num_samples):
    X_base, X_future, y = [], [], []
    
    for f in files:
        if len(X_base) >= num_samples:
            break
        
        collector = PIDDataCollector()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=collector)
        sim.rollout()
        
        X_base.extend(collector.base_states)
        X_future.extend(collector.future_seqs)
        y.extend(collector.actions)
    
    X_base = torch.FloatTensor(X_base[:num_samples])
    X_future = torch.FloatTensor(X_future[:num_samples])
    y = torch.FloatTensor(y[:num_samples]).unsqueeze(1)
    
    return X_base, X_future, y

def train():
    print("Collecting PID demonstrations with CORRECT future calculations...")
    X_base, X_future, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X_base)} samples")
    
    # Normalize
    X_base = X_base / torch.FloatTensor(BASE_SCALE)
    # Normalize each channel separately
    X_future[:, 0, :] = X_future[:, 0, :] / CURV_SCALE  # curvature
    X_future[:, 1, :] = X_future[:, 1, :] / ACCEL_SCALE  # a_ego
    
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
            future_batch = X_future[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(base_batch, future_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss/(len(X_base)//batch_size):.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'base_scale': BASE_SCALE,
        'curv_scale': CURV_SCALE,
        'accel_scale': ACCEL_SCALE
    }, 'experiments/exp024_correct_future/model.pth')
    print("✅ Model saved")
    
    # Evaluate
    print("\nEvaluating...")
    from experiments.exp024_correct_future.controller import Controller as CorrectController
    
    costs = []
    for f in test_files[:20]:
        ctrl = CorrectController()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    
    print(f"\nCORRECT future BC: {np.mean(costs):.2f}")
    print(f"Previous (wrong): 65.63")
    print(f"Improvement: {65.63 - np.mean(costs):.2f}")

if __name__ == '__main__':
    train()



