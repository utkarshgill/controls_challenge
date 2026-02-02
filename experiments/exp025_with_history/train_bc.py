"""
BC with ACTION HISTORY - crucial for minimizing jerk cost!
State: [error, error_i, error_d, v_ego, PREV_ACTION] + future sequences
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

# Updated: 5D base state (added prev_action)
BASE_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333, 0.5], dtype=np.float32)  # last is action scale
CURV_SCALE = 0.1573

class ConvBCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # MLP: 5 (base + prev_action) + 16*8 (conv) = 133
        self.mlp = nn.Sequential(
            nn.Linear(5 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, curv_sequence):
        # base_features: (batch, 5) - includes prev_action
        # curv_sequence: (batch, 49)
        
        curv_input = curv_sequence.unsqueeze(1)
        conv_out = self.conv(curv_input)
        conv_flat = conv_out.reshape(conv_out.size(0), -1)
        combined = torch.cat([base_features, conv_flat], dim=1)
        return self.mlp(combined)

class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.base_states = []
        self.curv_sequences = []
        self.actions = []
        self.prev_action = 0.0  # Track previous action
    
    def update(self, target, current, state, future_plan):
        error = target - current
        
        # Base features INCLUDING previous action
        base = np.array([
            error,
            self.pid.error_integral + error,
            error - self.pid.prev_error,
            state.v_ego,
            self.prev_action  # KEY: Network can now minimize jerk!
        ], dtype=np.float32)
        
        # Future curvatures
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                # Use future v_ego
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                curv = (lat - future_roll) / max(future_v ** 2, 1.0)
                curvs.append(curv)
            else:
                curvs.append(0.0)
        curv_seq = np.array(curvs, dtype=np.float32)
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.base_states.append(base)
        self.curv_sequences.append(curv_seq)
        self.actions.append(action)
        self.prev_action = action  # Update for next step
        
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
    print("Collecting PID demonstrations with ACTION HISTORY...")
    X_base, X_curv, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X_base)} samples")
    
    # Normalize
    X_base = X_base / torch.FloatTensor(BASE_SCALE)
    X_curv = X_curv / CURV_SCALE
    
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
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'base_scale': BASE_SCALE,
        'curv_scale': CURV_SCALE
    }, 'experiments/exp025_with_history/model.pth')
    print("✅ Model saved")
    
    # Evaluate
    print("\nEvaluating...")
    from experiments.exp025_with_history.controller import Controller as HistoryController
    
    costs = []
    for f in test_files[:20]:
        ctrl = HistoryController()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    
    print(f"\nWith ACTION HISTORY: {np.mean(costs):.2f}")
    print(f"Without history: 65.63")
    print(f"Difference: {65.63 - np.mean(costs):.2f}")

if __name__ == '__main__':
    train()



