"""
BC with FRICTION CIRCLE aware state representation.

KEY INSIGHT: Vehicle dynamics has friction circle constraint:
    sqrt(lat_accel² + long_accel²) ≤ μ×g

This means:
- When accelerating (high a_ego), LESS lateral grip available
- Error is "harder" to correct when friction is limited
- Control law must be friction-aware!

State representation that makes this learnable:
- Normalize errors by AVAILABLE lateral capacity
- Make friction constraint explicit in the input
- Network learns control within physical limits
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

# Physics constants
MU_G = 9.8  # Maximum combined acceleration (μ × g)
V_NOMINAL = 30.0
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
        
        # Input: 7 (friction-aware base) + 128 (conv)
        self.mlp = nn.Sequential(
            nn.Linear(7 + 16*8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, base_features, curv_sequence):
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
        self.prev_action = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        
        # FRICTION CIRCLE AWARE STATE
        v = max(state.v_ego, 1.0)
        v2 = max(v ** 2, 1.0)
        
        # Available lateral capacity after accounting for longitudinal accel
        a_long = state.a_ego
        friction_used_long = min(abs(a_long) / MU_G, 0.99)  # Cap at 0.99 to avoid sqrt(negative)
        available_lat_fraction = np.sqrt(1.0 - friction_used_long ** 2)
        available_lat_accel = available_lat_fraction * MU_G
        
        # Normalize error by available capacity
        error_friction_norm = error / max(available_lat_accel, 1.0)
        
        # Current friction utilization
        total_accel = np.sqrt(current ** 2 + a_long ** 2)
        friction_utilization = total_accel / MU_G
        
        base = np.array([
            error_friction_norm,                        # Error relative to available grip
            (self.pid.error_integral + error) / v,     # Distance-based integral
            error - self.pid.prev_error,               # Rate
            v / V_NOMINAL,                             # Normalized speed
            friction_utilization,                       # How much friction we're using
            available_lat_fraction,                     # How much friction available for lat
            self.prev_action                           # Previous action
        ], dtype=np.float32)
        
        # Future curvatures
        curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                future_v = future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego
                future_v2 = max(future_v ** 2, 1.0)
                future_roll = future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel
                lat = future_plan.lataccel[i]
                curv = (lat - future_roll) / future_v2
                curvs.append(curv)
            else:
                curvs.append(0.0)
        curv_seq = np.array(curvs, dtype=np.float32)
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.base_states.append(base)
        self.curv_sequences.append(curv_seq)
        self.actions.append(action)
        self.prev_action = action
        
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
    print("Collecting PID with FRICTION CIRCLE aware state...")
    X_base, X_curv, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X_base)} samples")
    
    print(f"\nFriction utilization stats:")
    print(f"  Mean: {X_base[:, 4].mean():.3f}")
    print(f"  Max: {X_base[:, 4].max():.3f}")
    print(f"  Available lat fraction: {X_base[:, 5].mean():.3f} ± {X_base[:, 5].std():.3f}")
    
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
        'curv_scale': CURV_SCALE,
        'v_nominal': V_NOMINAL,
        'mu_g': MU_G
    }, 'experiments/exp028_friction_circle/model.pth')
    print("✅ Model saved")
    
    # Evaluate
    print("\nEvaluating...")
    from experiments.exp028_friction_circle.controller import Controller as FrictionController
    
    costs = []
    for f in test_files[:20]:
        ctrl = FrictionController()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
        costs.append(sim.rollout()['total_cost'])
    
    print(f"\nFriction-circle BC: {np.mean(costs):.2f}")
    print(f"Previous (exp025): 65.02")
    print(f"Difference: {65.02 - np.mean(costs):.2f}")

if __name__ == '__main__':
    train()



