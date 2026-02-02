"""
BC with HAND-ENGINEERED temporal features (like LunarLander).

Instead of Conv1D learning compression implicitly,
we explicitly extract physics-motivated features from future trajectory.

State: [
    # Error correction (PID)
    error / v²,
    error_i / v,
    error_d,
    
    # Speed
    v / v_nominal,
    
    # Previous action (jerk minimization)
    prev_action,
    
    # MULTI-TIMESCALE FUTURE (explicit, not learned)
    immediate_curv,      # mean(curv[0:5]) - act NOW
    immediate_peak,      # max(abs(curv[0:5])) - urgency
    tactical_curv,       # mean(curv[5:20]) - prepare
    tactical_peak,       # max(abs(curv[5:20])) - difficulty
    strategic_curv,      # mean(curv[20:49]) - long-term
    
    # TRAJECTORY PROPERTIES
    curv_acceleration,   # How fast is turn tightening?
    curv_smoothness,     # std(curv[0:10]) - can track smoothly?
]

Total: 12 dimensions (vs 54 with Conv)
All have CLEAR physical meaning for control!
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

V_NOMINAL = 30.0

class SimpleMLPNetwork(nn.Module):
    """Simple MLP - no Conv, features are already engineered"""
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class PIDDataCollector:
    def __init__(self):
        from controllers.pid import Controller as PIDController
        self.pid = PIDController()
        self.states = []
        self.actions = []
        self.prev_action = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        v = max(state.v_ego, 1.0)
        v2 = max(v ** 2, 1.0)
        
        # Extract future curvatures
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
        curvs = np.array(curvs)
        
        # ENGINEER FEATURES (multi-timescale + trajectory properties)
        immediate_curv = curvs[0:5].mean()
        immediate_peak = np.abs(curvs[0:5]).max()
        tactical_curv = curvs[5:20].mean()
        tactical_peak = np.abs(curvs[5:20]).max()
        strategic_curv = curvs[20:49].mean()
        
        # Curvature acceleration (how fast is turn tightening?)
        curv_accel = (curvs[5] - curvs[0]) / 0.5 if len(curvs) > 5 else 0.0
        
        # Smoothness (can we track without jerking?)
        curv_smoothness = curvs[0:10].std()
        
        state_vec = np.array([
            error / v2,
            (self.pid.error_integral + error) / v,
            error - self.pid.prev_error,
            v / V_NOMINAL,
            self.prev_action,
            immediate_curv,
            immediate_peak,
            tactical_curv,
            tactical_peak,
            strategic_curv,
            curv_accel,
            curv_smoothness
        ], dtype=np.float32)
        
        action = self.pid.update(target, current, state, future_plan)
        
        self.states.append(state_vec)
        self.actions.append(action)
        self.prev_action = action
        
        return action

def collect_data(files, num_samples):
    X, y = [], []
    
    for f in files:
        if len(X) >= num_samples:
            break
        
        collector = PIDDataCollector()
        sim = TinyPhysicsSimulator(model_onnx, str(f), controller=collector)
        sim.rollout()
        
        X.extend(collector.states)
        y.extend(collector.actions)
    
    X = torch.FloatTensor(X[:num_samples])
    y = torch.FloatTensor(y[:num_samples]).unsqueeze(1)
    
    return X, y

def train():
    print("Collecting PID with ENGINEERED temporal features...")
    X, y = collect_data(train_files, num_samples=10000)
    print(f"Collected {len(X)} samples")
    print(f"State dimension: {X.shape[1]}")
    
    # Check feature stats
    print("\nFeature stats:")
    feature_names = ['error/v²', 'err_i/v', 'err_d', 'v/vnom', 'prev_a',
                     'imm_curv', 'imm_peak', 'tac_curv', 'tac_peak', 'str_curv',
                     'curv_acc', 'curv_smooth']
    for i, name in enumerate(feature_names):
        print(f"  {name:12s}: mean={X[:, i].mean():7.4f}, std={X[:, i].std():7.4f}")
    
    model = SimpleMLPNetwork(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    batch_size = 128
    num_epochs = 100
    
    for epoch in trange(num_epochs):
        indices = torch.randperm(len(X))
        epoch_loss = 0
        
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss/(len(X)//batch_size):.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'v_nominal': V_NOMINAL
    }, 'experiments/exp029_engineered_features/model.pth')
    print("✅ Model saved")
    
    print("\n" + "="*70)
    print("Creating controller for official evaluation...")
    print("="*70)

if __name__ == '__main__':
    train()



