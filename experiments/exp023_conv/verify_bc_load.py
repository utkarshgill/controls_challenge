"""Verify BC loading works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Load BC model
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
        self.mlp = nn.Sequential(
            nn.Linear(4 + 16*8, 128),
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

bc_net = ConvBCNetwork()
ckpt = torch.load('experiments/exp023_conv/model.pth', map_location='cpu', weights_only=False)
bc_net.load_state_dict(ckpt['model_state_dict'])
bc_net.eval()

print("BC model state dict keys:")
for k in ckpt['model_state_dict'].keys():
    print(f"  {k}")

# Test
BASE_SCALE = ckpt['base_scale']
CURV_SCALE = ckpt['curv_scale']

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:20000]

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

class Ctrl:
    def __init__(self):
        self.ei, self.pe = 0.0, 0.0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        
        base = np.array([e, self.ei, ed, state.v_ego], dtype=np.float32)
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        curv_seq = np.array(curvs, dtype=np.float32)
        
        base_norm = torch.FloatTensor(base / BASE_SCALE).unsqueeze(0)
        curv_norm = torch.FloatTensor(curv_seq / CURV_SCALE).unsqueeze(0)
        
        with torch.no_grad():
            action = bc_net(base_norm, curv_norm).item()
        
        return float(np.clip(action, -2.0, 2.0))

costs = []
for f in test_files[:20]:
    ctrl = Ctrl()
    sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
    costs.append(sim.rollout()['total_cost'])

print(f"\nBC test cost (20 routes): {np.mean(costs):.2f}")
print("Expected: ~65.63")

# Also check first 10 vs last 10
print(f"First 10: {np.mean(costs[:10]):.2f}")
print(f"Last 10: {np.mean(costs[10:]):.2f}")

