"""
Verify BC checkpoint loads correctly and debug weight mapping
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Load BC checkpoint
bc_path = Path('./experiments/exp020_normalized/model.pth')
bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)

print("BC checkpoint keys:", bc_checkpoint.keys())
print("\nBC model state_dict:")
for key, value in bc_checkpoint['model_state_dict'].items():
    print(f"  {key}: {value.shape}")

# Recreate BC network (from exp020)
bc_net = nn.Sequential(
    nn.Linear(53, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 1)
)
bc_net.load_state_dict(bc_checkpoint['model_state_dict'])
bc_net.eval()

OBS_SCALE = bc_checkpoint['obs_scale']

# Test BC network
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:20000]
model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

class BCController:
    def __init__(self, net, obs_scale):
        self.net = net
        self.obs_scale = obs_scale
        self.error_integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        v_ego = state.v_ego
        
        future_curvs = []
        for i in range(49):
            if i < len(future_plan.lataccel):
                lat = future_plan.lataccel[i]
                curv = (lat - state.roll_lataccel) / max(v_ego ** 2, 1.0)
                future_curvs.append(curv)
            else:
                future_curvs.append(0.0)
        
        raw_state = np.array([error, self.error_integral, error_diff, v_ego] + future_curvs, dtype=np.float32)
        x = torch.FloatTensor(raw_state / self.obs_scale).unsqueeze(0)
        
        with torch.no_grad():
            action = self.net(x).item()
        
        self.prev_error = error
        return float(max(-2.0, min(2.0, action)))

print("\nVerifying BC network performance...")
costs = []
for f in test_files[:5]:
    controller = BCController(bc_net, OBS_SCALE)
    sim = TinyPhysicsSimulator(model, str(f), controller=controller)
    cost_dict = sim.rollout()
    costs.append(cost_dict['total_cost'])

bc_cost = np.mean(costs)
print(f"BC cost (5 routes): {bc_cost:.2f}")
print(f"Expected: ~74.88")

if abs(bc_cost - 74.88) > 10:
    print("❌ BC checkpoint verification FAILED")
else:
    print("✅ BC checkpoint verified")



