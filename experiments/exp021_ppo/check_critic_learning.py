"""Is the critic actually learning to predict returns?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER

OBS_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333] + [0.1573] * 49, dtype=np.float32)
model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.critic = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(1) - 5.0)

# Load BC-initialized actor, random critic
ac = ActorCritic()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac.actor.load_state_dict(bc['model_state_dict'])

class Ctrl:
    def __init__(self):
        self.ei, self.pe = 0.0, 0.0
        self.states, self.rewards = [], []
        self.pl, self.sc = None, 0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        s_t = torch.FloatTensor(raw_state / OBS_SCALE)
        
        # Dense reward
        lat_err = (target - current) ** 2 * 100
        r = 0.0
        if self.pl and self.sc >= CONTROL_START_IDX and self.sc < COST_END_IDX:
            jerk = ((current - self.pl) / DEL_T) ** 2 * 100
            r = -(lat_err * LAT_ACCEL_COST_MULTIPLIER + jerk) / 1000.0
        
        self.states.append(s_t)
        self.rewards.append(r)
        self.pl = current
        self.sc += 1
        
        with torch.no_grad():
            action = ac.actor(s_t).item()
        return float(np.clip(action, -2.0, 2.0))

# Collect one trajectory
print("Collecting trajectory...")
ctrl = Ctrl()
sim = TinyPhysicsSimulator(model_onnx, str(all_files[0]), controller=ctrl)
cost = sim.rollout()['total_cost']

print(f"Cost: {cost:.2f}, Steps: {len(ctrl.states)}, Total reward: {sum(ctrl.rewards):.2f}\n")

# Compute actual returns (discounted future rewards)
gamma = 0.99
returns = []
G = 0
for r in reversed(ctrl.rewards):
    G = r + gamma * G
    returns.insert(0, G)
returns = np.array(returns)

# Get critic predictions
with torch.no_grad():
    states = torch.stack(ctrl.states)
    value_preds = ac.critic(states).squeeze().numpy()

print("Critic predictions vs actual returns:")
print(f"  Value pred range: [{value_preds.min():.3f}, {value_preds.max():.3f}]")
print(f"  Actual returns range: [{returns.min():.3f}, {returns.max():.3f}]")
print(f"  Value pred mean: {value_preds.mean():.3f}")
print(f"  Actual returns mean: {returns.mean():.3f}")
print(f"  Correlation: {np.corrcoef(value_preds, returns)[0,1]:.3f}")
print(f"  MSE: {((value_preds - returns)**2).mean():.3f}")

# Sample some predictions
print("\nSample predictions (first 20 steps):")
for i in range(min(20, len(returns))):
    print(f"  Step {i:3d}: pred={value_preds[i]:+7.3f}  actual={returns[i]:+7.3f}  error={value_preds[i]-returns[i]:+7.3f}")

print("\nConclusion:")
if abs(np.corrcoef(value_preds, returns)[0,1]) < 0.1:
    print("  ❌ Critic is RANDOM - no correlation with actual returns")
    print("  → PPO gradients are noise!")
else:
    print(f"  ✓ Critic has some signal (corr={np.corrcoef(value_preds, returns)[0,1]:.2f})")



