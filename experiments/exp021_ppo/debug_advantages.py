"""Debug: Check what advantages actually look like"""
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

ac = ActorCritic()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac.actor.load_state_dict(bc['model_state_dict'])

class Ctrl:
    def __init__(self):
        self.error_integral, self.prev_error = 0.0, 0.0
        self.states, self.rewards = [], []
        self.prev_lataccel, self.step_count = None, 0
        
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        e_diff = error - self.prev_error
        self.prev_error = error
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([error, self.error_integral, e_diff, state.v_ego] + curvs, dtype=np.float32)
        
        # Dense reward
        lat_err = (target - current) ** 2 * 100
        if self.prev_lataccel and self.step_count >= CONTROL_START_IDX and self.step_count < COST_END_IDX:
            jerk = ((current - self.prev_lataccel) / DEL_T) ** 2 * 100
            step_cost = lat_err * LAT_ACCEL_COST_MULTIPLIER + jerk
            reward = -step_cost / 1000.0
        else:
            reward = 0.0
            
        self.states.append(torch.FloatTensor(raw_state / OBS_SCALE))
        self.rewards.append(reward)
        self.prev_lataccel = current
        self.step_count += 1
        
        with torch.no_grad():
            mean = ac.actor(self.states[-1])
            action = torch.distributions.Normal(mean, ac.log_std.exp()).sample().item()
        return float(np.clip(action, -2.0, 2.0))

# Collect one trajectory
print("Collecting trajectory...")
ctrl = Ctrl()
sim = TinyPhysicsSimulator(model_onnx, str(all_files[0]), controller=ctrl)
cost = sim.rollout()['total_cost']

print(f"\nFinal cost: {cost:.2f}")
print(f"Steps: {len(ctrl.rewards)}")
print(f"Total reward: {sum(ctrl.rewards):.3f}")
print(f"Mean reward: {np.mean(ctrl.rewards):.6f}")
print(f"Reward range: [{min(ctrl.rewards):.6f}, {max(ctrl.rewards):.6f}]")

# Get values
with torch.no_grad():
    states = torch.stack(ctrl.states)
    values = ac.critic(states).squeeze()

print(f"\nValue stats:")
print(f"  Mean: {values.mean().item():.6f}")
print(f"  Std: {values.std().item():.6f}")
print(f"  Range: [{values.min().item():.6f}, {values.max().item():.6f}]")

# Compute advantages (simple TD)
rewards = torch.FloatTensor(ctrl.rewards)
advantages = []
for t in range(len(rewards)):
    if t < len(rewards) - 1:
        td_target = rewards[t] + 0.99 * values[t+1]
    else:
        td_target = rewards[t]
    adv = td_target - values[t]
    advantages.append(adv.item())

advantages = np.array(advantages)
print(f"\nAdvantage stats (before normalization):")
print(f"  Mean: {advantages.mean():.6f}")
print(f"  Std: {advantages.std():.6f}")
print(f"  Range: [{advantages.min():.6f}, {advantages.max():.6f}]")
print(f"  Positive: {(advantages > 0).sum()}/{len(advantages)}")

# After normalization
adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
print(f"\nAdvantage stats (after normalization):")
print(f"  Mean: {adv_norm.mean():.6f}")
print(f"  Std: {adv_norm.std():.6f}")
print(f"  Range: [{adv_norm.min():.6f}, {adv_norm.max():.6f}]")

