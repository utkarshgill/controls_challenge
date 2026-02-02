"""Evaluate the 'best' model from Terminal 16 - is 74.3 real or noise?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random, numpy as np, torch, torch.nn as nn
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

OBS_SCALE = np.array([0.3664, 7.1769, 0.1396, 38.7333] + [0.1573] * 49, dtype=np.float32)
model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:20000]

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(53, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

class Ctrl:
    def __init__(self, actor):
        self.actor = actor
        self.ei, self.pe = 0.0, 0.0
    
    def update(self, target, current, state, future_plan):
        e = target - current
        self.ei += e
        ed = e - self.pe
        self.pe = e
        curvs = [(future_plan.lataccel[i] - state.roll_lataccel) / max(state.v_ego**2, 1.0) 
                 if i < len(future_plan.lataccel) else 0.0 for i in range(49)]
        raw_state = np.array([e, self.ei, ed, state.v_ego] + curvs, dtype=np.float32)
        
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(raw_state / OBS_SCALE)).item()
        
        return float(np.clip(action, -2.0, 2.0))

# Load BC baseline
ac_bc = ActorCritic()
bc = torch.load('./experiments/exp020_normalized/model.pth', map_location='cpu', weights_only=False)
ac_bc.actor.load_state_dict(bc['model_state_dict'])

# Load PPO "best"
best_path = Path('./experiments/exp021_ppo/model_best.pth')
if best_path.exists():
    ac_ppo = ActorCritic()
    ppo_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    # Extract actor weights from ActorCritic state dict
    actor_state = {k.replace('actor.', ''): v for k, v in ppo_ckpt['model_state_dict'].items() if k.startswith('actor.')}
    ac_ppo.actor.load_state_dict(actor_state)
    
    print("Evaluating on 50 test routes...")
    print("="*60)
    
    bc_costs, ppo_costs = [], []
    for f in test_files[:50]:
        ctrl_bc = Ctrl(ac_bc.actor)
        sim_bc = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl_bc)
        bc_costs.append(sim_bc.rollout()['total_cost'])
        
        ctrl_ppo = Ctrl(ac_ppo.actor)
        sim_ppo = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl_ppo)
        ppo_costs.append(sim_ppo.rollout()['total_cost'])
    
    print(f"\nBC baseline:")
    print(f"  Mean: {np.mean(bc_costs):.2f}")
    print(f"  Std:  {np.std(bc_costs):.2f}")
    print(f"  Med:  {np.median(bc_costs):.2f}")
    
    print(f"\nPPO 'best':")
    print(f"  Mean: {np.mean(ppo_costs):.2f}")
    print(f"  Std:  {np.std(ppo_costs):.2f}")
    print(f"  Med:  {np.median(ppo_costs):.2f}")
    
    print(f"\nDifference:")
    print(f"  Mean: {np.mean(bc_costs) - np.mean(ppo_costs):+.2f}")
    print(f"  Routes where PPO better: {sum(p < b for p, b in zip(ppo_costs, bc_costs))}/50")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(bc_costs, ppo_costs)
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        if np.mean(ppo_costs) < np.mean(bc_costs):
            print("  ✓ PPO is statistically significantly BETTER")
        else:
            print("  ✗ PPO is statistically significantly WORSE")
    else:
        print("  → No significant difference (PPO ≈ BC)")
else:
    print("model_best.pth not found")



