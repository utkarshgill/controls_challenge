"""Debug why eval cost is 111 instead of 75"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from train import ActorCritic, HybridController, tinyphysics_model, prepare_future_plan
from tinyphysics import TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
eval_files = sorted(DATA_PATH.glob('*.csv'))[15000:15010]

print("Debugging eval cost discrepancy...")
print("="*60)

# Test 1: Pure PID baseline
print("\n1. Pure PID:")
costs_pid = []
for f in eval_files:
    pid = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(f), controller=pid, debug=False)
    cost = sim.rollout()
    costs_pid.append(cost['total_cost'])
print(f"   Mean: {np.mean(costs_pid):.1f}")
print(f"   Range: [{np.min(costs_pid):.1f}, {np.max(costs_pid):.1f}]")

# Test 2: Hybrid with untrained FF
print("\n2. Hybrid (untrained FF):")
actor_critic = ActorCritic()
costs_hybrid = []
ff_outputs = []
for f in eval_files:
    controller = HybridController(actor_critic, deterministic=True)
    
    # Monkey patch to collect FF outputs
    orig_update = controller.update
    episode_ff = []
    def debug_update(target, current, state, future_plan):
        result = orig_update(target, current, state, future_plan)
        future_plan_array = prepare_future_plan(future_plan, state)
        ff, _ = actor_critic.act(future_plan_array, deterministic=True)
        episode_ff.append(ff)
        return result
    controller.update = debug_update
    
    sim = TinyPhysicsSimulator(tinyphysics_model, str(f), controller=controller, debug=False)
    cost = sim.rollout()
    costs_hybrid.append(cost['total_cost'])
    ff_outputs.extend(episode_ff)

print(f"   Mean cost: {np.mean(costs_hybrid):.1f}")
print(f"   Range: [{np.min(costs_hybrid):.1f}, {np.max(costs_hybrid):.1f}]")
print(f"   FF mean: {np.mean(ff_outputs):.6f}, std: {np.std(ff_outputs):.6f}")
print(f"   FF range: [{np.min(ff_outputs):.6f}, {np.max(ff_outputs):.6f}]")

print("\n" + "="*60)
if np.mean(costs_hybrid) > np.mean(costs_pid) + 5:
    print("❌ Hybrid WORSE than PID - FF is hurting!")
elif abs(np.mean(costs_hybrid) - np.mean(costs_pid)) < 5:
    print("✓ Hybrid ≈ PID - FF outputting ~0 (good init)")
else:
    print("✓ Hybrid BETTER than PID - FF helping!")

