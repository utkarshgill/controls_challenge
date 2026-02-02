"""
Test eval rollout to see what costs are being collected
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from step6_ppo_clean import make_env, eval_routes, FFPolicy, rollout

if __name__ == '__main__':
    print("="*60)
    print("TESTING EVAL ROLLOUT")
    print("="*60)
    
    eval_env = make_env(eval_routes)
    policy = FFPolicy()
    
    print(f"\nRunning eval rollout (deterministic=True)...")
    _, _, _, _, eval_costs = rollout(eval_env, policy, 20000, deterministic=True)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Collected {len(eval_costs)} episode costs")
    if len(eval_costs) > 0:
        print(f"Mean cost: {np.mean(eval_costs):.1f}")
        print(f"Median cost: {np.median(eval_costs):.1f}")
        print(f"Std cost: {np.std(eval_costs):.1f}")
        print(f"Min cost: {np.min(eval_costs):.1f}")
        print(f"Max cost: {np.max(eval_costs):.1f}")
        print(f"\nFirst 20 costs: {[f'{c:.1f}' for c in eval_costs[:20]]}")
    else:
        print("ERROR: No costs collected!")
    
    eval_env.close()
    
    print(f"\n{'='*60}")
    print("For comparison, batch metrics with same weights shows ~84-85")
    print("="*60)
