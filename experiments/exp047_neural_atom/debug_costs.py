"""
Debug exactly where cost tracking breaks
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from step6_ppo_clean import make_env, train_routes, FFPolicy

if __name__ == '__main__':
    print("="*60)
    print("DEBUGGING COST TRACKING")
    print("="*60)

    # Use 5 routes
    test_routes = train_routes[:5]
    env = make_env(test_routes)
    policy = FFPolicy()

    print(f"\nNum envs: {env.num_envs}")
    print(f"Routes being used: {[str(r.name) for r in test_routes]}")
    print()

    states, _ = env.reset()
    costs_collected = []
    step_count = 0

    print("Starting rollout...")
    for step in range(1000):
        # Use deterministic for consistency
        actions = policy.act(states, deterministic=True)
        states, rewards, terminated, truncated, infos = env.step(actions)
        
        d = np.logical_or(terminated, truncated)
        
        if np.any(d):
            print(f"\nStep {step}: Episode(s) finished")
            print(f"  Done mask: {d}")
            print(f"  Infos type: {type(infos)}")
            print(f"  Infos keys: {infos.keys() if isinstance(infos, dict) else 'not a dict'}")
            
            # Check official_cost
            if 'official_cost' in infos:
                official_costs = infos['official_cost']
                print(f"  official_cost: {official_costs}")
                print(f"  official_cost type: {type(official_costs)}")
                
                for idx in np.where(d)[0]:
                    cost = official_costs[idx]
                    print(f"    Env {idx}: cost={cost:.1f}")
                    costs_collected.append(cost)
            else:
                print(f"  ERROR: 'official_cost' not in infos!")
                print(f"  Available keys: {list(infos.keys())}")
        
        step_count += 1
        if len(costs_collected) >= 16:
            break

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Collected {len(costs_collected)} episode costs")
    if len(costs_collected) > 0:
        print(f"Mean cost: {np.mean(costs_collected):.1f}")
        print(f"Std cost: {np.std(costs_collected):.1f}")
        print(f"Individual costs: {[f'{c:.1f}' for c in costs_collected]}")
    else:
        print("ERROR: No costs collected!")

    env.close()
