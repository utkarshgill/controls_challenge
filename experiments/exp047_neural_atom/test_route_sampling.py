"""Test if routes are being sampled properly"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from step6_ppo_clean import make_env, train_routes

if __name__ == '__main__':
    print("Testing route sampling over 100 resets...")
    
    env = make_env(train_routes[:50])  # Use 50 routes
    
    sampled_routes = []
    for i in range(100):
        obs, _ = env.reset()
        # Can't directly access route from AsyncVectorEnv, so just track resets
        sampled_routes.append(i)
    
    print(f"Performed {len(sampled_routes)} resets")
    print("If route sampling works, we should see different episodes")
    
    # Instead, let's test a single env directly
    from step6_ppo_clean import TinyPhysicsEnv, model_path
    
    single_env = TinyPhysicsEnv(model_path, train_routes[:50], worker_id=0)
    
    routes_sampled = []
    for i in range(20):
        single_env.reset()
        routes_sampled.append(single_env.data_path.name)
    
    print(f"\n20 resets from single env:")
    print(f"Unique routes: {len(set(routes_sampled))}/20")
    print(f"First 10: {routes_sampled[:10]}")
    
    if len(set(routes_sampled)) < 10:
        print("\n⚠️ WARNING: Not enough diversity in route sampling!")
    else:
        print("\n✅ Route sampling looks good")
    
    env.close()
