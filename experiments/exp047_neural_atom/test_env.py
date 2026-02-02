"""
Test that the Gym environment matches the official simulator
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.exp047_neural_atom.env import TinyPhysicsEnv

# Load supervised learning weights
params = np.load(Path(__file__).parent / 'tanh_params_proper.npz')
w = params['w']
b = params['b']

print("="*60)
print("Testing Gym environment with supervised learning weights")
print("="*60)
print(f"Weights: {w}")
print(f"Bias: {b}")
print()

model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
data_dir = Path(__file__).parent.parent.parent / 'data'

# Test on 10 routes
routes = sorted(list(data_dir.glob('*.csv')))[:10]

costs = []
for route in routes:
    env = TinyPhysicsEnv(model_path, route)
    obs, _ = env.reset()
    
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        # Use supervised learning action
        action = np.tanh(np.dot(w, obs) + b)
        obs, reward, terminated, truncated, _ = env.step(np.array([action]))
        done = terminated or truncated
        episode_reward += reward
        steps += 1
    
    # Convert reward to cost (divide by steps to get mean)
    avg_reward = episode_reward / steps
    cost_from_reward = -avg_reward * 100
    
    # Get official cost from simulator
    official_cost = env.sim.compute_cost()['total_cost']
    
    costs.append(official_cost)
    print(f"Route {route.name}: steps={steps}, cost_from_reward={cost_from_reward:.1f}, official_cost={official_cost:.1f}")

print()
print(f"Average cost: {np.mean(costs):.1f} ± {np.std(costs):.1f}")
print(f"Expected from batch eval: 120.8")
print()

if abs(np.mean(costs) - 120.8) > 20:
    print("⚠️  WARNING: Large discrepancy detected!")
else:
    print("✓ Environment looks correct")
