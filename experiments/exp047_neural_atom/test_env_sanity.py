"""
Test if environment is sane - PID alone should give ~85 cost
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from step6_ppo_clean import TinyPhysicsEnv, model_path

# Test single episode with zero FF (pure PID)
data_file = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
env = TinyPhysicsEnv(model_path, data_file)

obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    action = np.array([0.0])  # Zero FF = pure PID
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1

print(f"Steps: {steps}")
print(f"Total reward: {total_reward:.2f}")
print(f"Official cost: {info.get('official_cost', 'NOT FOUND')}")
print(f"Expected: ~85-100 for PID alone")
