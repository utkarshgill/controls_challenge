#!/usr/bin/env python3
"""Quick test to see what AsyncVectorEnv returns for infos"""

import numpy as np
import gymnasium as gym
from train_ppo_parallel import make_env

# Create 2 envs for testing
model_path = "./models/tinyphysics.onnx"
import glob
train_files = sorted(glob.glob("./data/*.csv"))[:100]

env = gym.vector.AsyncVectorEnv([
    make_env(model_path, train_files) for _ in range(2)
])

states, _ = env.reset()
print("Initial reset done")

for step in range(1000):
    actions = np.random.uniform(-2, 2, (2, 1))
    states, rewards, terminated, truncated, infos = env.step(actions)
    dones = np.logical_or(terminated, truncated)
    
    if np.any(dones):
        print(f"\nStep {step}: Episode finished!")
        print(f"infos type: {type(infos)}")
        print(f"infos keys: {infos.keys() if isinstance(infos, dict) else 'not a dict'}")
        print(f"infos: {infos}")
        break

env.close()

