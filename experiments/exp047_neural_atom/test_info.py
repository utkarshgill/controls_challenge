"""
Test that info dict contains official_cost
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.exp047_neural_atom.env import TinyPhysicsEnv

model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
data_file = Path(__file__).parent.parent.parent / 'data' / '00000.csv'

env = TinyPhysicsEnv(model_path, data_file)
obs, _ = env.reset()

done = False
step_count = 0
while not done:
    action = np.array([0.0])  # Zero action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step_count += 1
    
    if done:
        print(f"Episode ended after {step_count} steps")
        print(f"Info dict: {info}")
        print(f"Official cost: {info.get('official_cost', 'NOT FOUND')}")
