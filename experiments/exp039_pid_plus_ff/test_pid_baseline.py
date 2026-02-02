"""Test that PID baseline works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# Test PID on a few files
test_files = sorted(DATA_PATH.glob('*.csv'))[:10]

costs = []
for test_file in test_files:
    controller = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(test_file), controller=controller, debug=False)
    cost_dict = sim.rollout()
    costs.append(cost_dict['total_cost'])
    print(f"{test_file.name}: {cost_dict['total_cost']:.2f}")

print(f"\nPID baseline: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
print("Expected: ~75")

