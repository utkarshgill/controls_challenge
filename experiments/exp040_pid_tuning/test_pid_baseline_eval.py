"""Verify what PID actually achieves on our eval_files"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import random
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'

tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# Same split as train.py
all_files = sorted(DATA_PATH.glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
eval_files = all_files[15000:17500]

print("Testing PID on eval_files (same as used in BC evaluation)")
print("="*60)

costs = []
for i in range(32):  # Same as BC eval
    controller = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(eval_files[i]), controller=controller, debug=False)
    cost_dict = sim.rollout()
    costs.append(cost_dict['total_cost'])

mean_cost = np.mean(costs)
std_cost = np.std(costs)

print(f"PID on eval_files: {mean_cost:.1f} ± {std_cost:.1f}")
print(f"\nIf BC eval is ~114 and this is ~114, BC is perfect.")
print(f"If this is ~75, there's a bug in OneNeuronController.")

