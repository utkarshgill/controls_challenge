"""
Debug: WITHOUT setting seeds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'

# NO SEEDS SET
all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

pid_costs = []
for i in range(100):
    pid_controller = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(eval_files[i]), controller=pid_controller, debug=False)
    cost_dict = sim.rollout()
    pid_costs.append(cost_dict['total_cost'])

print(f"PID baseline (NO seeds): {np.mean(pid_costs):.1f}")

