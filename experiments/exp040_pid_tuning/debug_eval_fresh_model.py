"""
Debug: Load FRESH model for each evaluation (like tinyphysics.py multiprocessing)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'

all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

pid_costs = []
for i in range(100):
    # FRESH model each time (like multiprocessing does)
    model = TinyPhysicsModel(str(MODEL_PATH), debug=False)
    pid_controller = PIDController()
    sim = TinyPhysicsSimulator(model, str(eval_files[i]), controller=pid_controller, debug=False)
    cost_dict = sim.rollout()
    pid_costs.append(cost_dict['total_cost'])

print(f"PID baseline (fresh model each time): {np.mean(pid_costs):.1f}")

