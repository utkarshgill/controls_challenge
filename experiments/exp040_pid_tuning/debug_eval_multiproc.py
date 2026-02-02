"""
Debug: Use multiprocessing like tinyphysics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'

def run_eval(data_file):
    model = TinyPhysicsModel(str(MODEL_PATH), debug=False)
    controller = PIDController()
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    cost_dict = sim.rollout()
    return cost_dict['total_cost']

all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

# Use multiprocessing like tinyphysics.py
results = process_map(run_eval, eval_files, max_workers=16, chunksize=10)

print(f"PID baseline (multiprocessing): {np.mean(results):.2f}")

