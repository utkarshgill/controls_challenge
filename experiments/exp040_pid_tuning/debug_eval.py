"""
Debug: Replicate exact exp040 evaluation logic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import random
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

# EXACT same initialization as exp040
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'

# EXACT same file selection as exp040
all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

print(f"Eval files[0]: {eval_files[0].name}")
print(f"Eval files[99]: {eval_files[99].name}")
print(f"Eval files[10-15]: {[f.name for f in eval_files[10:15]]}")

# Load simulator AFTER seeds (same as exp040)
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

# EXACT same rollout logic as exp040
pid_costs = []
for i in range(100):
    pid_controller = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(eval_files[i]), controller=pid_controller, debug=False)
    cost_dict = sim.rollout()
    pid_costs.append(cost_dict['total_cost'])

print(f"\nPID baseline: {np.mean(pid_costs):.1f}")
print(f"PID min: {np.min(pid_costs):.1f}, max: {np.max(pid_costs):.1f}")

