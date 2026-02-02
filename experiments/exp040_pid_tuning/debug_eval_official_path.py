"""
Debug: Use EXACT same code path as tinyphysics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tinyphysics import run_rollout

DATA_PATH = Path(__file__).parent.parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'

all_files_sorted = sorted(DATA_PATH.glob('*.csv'))
eval_files = all_files_sorted[:100]

print(f"Using files: {eval_files[0].name} to {eval_files[99].name}")

costs = []
for f in eval_files:
    result = run_rollout(f, 'pid', str(MODEL_PATH), debug=False)
    costs.append(result[0]['total_cost'])

print(f"\nPID baseline (using tinyphysics.run_rollout): {np.mean(costs):.2f}")
print(f"Min: {np.min(costs):.1f}, Max: {np.max(costs):.1f}")

