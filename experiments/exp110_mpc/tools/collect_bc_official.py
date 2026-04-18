"""Collect (obs, delta) from the official tinyphysics.py CPU sim.
Patches the lookup controller to record deltas during the batch eval.
Run: .venv/bin/python collect_bc_official.py"""

import sys, numpy as np
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
)

N_CTRL = COST_END_IDX - CONTROL_START_IDX
ACTIONS_PATH = Path("experiments/exp110_mpc/checkpoints/actions_5k_final.npz")
N_ROUTES = 5000

# Load actions
_actions_data = np.load(ACTIONS_PATH)
_actions_dict = {k: _actions_data[k] for k in _actions_data.files}
print(f"Loaded {len(_actions_dict)} routes")


def run_one(data_path):
    """Run one route, collect deltas."""
    fname = Path(data_path).name
    if fname not in _actions_dict:
        return None

    actions = _actions_dict[fname]
    deltas = []
    prev_action = [0.0]

    class Ctrl:
        def __init__(self):
            self.step = 0

        def update(self, target, current, state, future_plan=None):
            self.step += 1
            step_idx = self.step + CONTEXT_LENGTH - 1
            if step_idx < CONTROL_START_IDX:
                return 0.0
            ci = step_idx - CONTROL_START_IDX
            if ci >= N_CTRL:
                return 0.0
            action = float(actions[ci])
            delta = action - prev_action[0]
            deltas.append(delta)
            prev_action[0] = action
            return action

    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    ctrl = Ctrl()
    sim = TinyPhysicsSimulator(model, str(data_path), ctrl)
    cost = sim.rollout()

    return fname, np.array(deltas, dtype=np.float32), cost["total_cost"]


data_path = Path("./data")
files = sorted(data_path.iterdir())[:N_ROUTES]
print(f"Processing {len(files)} routes on CPU...")

results = process_map(run_one, files, max_workers=8, chunksize=50)

# Filter and save
all_deltas = {}
costs = []
for r in results:
    if r is not None:
        fname, deltas, cost = r
        all_deltas[fname] = deltas
        costs.append(cost)

print(f"Collected {len(all_deltas)} routes, mean cost={np.mean(costs):.1f}")
np.savez("experiments/exp110_mpc/checkpoints/bc_deltas.npz", **all_deltas)
print("Saved to experiments/exp110_mpc/checkpoints/bc_deltas.npz")
