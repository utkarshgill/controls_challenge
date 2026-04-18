"""Collect (obs, raw_delta) pairs by running the official CPU sim with MPC actions.
Uses exp055's _build_obs_bc to build the exact same 256-dim observation vectors.

Usage: .venv/bin/python collect_bc.py
"""

import sys, numpy as np, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    DEL_T,
    ACC_G,
    FUTURE_PLAN_STEPS,
    State,
    FuturePlan,
)
from experiments.exp055_batch_of_batch.train import (
    _build_obs_bc,
    HIST_LEN,
    DELTA_SCALE_MAX,
    OBS_DIM,
)

N_CTRL = COST_END_IDX - CONTROL_START_IDX
ACTIONS_PATH = Path("experiments/exp110_mpc/checkpoints/actions_5k_final.npz")
N_ROUTES = 5000

actions_data = dict(np.load(ACTIONS_PATH))
print(f"Loaded {len(actions_data)} routes of MPC actions")


def process_one(data_path):
    fname = Path(data_path).name
    if fname not in actions_data:
        return None

    actions = actions_data[fname]
    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)

    h_act = [0.0] * HIST_LEN
    h_lat = [0.0] * HIST_LEN
    obs_list = []
    raw_list = []

    class Ctrl:
        def __init__(self):
            self.step = 0

        def update(self, target, current, state, future_plan=None):
            self.step += 1
            step_idx = self.step + CONTEXT_LENGTH - 1

            # Update h_lat with current lataccel
            if step_idx < CONTROL_START_IDX:
                h_act.append(0.0)
                h_act.pop(0)
                h_lat.append(target)
                h_lat.pop(0)
                return 0.0

            ci = step_idx - CONTROL_START_IDX
            if ci >= N_CTRL:
                return 0.0

            # Build obs using exp055's exact function
            obs = _build_obs_bc(target, current, state, future_plan, h_act, h_lat)

            # Delta target: MPC action - previous action
            action = float(actions[ci])
            raw = np.clip((action - h_act[-1]) / DELTA_SCALE_MAX, -1.0, 1.0)

            obs_list.append(obs)
            raw_list.append(np.float32(raw))

            # Update histories
            h_act.append(action)
            h_act.pop(0)
            h_lat.append(current)
            h_lat.pop(0)
            return action

    ctrl = Ctrl()
    sim = TinyPhysicsSimulator(model, str(data_path), ctrl)
    cost = sim.rollout()

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(raw_list, dtype=np.float32),
        cost["total_cost"],
    )


if __name__ == "__main__":
    data_dir = Path("./data")
    files = sorted(data_dir.iterdir())[:N_ROUTES]
    print(f"Processing {len(files)} routes...")

    from tqdm.contrib.concurrent import process_map

    results = process_map(process_one, files, max_workers=16, chunksize=50)

    results = [r for r in results if r is not None]
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    costs = [r[2] for r in results]

    print(f"Collected {len(all_obs)} (obs, raw) pairs from {len(results)} routes")
    print(f"Mean cost: {np.mean(costs):.1f}")
    print(f"Obs shape: {all_obs.shape}, Raw shape: {all_raw.shape}")

    save_path = "experiments/exp110_mpc/checkpoints/bc_data.npz"
    np.savez(save_path, obs=all_obs, raw=all_raw)
    print(f"Saved to {save_path}")
