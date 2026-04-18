#!/usr/bin/env python3
"""Collect SteerGPT training data using official tinyphysics.py with monkey-patching.

Uses the same process_map(16 workers) as the official eval command, just patches
run_rollout to also return the sim histories.

Usage: .venv/bin/python experiments/exp111_steergpt/collect_data.py
"""

import sys, os, numpy as np, time
from pathlib import Path
from functools import partial

sys.path.insert(0, ".")

import tinyphysics
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, STEER_RANGE,
)
from controllers import BaseController
from tqdm.contrib.concurrent import process_map

N_ROUTES = int(os.getenv("N_ROUTES", "5000"))
T_COLLECT = COST_END_IDX  # 500

ACTIONS_PATH = Path("experiments/exp110_mpc/checkpoints/actions_5k_v2.npz")
SAVE_PATH = Path("experiments/exp111_steergpt/data/steergpt_data.npz")
MODEL_PATH = "models/tinyphysics.onnx"

# Load MPC actions globally so workers inherit via fork
_ACTIONS = dict(np.load(ACTIONS_PATH))


class ReplayController(BaseController):
    def __init__(self, actions):
        self.actions = actions
        self.call = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.call += 1
        action_idx = self.call - 81  # step 100 = call 81
        if 0 <= action_idx < len(self.actions):
            return float(self.actions[action_idx])
        return 0.0


def run_one(data_path):
    """Run one route, return cost + histories."""
    fname = Path(data_path).name
    actions = _ACTIONS.get(fname)
    if actions is None:
        return None

    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ReplayController(actions)
    sim = TinyPhysicsSimulator(model, str(data_path), controller, debug=False)
    cost = sim.rollout()

    T = min(len(sim.action_history), T_COLLECT)
    states = np.array([(s.roll_lataccel, s.v_ego, s.a_ego)
                       for s in sim.state_history[:T]])
    return {
        "cost": cost["total_cost"],
        "steer": np.array(sim.action_history[:T], dtype=np.float32),
        "current_la": np.array(sim.current_lataccel_history[:T], dtype=np.float32),
        "target_la": np.array(sim.target_lataccel_history[:T], dtype=np.float32),
        "roll_la": states[:, 0].astype(np.float32),
        "v_ego": states[:, 1].astype(np.float32),
        "a_ego": states[:, 2].astype(np.float32),
    }


def main():
    print(f"Loaded {len(_ACTIONS)} MPC action sets", flush=True)

    csv_files = sorted(Path("data").glob("*.csv"))[:N_ROUTES]
    print(f"Running {len(csv_files)} routes with 16 workers...", flush=True)

    t0 = time.time()
    results = process_map(run_one, csv_files, max_workers=16, chunksize=10)

    # Collect results
    valid = [r for r in results if r is not None]
    N = len(valid)
    T = T_COLLECT
    print(f"\n{N} routes completed, {len(results) - N} skipped", flush=True)

    steer = np.zeros((N, T), dtype=np.float32)
    current_la = np.zeros((N, T), dtype=np.float32)
    target_la = np.zeros((N, T), dtype=np.float32)
    roll_la = np.zeros((N, T), dtype=np.float32)
    v_ego = np.zeros((N, T), dtype=np.float32)
    a_ego = np.zeros((N, T), dtype=np.float32)
    costs = []

    for i, r in enumerate(valid):
        L = min(len(r["steer"]), T)
        steer[i, :L] = r["steer"][:L]
        current_la[i, :L] = r["current_la"][:L]
        target_la[i, :L] = r["target_la"][:L]
        roll_la[i, :L] = r["roll_la"][:L]
        v_ego[i, :L] = r["v_ego"][:L]
        a_ego[i, :L] = r["a_ego"][:L]
        costs.append(r["cost"])

    delta = np.diff(steer, axis=1)
    print(f"Delta steer: std={delta.std():.6f} p99={np.percentile(np.abs(delta), 99):.6f}")
    print(f"Mean cost: {np.mean(costs):.2f}")

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAVE_PATH, steer=steer, current_la=current_la,
                        target_la=target_la, roll_la=roll_la, v_ego=v_ego, a_ego=a_ego)

    elapsed = time.time() - t0
    print(f"Saved to {SAVE_PATH} ({SAVE_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
