#!/usr/bin/env python3
"""exp110 MPC worker — runs on a single GPU for a slice of routes.
Called by mpc_multi.py. Reads ROUTE_START, N_ROUTES, SAVE_DIR from env."""

import numpy as np, os, sys, time, torch, torch.nn.functional as tF
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
)
from tinyphysics_batched import (
    BatchedSimulator,
    BatchedPhysicsModel,
    CSVCache,
    make_ort_session,
)
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
)

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH
BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)

ROUTE_START = int(os.getenv("ROUTE_START", "0"))
N_ROUTES_END = int(os.getenv("N_ROUTES", "5000"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "100"))
MPC_K = int(os.getenv("MPC_K", "64"))
MPC_H = int(os.getenv("MPC_H", "50"))
MPC_W = int(os.getenv("MPC_W", "20"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "10"))
CEM_SIGMA = float(os.getenv("CEM_SIGMA", "0.1"))
CEM_ELITE_FRAC = float(os.getenv("CEM_ELITE", "0.15"))
N_BASIS_W = int(os.getenv("N_BASIS_W", "8"))
SAVE_DIR = Path(
    os.getenv("SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints"))
)


# Import the core MPC function from mpc.py
from experiments.exp110_mpc.mpc import FastMPCPredictor, run_mpc_batch


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    # Load all routes, then slice
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES_END]
    my_csv = all_csv[ROUTE_START:N_ROUTES_END]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Load policy once
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "?")
    n_my_routes = len(my_csv)
    print(
        f"Worker GPU={gpu_id}: {n_my_routes} routes [{ROUTE_START}:{N_ROUTES_END}]",
        flush=True,
    )

    # Load warm-start from previous run if available
    warm_npz = os.getenv("WARM_START_NPZ", "")
    warm_actions_dict = None
    if warm_npz and Path(warm_npz).exists():
        warm_data = np.load(warm_npz)
        warm_actions_dict = {k: warm_data[k] for k in warm_data.files}
        print(f"  Loaded warm-start: {len(warm_actions_dict)} routes", flush=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    actions_dict = {}
    all_costs = []
    mpc_phys_cached = None
    mpc_ort_cached = make_ort_session(mdl_path)

    # Resume: load already-completed routes from previous run
    existing_npz = SAVE_DIR / "actions.npz"
    done_set = set()
    if existing_npz.exists():
        prev = np.load(existing_npz)
        for k in prev.files:
            actions_dict[k] = prev[k]
            done_set.add(k)
        print(f"  Resuming: {len(done_set)} routes already done", flush=True)

    for batch_start_rel in range(0, n_my_routes, BATCH_ROUTES):
        batch_end_rel = min(batch_start_rel + BATCH_ROUTES, n_my_routes)
        batch_csv_full = my_csv[batch_start_rel:batch_end_rel]
        # Skip routes already done
        batch_csv = [f for f in batch_csv_full if f.name not in done_set]
        if not batch_csv:
            print(
                f"  Skipping batch {batch_start_rel}-{batch_end_rel} (all done)",
                flush=True,
            )
            continue
        batch_start_abs = ROUTE_START + batch_start_rel
        batch_end_abs = ROUTE_START + batch_end_rel

        t0 = time.time()
        actions, costs, mpc_phys_cached = run_mpc_batch(
            batch_csv,
            mdl_path,
            ort_sess,
            csv_cache,
            ac,
            ds,
            mpc_phys=mpc_phys_cached,
            mpc_ort=mpc_ort_cached,
            warm_actions_dict=warm_actions_dict,
        )
        dt = time.time() - t0

        actions_np = actions.cpu().numpy()
        for i, f in enumerate(batch_csv):
            actions_dict[f.name] = actions_np[i]
            all_costs.append(costs[i])

        # Save incrementally
        np.savez(SAVE_DIR / "actions.npz", **actions_dict)
        running_mean = np.mean(all_costs)
        print(
            f"  running mean={running_mean:.1f}  ({len(all_costs)}/{n_my_routes} routes)  ⏱{dt:.0f}s",
            flush=True,
        )

    print(
        f"Worker GPU={gpu_id} done: {len(all_costs)} routes, mean={np.mean(all_costs):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
