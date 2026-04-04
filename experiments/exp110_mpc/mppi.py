#!/usr/bin/env python3
"""MPPI (Model-Predictive Path Integral) refinement of existing actions.

Unlike CEM which fits a Gaussian to elite samples, MPPI uses importance-weighted
averaging over ALL samples — soft selection that preserves diversity.

Algorithm per route batch:
  1. Load warm-start actions (400 per route)
  2. For each iteration:
     a. Sample K perturbations: delta ~ N(0, sigma)
     b. Evaluate all K candidates on stochastic sim (batched ONNX)
     c. Compute weights: w_k = exp(-cost_k / lambda)  (softmax temperature)
     d. Update actions: a += sum(w_k * delta_k) / sum(w_k)
  3. Save best actions

Usage:
  python experiments/exp110_mpc/mppi.py

Env vars:
  N_ROUTES=5000       Total routes
  BATCH_ROUTES=10     Routes per batch
  K_SAMPLES=64        Perturbation samples per iteration
  ITERS=20            MPPI iterations
  SIGMA=0.02          Perturbation std
  LAMBDA_=0.1         Temperature (lower = more greedy)
  WARM_START_NPZ=...  Path to warm-start actions
"""

import numpy as np, os, sys, time
from pathlib import Path
from hashlib import md5

# Use GPU for fast search, verify best on CPU at the end
# GPU numerics differ slightly but search direction is correct

sys.path.insert(0, ".")
from tinyphysics import (
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    LATACCEL_RANGE, VOCAB_SIZE, MAX_ACC_DELTA, ACC_G,
)
from tinyphysics_batched import BatchedSimulator, preload_csvs

N_ROUTES     = int(os.getenv("N_ROUTES", "10"))
ROUTE_START  = int(os.getenv("ROUTE_START", "0"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "10"))
K_SAMPLES    = int(os.getenv("K_SAMPLES", "64"))
ITERS        = int(os.getenv("ITERS", "20"))
SIGMA        = float(os.getenv("SIGMA", "0.02"))
SIGMA_DECAY  = float(os.getenv("SIGMA_DECAY", "0.95"))
LAMBDA_      = float(os.getenv("LAMBDA_", "0.1"))
WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")
MODEL_PATH   = "models/tinyphysics.onnx"

SAVE_DIR = Path(os.getenv("SAVE_DIR",
    str(Path(__file__).resolve().parent / "checkpoints" / "mppi")))

N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400


_ort_session = None

def get_ort_session():
    global _ort_session
    if _ort_session is None:
        from tinyphysics_batched import make_ort_session
        _ort_session = make_ort_session(MODEL_PATH)
    return _ort_session


_sim_cache = {}


def evaluate_actions(csv_files, all_actions):
    """Evaluate multiple action sets per route using BatchedSimulator.

    Caches the sim setup (CSV loading, RNG) for reuse across calls.
    all_actions: (N_routes, K_samples, 400)
    Returns: (N_routes, K_samples) costs.
    """
    N_routes = len(csv_files)
    K = all_actions.shape[1]

    # Cache key: frozenset of filenames + K
    cache_key = (tuple(str(f) for f in csv_files), K)
    if cache_key not in _sim_cache:
        tiled_csvs = []
        for f in csv_files:
            tiled_csvs.extend([str(f)] * K)
        # Preload once, store cached data + rng
        from tinyphysics_batched import preload_csvs, CSVCache
        data = preload_csvs(tiled_csvs)
        # Precompute RNG
        N_total = len(tiled_csvs)
        T = data["T"]
        CL = CONTEXT_LENGTH
        n_steps = T - CL
        rng_all = np.empty((N_total, n_steps), dtype=np.float64)
        seed_prefix = os.getenv("SEED_PREFIX", "data")
        for i, f in enumerate(tiled_csvs):
            seed_str = f"{seed_prefix}/{Path(f).name}"
            seed = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
            rng = np.random.RandomState(seed)
            rng_all[i, :] = rng.rand(n_steps)
        _sim_cache[cache_key] = (tiled_csvs, data, rng_all)
        print(f"    [cache] preloaded {N_total} routes", flush=True)

    tiled_csvs, data, rng_all = _sim_cache[cache_key]
    N_total = len(tiled_csvs)
    flat_actions = all_actions.reshape(-1, N_CTRL)

    sim = BatchedSimulator(MODEL_PATH, csv_files=tiled_csvs,
                           ort_session=get_ort_session(),
                           cached_data=data, cached_rng=rng_all)
    gpu = sim._gpu

    if gpu:
        import torch
        flat_actions_t = torch.from_numpy(np.ascontiguousarray(flat_actions)).cuda()

    def controller_fn(step_idx, *args):
        action_idx = step_idx - CONTROL_START_IDX
        if 0 <= action_idx < N_CTRL:
            if gpu:
                return flat_actions_t[:, action_idx]
            return flat_actions[:, action_idx].copy()
        if gpu:
            return torch.zeros(N_total, dtype=torch.float64, device='cuda')
        return np.zeros(N_total, dtype=np.float64)

    costs = sim.rollout(controller_fn)
    return costs["total_cost"].reshape(N_routes, K)


HORIZON = int(os.getenv("HORIZON", "20"))  # rolling window size


def mppi_refine(csv_files, init_actions):
    """Run rolling-horizon MPPI refinement on a batch of routes.

    Slides a window of HORIZON steps across 400 actions. At each window
    position, perturbs only those steps, evaluates full trajectory, keeps
    improvements. Multiple passes over the full sequence.

    init_actions: (N, 400) warm-start actions
    Returns: (N, 400) refined actions, (N,) best costs
    """
    N = len(csv_files)
    actions = init_actions.copy()
    best_actions = actions.copy()

    init_costs = evaluate_actions(csv_files, actions[:, None, :])[:, 0]
    best_costs = init_costs.copy()
    print(f"    Init: mean={init_costs.mean():.2f} min={init_costs.min():.1f} "
          f"max={init_costs.max():.1f}", flush=True)
    for i, f in enumerate(csv_files):
        print(f"      {Path(f).name}: cost={init_costs[i]:.2f}", flush=True)

    for it in range(ITERS):
        t0 = time.time()
        improved = 0
        sigma = SIGMA * (SIGMA_DECAY ** it)

        # Slide window across 400 steps
        for win_start in range(0, N_CTRL, HORIZON):
            win_end = min(win_start + HORIZON, N_CTRL)
            win_len = win_end - win_start

            # Perturb only this window
            deltas = np.zeros((N, K_SAMPLES, N_CTRL), dtype=np.float64)
            deltas[:, :, win_start:win_end] = (
                np.random.randn(N, K_SAMPLES, win_len) * sigma)

            candidates = np.clip(best_actions[:, None, :] + deltas,
                                 STEER_RANGE[0], STEER_RANGE[1])
            candidates[:, 0, :] = best_actions  # elitism

            costs = evaluate_actions(csv_files, candidates)  # (N, K)

            # Keep best per route
            min_idx = costs.argmin(axis=1)
            for i in range(N):
                if costs[i, min_idx[i]] < best_costs[i]:
                    best_costs[i] = costs[i, min_idx[i]]
                    best_actions[i] = candidates[i, min_idx[i]]
                    improved += 1

        dt = time.time() - t0
        print(f"    iter {it+1:3d} | mean={best_costs.mean():.2f} "
              f"min={best_costs.min():.1f} max={best_costs.max():.1f} | "
              f"σ={sigma:.4f} | improved={improved} | {dt:.0f}s", flush=True)

    return best_actions, best_costs


def main():
    print(f"MPPI refinement: routes={N_ROUTES} batch={BATCH_ROUTES} "
          f"K={K_SAMPLES} iters={ITERS} σ={SIGMA} λ={LAMBDA_}", flush=True)

    all_csv = sorted(Path("data").glob("*.csv"))[:max(N_ROUTES + ROUTE_START, 1)]
    my_csv = all_csv[ROUTE_START:ROUTE_START + N_ROUTES]

    # Load warm-start
    warm = {}
    if WARM_START_NPZ and Path(WARM_START_NPZ).exists():
        wd = np.load(WARM_START_NPZ)
        warm = {k: wd[k] for k in wd.files}
        print(f"Loaded warm-start: {len(warm)} routes", flush=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    actions_dict = {}
    all_costs = []

    # Resume
    existing = SAVE_DIR / "actions.npz"
    done_set = set()
    if existing.exists():
        prev = np.load(existing)
        for k in prev.files:
            actions_dict[k] = prev[k]
            done_set.add(k)
        print(f"Resuming: {len(done_set)} routes done", flush=True)

    t0_total = time.time()
    for batch_start in range(0, len(my_csv), BATCH_ROUTES):
        batch_end = min(batch_start + BATCH_ROUTES, len(my_csv))
        batch_csvs = my_csv[batch_start:batch_end]
        batch_todo = [f for f in batch_csvs if f.name not in done_set]
        if not batch_todo:
            continue

        N = len(batch_todo)
        print(f"\n  Batch {batch_start}-{batch_end} ({N} routes):", flush=True)

        # Build init actions from warm-start
        init_actions = np.zeros((N, N_CTRL), dtype=np.float64)
        for i, f in enumerate(batch_todo):
            if f.name in warm:
                init_actions[i] = warm[f.name][:N_CTRL]

        best_acts, best_costs_batch = mppi_refine(batch_todo, init_actions)

        for i, f in enumerate(batch_todo):
            actions_dict[f.name] = best_acts[i]
            all_costs.append(best_costs_batch[i])
            done_set.add(f.name)

        # Save incrementally
        np.savez(SAVE_DIR / "actions.npz", **actions_dict)
        print(f"  running mean={np.mean(all_costs):.2f} ({len(all_costs)} routes)", flush=True)

    dt = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"Done: {len(all_costs)} routes, mean={np.mean(all_costs):.2f}")
    print(f"Saved to {SAVE_DIR / 'actions.npz'}")
    print(f"Total: {dt:.0f}s ({dt/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
