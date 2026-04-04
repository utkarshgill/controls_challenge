#!/usr/bin/env python3
"""MPPI v3: Receding-horizon with ALL windows batched into one rollout.

Same logic as working mppi.py but ~10-40× faster:
- v1: 40 sequential rollouts of NK=2560 per iteration (216s)
- v3: 1 rollout of NK = N × K × W windows (5-30s depending on K)

Each "virtual trajectory" has the same base actions except one HORIZON-sized
window is perturbed. Full trajectory cost is correct for all candidates.

Usage:
  python experiments/exp110_mpc/mppi3.py
"""

import numpy as np, os, sys, time
from pathlib import Path
from hashlib import md5

sys.path.insert(0, ".")
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
)
from tinyphysics_batched import BatchedSimulator, preload_csvs, make_ort_session

N_ROUTES = int(os.getenv("N_ROUTES", "10"))
ROUTE_START = int(os.getenv("ROUTE_START", "0"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "10"))
K_SAMPLES = int(os.getenv("K_SAMPLES", "64"))
ITERS = int(os.getenv("ITERS", "20"))
HORIZON = int(os.getenv("HORIZON", "10"))
SIGMA = float(os.getenv("SIGMA", "0.05"))
SIGMA_DECAY = float(os.getenv("SIGMA_DECAY", "0.95"))
WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")
MODEL_PATH = "models/tinyphysics.onnx"

SAVE_DIR = Path(
    os.getenv(
        "SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints" / "mppi3")
    )
)

N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

_ort = None


def get_ort():
    global _ort
    if _ort is None:
        _ort = make_ort_session(MODEL_PATH)
    return _ort


def mppi_refine(csv_files, init_actions):
    """Receding-horizon MPPI with all windows batched.

    For each iteration:
      1. Build action arrays for all (route, window, candidate) combos
      2. Evaluate ALL in one batched rollout
      3. For each route, scan windows and greedily accept improvements
    """
    N = len(csv_files)
    best_actions = init_actions.copy()
    ort_session = get_ort()

    # Preload data and RNG once
    base_data = preload_csvs([str(f) for f in csv_files])
    T = base_data["T"]
    n_steps = T - CONTEXT_LENGTH
    rng_base = np.empty((N, n_steps), dtype=np.float64)
    seed_prefix = os.getenv("SEED_PREFIX", "data")
    for i, f in enumerate(csv_files):
        seed_str = f"{seed_prefix}/{Path(f).name}"
        seed = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
        rng_base[i, :] = np.random.RandomState(seed).rand(n_steps)

    windows = list(range(0, N_CTRL, HORIZON))
    W = len(windows)
    K = K_SAMPLES
    NK_total = N * W * K  # total virtual trajectories

    print(f"    Windows={W}, K={K}, NK_total={NK_total}", flush=True)

    # Initial eval (just base, no perturbations)
    t0 = time.time()
    tiled_csvs_init = [str(f) for f in csv_files]
    tiled_data_init = base_data
    tiled_rng_init = rng_base
    sim_init = BatchedSimulator(
        MODEL_PATH,
        csv_files=tiled_csvs_init,
        ort_session=ort_session,
        cached_data=tiled_data_init,
        cached_rng=tiled_rng_init,
    )
    gpu = sim_init._gpu
    if gpu:
        import torch

        init_actions_t = torch.from_numpy(np.ascontiguousarray(best_actions)).cuda()

    def init_ctrl(step_idx, sim_ref):
        ai = step_idx - CONTROL_START_IDX
        if 0 <= ai < N_CTRL:
            return init_actions_t[:, ai] if gpu else best_actions[:, ai].copy()
        return (
            torch.zeros(N, dtype=torch.float64, device="cuda") if gpu else np.zeros(N)
        )

    init_costs = sim_init.rollout(init_ctrl)["total_cost"]
    best_costs = init_costs.copy()
    dt = time.time() - t0
    print(
        f"    Init: mean={best_costs.mean():.2f} min={best_costs.min():.1f} "
        f"max={best_costs.max():.1f} ({dt:.1f}s)",
        flush=True,
    )

    # Pre-tile CSV data and RNG for the big batch: N routes × W windows × K candidates
    # Layout: for route r, window w, candidate k → index = r*(W*K) + w*K + k
    tiled_csvs = []
    for f in csv_files:
        tiled_csvs.extend([str(f)] * (W * K))
    tiled_data = {"T": T, "N": NK_total}
    for key in ("roll_lataccel", "v_ego", "a_ego", "target_lataccel", "steer_command"):
        tiled_data[key] = np.repeat(base_data[key], W * K, axis=0)
    tiled_rng = np.repeat(rng_base, W * K, axis=0)

    for it in range(ITERS):
        t0 = time.time()
        sigma = SIGMA * (SIGMA_DECAY**it)

        # Build the full (NK_total, 400) action array
        # Start from base actions tiled
        flat_actions = np.tile(best_actions, (1, W * K)).reshape(NK_total, N_CTRL)
        # Actually need: for each (route r, window w, candidate k):
        #   actions = best_actions[r] with window w perturbed by candidate k
        # flat_actions[r*(W*K) + w*K + k, :] = best_actions[r, :]
        # flat_actions[r*(W*K) + w*K + k, win_start:win_end] = perturbed

        # Build it properly
        for r in range(N):
            base = best_actions[r]  # (400,)
            for wi, win_start in enumerate(windows):
                win_end = min(win_start + HORIZON, N_CTRL)
                win_len = win_end - win_start
                offset = r * (W * K) + wi * K

                # All K candidates start from base
                flat_actions[offset : offset + K, :] = base[None, :]

                # Candidate 0 = elitism (base, no perturbation)
                # Candidates 1..K-1 = perturbed window
                noise = np.random.randn(K - 1, win_len) * sigma
                flat_actions[offset + 1 : offset + K, win_start:win_end] = np.clip(
                    base[None, win_start:win_end] + noise,
                    STEER_RANGE[0],
                    STEER_RANGE[1],
                )

        # One big rollout
        sim = BatchedSimulator(
            MODEL_PATH,
            csv_files=tiled_csvs,
            ort_session=ort_session,
            cached_data=tiled_data,
            cached_rng=tiled_rng,
        )
        if gpu:
            flat_actions_t = torch.from_numpy(np.ascontiguousarray(flat_actions)).cuda()

        def ctrl_fn(step_idx, sim_ref):
            ai = step_idx - CONTROL_START_IDX
            if 0 <= ai < N_CTRL:
                return flat_actions_t[:, ai] if gpu else flat_actions[:, ai].copy()
            if gpu:
                return torch.zeros(NK_total, dtype=torch.float64, device="cuda")
            return np.zeros(NK_total)

        all_costs = sim.rollout(ctrl_fn)["total_cost"]  # (NK_total,)
        costs_3d = all_costs.reshape(N, W, K)  # (N, W, K)

        # For each route, scan windows and accept if any candidate beats
        # the elitism (candidate 0 = base actions) IN THE SAME BATCH.
        # This avoids NK-dependent numerical mismatches between init and batch evals.
        improved = 0
        for r in range(N):
            for wi, win_start in enumerate(windows):
                win_end = min(win_start + HORIZON, N_CTRL)
                window_costs = costs_3d[r, wi, :]  # (K,)
                elitism_cost = window_costs[0]  # base actions, same batch
                best_k = window_costs[1:].argmin() + 1  # best non-elitism
                if window_costs[best_k] < elitism_cost:
                    idx = r * (W * K) + wi * K + best_k
                    best_actions[r, win_start:win_end] = flat_actions[
                        idx, win_start:win_end
                    ]
                    improved += 1

        # Re-evaluate with updated actions to get true costs
        sim_re = BatchedSimulator(
            MODEL_PATH,
            csv_files=[str(f) for f in csv_files],
            ort_session=ort_session,
            cached_data=base_data,
            cached_rng=rng_base,
        )
        if gpu:
            re_actions_t = torch.from_numpy(np.ascontiguousarray(best_actions)).cuda()

        def re_ctrl(step_idx, sim_ref):
            ai = step_idx - CONTROL_START_IDX
            if 0 <= ai < N_CTRL:
                return re_actions_t[:, ai] if gpu else best_actions[:, ai].copy()
            return (
                torch.zeros(N, dtype=torch.float64, device="cuda")
                if gpu
                else np.zeros(N)
            )

        best_costs = sim_re.rollout(re_ctrl)["total_cost"]

        dt = time.time() - t0
        print(
            f"    iter {it + 1:3d} | mean={best_costs.mean():.2f} "
            f"min={best_costs.min():.1f} max={best_costs.max():.1f} | "
            f"σ={sigma:.4f} | improved={improved} | {dt:.0f}s",
            flush=True,
        )

    return best_actions, best_costs


def main():
    print(
        f"MPPI v3: routes={N_ROUTES} batch={BATCH_ROUTES} K={K_SAMPLES} "
        f"iters={ITERS} H={HORIZON} σ={SIGMA}",
        flush=True,
    )

    all_csv = sorted(Path("data").glob("*.csv"))[: max(N_ROUTES + ROUTE_START, 1)]
    my_csv = all_csv[ROUTE_START : ROUTE_START + N_ROUTES]

    warm = {}
    if WARM_START_NPZ and Path(WARM_START_NPZ).exists():
        wd = np.load(WARM_START_NPZ)
        warm = {k: wd[k] for k in wd.files}
        print(f"Loaded warm-start: {len(warm)} routes", flush=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    actions_dict = {}
    all_costs = []
    done_set = set()
    existing = SAVE_DIR / "actions.npz"
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

        init_actions = np.zeros((N, N_CTRL), dtype=np.float64)
        for i, f in enumerate(batch_todo):
            if f.name in warm:
                init_actions[i] = warm[f.name][:N_CTRL]

        best_acts, best_costs_batch = mppi_refine(
            [str(f) for f in batch_todo], init_actions
        )

        for i, f in enumerate(batch_todo):
            actions_dict[f.name] = best_acts[i]
            all_costs.append(best_costs_batch[i])
            done_set.add(f.name)

        np.savez(SAVE_DIR / "actions.npz", **actions_dict)
        print(
            f"  running mean={np.mean(all_costs):.2f} ({len(all_costs)} routes)",
            flush=True,
        )

    dt = time.time() - t0_total
    print(f"\n{'=' * 60}")
    print(f"Done: {len(all_costs)} routes, mean={np.mean(all_costs):.2f}")
    print(f"Saved to {SAVE_DIR / 'actions.npz'}")
    print(f"Total: {dt:.0f}s ({dt / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
