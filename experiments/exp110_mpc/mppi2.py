#!/usr/bin/env python3
"""MPPI v2: Receding-horizon with forked rollouts.

Key speedup: instead of re-simulating all 400 steps per window, fork from a
cached base trajectory snapshot and only simulate HORIZON + LOOKAHEAD steps.

Algorithm:
  1. Pre-roll base trajectory once (N routes, 400 steps)
  2. Snapshot state at each window boundary [0, HORIZON, 2*HORIZON, ...]
  3. For each window:
     a. Create N×K tiled sim, restore from snapshot at window start
     b. Perturb actions only in [win_start, win_start+HORIZON)
     c. Simulate HORIZON + LOOKAHEAD steps (not full 400)
     d. Score using partial cost (the window region) + tail cost from base
     e. Keep best per route
  4. After all windows, update base trajectory and repeat

Usage:
  python experiments/exp110_mpc/mppi2.py

Env vars:
  N_ROUTES=10  BATCH_ROUTES=10  K_SAMPLES=256  ITERS=10
  HORIZON=10  LOOKAHEAD=20  SIGMA=0.05  SIGMA_DECAY=0.95
  WARM_START_NPZ=...
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
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
    ACC_G,
)
from tinyphysics_batched import BatchedSimulator, preload_csvs, make_ort_session

N_ROUTES = int(os.getenv("N_ROUTES", "10"))
ROUTE_START = int(os.getenv("ROUTE_START", "0"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "10"))
K_SAMPLES = int(os.getenv("K_SAMPLES", "256"))
ITERS = int(os.getenv("ITERS", "10"))
HORIZON = int(os.getenv("HORIZON", "10"))
LOOKAHEAD = int(os.getenv("LOOKAHEAD", "20"))
SIGMA = float(os.getenv("SIGMA", "0.05"))
SIGMA_DECAY = float(os.getenv("SIGMA_DECAY", "0.95"))
WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")
MODEL_PATH = "models/tinyphysics.onnx"

SAVE_DIR = Path(
    os.getenv(
        "SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints" / "mppi2")
    )
)

N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400
CL = CONTEXT_LENGTH  # 20

_ort = None


def get_ort():
    global _ort
    if _ort is None:
        _ort = make_ort_session(MODEL_PATH)
    return _ort


def preroll_base(csv_files, actions, ort_session):
    """Roll out base trajectory, return snapshots at each window boundary.

    csv_files: list of N file paths
    actions: (N, 400) base actions
    Returns: (base_costs (N,), snapshots dict {step_idx: snapshot})
    """
    N = len(csv_files)
    data = preload_csvs(csv_files)
    T = data["T"]

    # Precompute RNG
    n_steps = T - CL
    rng_all = np.empty((N, n_steps), dtype=np.float64)
    seed_prefix = os.getenv("SEED_PREFIX", "data")
    for i, f in enumerate(csv_files):
        seed_str = f"{seed_prefix}/{Path(f).name}"
        seed = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
        rng_all[i, :] = np.random.RandomState(seed).rand(n_steps)

    sim = BatchedSimulator(
        MODEL_PATH,
        csv_files=csv_files,
        ort_session=ort_session,
        cached_data=data,
        cached_rng=rng_all,
    )
    gpu = sim._gpu
    if gpu:
        import torch

        actions_t = torch.from_numpy(np.ascontiguousarray(actions)).cuda()

    # Snapshot at window boundaries
    snapshots = {}
    # Need snapshots at step_idx = CONTROL_START_IDX + win_start for each window
    window_starts_ctrl = list(range(0, N_CTRL, HORIZON))  # in action-space
    snap_steps = set(CONTROL_START_IDX + ws for ws in window_starts_ctrl)

    # Also snapshot right before control starts (for the initial state)
    snap_steps.add(CONTROL_START_IDX)

    def controller_fn(step_idx, sim_ref):
        action_idx = step_idx - CONTROL_START_IDX
        if 0 <= action_idx < N_CTRL:
            if gpu:
                return actions_t[:, action_idx]
            return actions[:, action_idx].copy()
        if gpu:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        return np.zeros(N, dtype=np.float64)

    # Run step by step, capturing snapshots
    dg = sim.data_gpu if gpu else None
    for step_idx in range(CL, sim.T):
        if step_idx in snap_steps:
            snapshots[step_idx] = sim.snapshot()

        acts = controller_fn(step_idx, sim)
        h = sim._hist_len
        if gpu:
            sim.state_history[:, h, 0] = dg["roll_lataccel"][:, step_idx]
            sim.state_history[:, h, 1] = dg["v_ego"][:, step_idx]
            sim.state_history[:, h, 2] = dg["a_ego"][:, step_idx]
        else:
            d = sim.data
            sim.state_history[:, h, 0] = d["roll_lataccel"][:, step_idx]
            sim.state_history[:, h, 1] = d["v_ego"][:, step_idx]
            sim.state_history[:, h, 2] = d["a_ego"][:, step_idx]
        sim.control_step(step_idx, acts)
        sim.sim_step(step_idx)

    costs = sim.compute_cost()
    return costs["total_cost"], snapshots, data, rng_all


def evaluate_window(
    csv_files,
    base_actions,
    perturbed_window,
    win_start_ctrl,
    snapshot,
    data,
    rng_all,
    ort_session,
):
    """Evaluate K perturbation candidates for one window position.

    base_actions: (N, 400)
    perturbed_window: (N, K, HORIZON) — perturbed values for this window
    win_start_ctrl: 0-based index in action space (0, 10, 20, ...)
    snapshot: sim state at CONTROL_START_IDX + win_start_ctrl

    Returns: (N, K) costs
    """
    N = len(csv_files)
    K = perturbed_window.shape[1]
    NK = N * K

    # Build tiled CSV list: each route repeated K times
    tiled_csvs = []
    for f in csv_files:
        tiled_csvs.extend([str(f)] * K)

    # Tile the data arrays
    tiled_data = {}
    T = data["T"]
    for key in ("roll_lataccel", "v_ego", "a_ego", "target_lataccel", "steer_command"):
        arr = data[key]  # (N, T)
        tiled_data[key] = np.repeat(arr, K, axis=0)  # (NK, T)
    tiled_data["N"] = NK
    tiled_data["T"] = T

    # Tile RNG
    tiled_rng = np.repeat(rng_all, K, axis=0)  # (NK, n_steps)

    # Build full action arrays: base actions with window replaced
    full_actions = np.tile(base_actions, (1, 1))  # (N, 400) — copy
    # For each candidate k, replace the window
    # Shape: (N, K, 400)
    all_actions = np.broadcast_to(base_actions[:, None, :], (N, K, N_CTRL)).copy()
    win_end_ctrl = min(win_start_ctrl + HORIZON, N_CTRL)
    all_actions[:, :, win_start_ctrl:win_end_ctrl] = perturbed_window
    all_actions[:, 0, :] = base_actions  # elitism: candidate 0 = base
    flat_actions = all_actions.reshape(NK, N_CTRL)

    # Create tiled sim from snapshot
    sim = BatchedSimulator(
        MODEL_PATH,
        csv_files=tiled_csvs,
        ort_session=ort_session,
        cached_data=tiled_data,
        cached_rng=tiled_rng,
    )

    # Restore from tiled snapshot
    gpu = sim._gpu
    if gpu:
        import torch

        tiled_snap = {}
        for key in ("action_history", "state_history", "current_lataccel_history"):
            tiled_snap[key] = snapshot[key].repeat_interleave(K, dim=0)
        tiled_snap["current_lataccel"] = snapshot["current_lataccel"].repeat_interleave(
            K, dim=0
        )
        tiled_snap["_hist_len"] = snapshot["_hist_len"]
        sim.restore(tiled_snap)
        flat_actions_t = torch.from_numpy(np.ascontiguousarray(flat_actions)).cuda()
    else:
        tiled_snap = {}
        for key in ("action_history", "state_history", "current_lataccel_history"):
            tiled_snap[key] = np.repeat(snapshot[key], K, axis=0)
        tiled_snap["current_lataccel"] = np.repeat(
            snapshot["current_lataccel"], K, axis=0
        )
        tiled_snap["_hist_len"] = snapshot["_hist_len"]
        sim.restore(tiled_snap)

    # Only simulate from window start to min(window_end + LOOKAHEAD, T)
    step_start = CONTROL_START_IDX + win_start_ctrl
    step_end = min(CONTROL_START_IDX + win_end_ctrl + LOOKAHEAD, sim.T)

    def controller_fn(step_idx, sim_ref):
        action_idx = step_idx - CONTROL_START_IDX
        if 0 <= action_idx < N_CTRL:
            if gpu:
                return flat_actions_t[:, action_idx]
            return flat_actions[:, action_idx].copy()
        if gpu:
            return torch.zeros(NK, dtype=torch.float64, device="cuda")
        return np.zeros(NK, dtype=np.float64)

    sim.rollout_from(step_start, step_end, controller_fn)

    # Compute cost only over the simulated window for ranking.
    # The prefix (before step_start) is identical across candidates (from snapshot).
    # The suffix (after step_end) hasn't been simulated.
    # For ranking, we only need cost in [step_start, step_end).
    # But clamp to the official cost window [CONTROL_START_IDX, COST_END_IDX).
    cost_lo = max(step_start, CONTROL_START_IDX)
    cost_hi = min(step_end, COST_END_IDX)
    if cost_hi <= cost_lo:
        return np.zeros((N, K))

    if gpu:
        import torch as _t

        target = sim.data_gpu["target_lataccel"][:, cost_lo:cost_hi]
        pred = sim.current_lataccel_history[:, cost_lo:cost_hi]
        lat_cost = (target - pred).pow(2).mean(dim=1) * 100
        if cost_hi - cost_lo > 1:
            jerk = _t.diff(pred, dim=1) / DEL_T
            jerk_cost = jerk.pow(2).mean(dim=1) * 100
        else:
            jerk_cost = _t.zeros(NK, device="cuda")
        total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return total.cpu().numpy().reshape(N, K)
    else:
        target = sim.data["target_lataccel"][:, cost_lo:cost_hi]
        pred = sim.current_lataccel_history[:, cost_lo:cost_hi]
        lat_cost = np.mean((target - pred) ** 2, axis=1) * 100
        if cost_hi - cost_lo > 1:
            jerk_cost = np.mean((np.diff(pred, axis=1) / DEL_T) ** 2, axis=1) * 100
        else:
            jerk_cost = np.zeros(NK)
        total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return total.reshape(N, K)


def mppi_refine(csv_files, init_actions):
    """Run receding-horizon MPPI with forked rollouts."""
    N = len(csv_files)
    best_actions = init_actions.copy()
    ort_session = get_ort()

    # Pre-roll base
    t0 = time.time()
    base_costs, snapshots, data, rng_all = preroll_base(
        csv_files, best_actions, ort_session
    )
    best_costs = base_costs.copy()
    dt = time.time() - t0

    print(
        f"    Init: mean={base_costs.mean():.2f} min={base_costs.min():.1f} "
        f"max={base_costs.max():.1f} ({dt:.1f}s)",
        flush=True,
    )

    for it in range(ITERS):
        t0 = time.time()
        sigma = SIGMA * (SIGMA_DECAY**it)
        improved = 0

        window_starts = list(range(0, N_CTRL, HORIZON))

        # Sweep windows sequentially: accept greedily, update actions in place.
        # Snapshots are from start of iteration — slightly stale for later
        # windows, but the greedy single-window changes are small enough
        # that the snapshot remains a good approximation.
        candidate_actions = best_actions.copy()

        for win_start in window_starts:
            win_end = min(win_start + HORIZON, N_CTRL)
            win_len = win_end - win_start

            snap_step = CONTROL_START_IDX + win_start
            if snap_step not in snapshots:
                continue

            # Generate perturbations around CURRENT candidate actions
            noise = np.random.randn(N, K_SAMPLES, win_len) * sigma
            perturbed = np.clip(
                candidate_actions[:, None, win_start:win_end] + noise,
                STEER_RANGE[0],
                STEER_RANGE[1],
            )

            costs = evaluate_window(
                csv_files,
                candidate_actions,  # current (may have earlier window changes)
                perturbed,
                win_start,
                snapshots[snap_step],
                data,
                rng_all,
                ort_session,
            )

            # Candidate 0 = current actions (elitism). Accept if better.
            min_idx = costs.argmin(axis=1)
            for i in range(N):
                if min_idx[i] > 0 and costs[i, min_idx[i]] < costs[i, 0]:
                    candidate_actions[i, win_start:win_end] = perturbed[i, min_idx[i]]
                    improved += 1

        # Re-roll full trajectory with candidate actions to get TRUE cost
        cand_costs, snapshots_new, data, rng_all = preroll_base(
            csv_files, candidate_actions, ort_session
        )

        # Accept per route only if full-trajectory cost improved
        accepted = 0
        for i in range(N):
            if cand_costs[i] < best_costs[i]:
                best_costs[i] = cand_costs[i]
                best_actions[i] = candidate_actions[i]
                accepted += 1

        # Use new snapshots for next iteration
        if accepted > 0:
            snapshots = snapshots_new
        # If nothing accepted, keep old snapshots (unchanged base)

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
        f"MPPI v2: routes={N_ROUTES} batch={BATCH_ROUTES} K={K_SAMPLES} "
        f"iters={ITERS} H={HORIZON} LA={LOOKAHEAD} σ={SIGMA}",
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
