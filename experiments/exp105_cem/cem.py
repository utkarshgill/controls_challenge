#!/usr/bin/env python3
"""exp105 — CEM full-trajectory optimizer

For each route, optimize the full 400-action trajectory using CEM
(Cross-Entropy Method). Warm-start from policy MPC actions.

The sim's RNG is deterministic per segment, so the same action sequence
always produces the same cost. CEM treats the 400-dim action vector as
the search space and iteratively refines it.

Usage:
  CUDA=1 TRT=1 N_ROUTES=10 python experiments/exp105_cem/cem.py
  RESUME=1 ... (loads latest checkpoint)
"""

import numpy as np, os, sys, time, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

# ── Config ──
N_ROUTES = int(os.getenv("N_ROUTES", "10"))
CEM_K = int(os.getenv("CEM_K", "64"))  # candidates per route per iteration
CEM_ELITE = float(os.getenv("CEM_ELITE", "0.1"))  # top fraction to keep
CEM_ITERS = int(os.getenv("CEM_ITERS", "200"))
CEM_SIGMA = float(os.getenv("CEM_SIGMA", "0.1"))  # initial std in basis space
CEM_SIGMA_MIN = float(os.getenv("CEM_SIGMA_MIN", "0.005"))  # min std
N_BASIS = int(os.getenv("N_BASIS", "40"))  # number of smooth basis functions
RESUME = int(os.getenv("RESUME", "0"))
SAVE_DIR = Path(
    os.getenv("SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints"))
)
USE_POLICY_INIT = int(os.getenv("USE_POLICY_INIT", "1"))


def evaluate_actions(csv_files, actions, mdl_path, ort_sess, csv_cache):
    """Evaluate a batch of action trajectories in parallel.

    actions: (N_total, N_CTRL) where N_total = N_routes * K
    csv_files: list of csv files, repeated K times each

    Returns: (N_total,) costs
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        return actions[:, ci].double()

    return sim.rollout(ctrl)["total_cost"]


def get_policy_actions(csv_files, mdl_path, ort_sess, csv_cache):
    """Run the trained policy to get initial action trajectories."""
    from experiments.exp055_batch_of_batch.train import (
        ActorCritic,
        _precompute_future_windows,
        fill_obs,
        HIST_LEN,
        OBS_DIM,
        DELTA_SCALE_MAX,
    )
    import torch.nn.functional as F_

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)
    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1
    stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_err = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_err
        ei = err_sum * (DEL_T / HIST_LEN)
        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")
        fill_obs(
            obs_buf,
            target.float(),
            cur32,
            dg["roll_lataccel"][:, step_idx].float(),
            dg["v_ego"][:, step_idx].float(),
            dg["a_ego"][:, step_idx].float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )
        with torch.no_grad():
            logits = ac.actor(obs_buf)
        a_p = F_.softplus(logits[..., 0]) + 1.0
        b_p = F_.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        ci = step_idx - CONTROL_START_IDX
        if ci < N_CTRL:
            stored[:, ci] = action.float()
        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    return stored, costs


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])
    N = len(all_csv)
    K = CEM_K
    n_elite = max(1, int(K * CEM_ELITE))

    print(f"exp105 — CEM: {N} routes, K={K}, elite={n_elite}, iters={CEM_ITERS}")
    print(f"  σ_init={CEM_SIGMA}, σ_min={CEM_SIGMA_MIN}")

    # Initialize: mean actions from policy or zeros
    if RESUME and (SAVE_DIR / "latest.pt").exists():
        ckpt = torch.load(SAVE_DIR / "latest.pt", weights_only=False, map_location=DEV)
        mean = ckpt["mean"].to(DEV)
        std = ckpt["std"].to(DEV)
        best_actions = ckpt["best_actions"].to(DEV)
        best_costs = ckpt["best_costs"]
        start_iter = ckpt["iter"] + 1
        print(
            f"  Resumed from iter {start_iter - 1}, mean cost={np.mean(best_costs):.1f}"
        )
    else:
        if USE_POLICY_INIT:
            print("  Running policy for warm-start...")
            t0 = time.time()
            policy_actions, policy_costs = get_policy_actions(
                all_csv, mdl_path, ort_sess, csv_cache
            )
            print(
                f"  Policy: mean={np.mean(policy_costs):.1f}  ⏱{time.time() - t0:.0f}s"
            )
            mean = policy_actions.clone()  # (N, N_CTRL)
        else:
            mean = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")

        std = torch.full((N, N_CTRL), CEM_SIGMA, dtype=torch.float32, device="cuda")
        best_actions = mean.clone()
        best_costs = np.full(N, float("inf"))
        start_iter = 0

    # Build smooth cosine basis: (N_BASIS, N_CTRL)
    # Each basis function is a cosine wave of increasing frequency
    t_grid = torch.linspace(0, 1, N_CTRL, device="cuda")
    basis = torch.stack(
        [torch.cos(np.pi * k * t_grid) for k in range(N_BASIS)]
    )  # (N_BASIS, N_CTRL)
    # Normalize each basis function
    basis = basis / basis.norm(dim=1, keepdim=True)
    print(f"  Smooth basis: {N_BASIS} functions over {N_CTRL} steps")

    # CEM in basis coefficient space
    # mean_coeffs: (N, N_BASIS) — coefficients for each route
    # std_coeffs: (N, N_BASIS)
    # actual actions = mean_actions + coeffs @ basis
    mean_coeffs = torch.zeros((N, N_BASIS), dtype=torch.float32, device="cuda")
    std_coeffs = torch.full((N, N_BASIS), CEM_SIGMA, dtype=torch.float32, device="cuda")

    for it in range(start_iter, CEM_ITERS):
        t0 = time.time()

        # Sample K candidate coefficient vectors per route
        noise = torch.randn(N, K, N_BASIS, device="cuda") * std_coeffs.unsqueeze(1)
        cand_coeffs = mean_coeffs.unsqueeze(1) + noise  # (N, K, N_BASIS)
        cand_coeffs[:, 0] = mean_coeffs  # candidate 0 = current mean

        # Convert to action space: mean_actions + coeffs @ basis
        perturbation = torch.einsum("nkb,bt->nkt", cand_coeffs, basis)  # (N, K, N_CTRL)
        candidates = (mean.unsqueeze(1) + perturbation).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        # Flatten and evaluate
        flat_actions = candidates.reshape(N * K, N_CTRL)
        flat_csvs = [f for f in all_csv for _ in range(K)]
        flat_costs = evaluate_actions(
            flat_csvs, flat_actions, mdl_path, ort_sess, csv_cache
        )
        costs_2d = flat_costs.reshape(N, K)

        # Per-route: select elite, update coefficients
        for i in range(N):
            route_costs = costs_2d[i]
            elite_idx = np.argsort(route_costs)[:n_elite]
            elite_coeffs = cand_coeffs[i, elite_idx]  # (n_elite, N_BASIS)

            new_mean_c = elite_coeffs.mean(dim=0)
            new_std_c = elite_coeffs.std(dim=0).clamp(min=CEM_SIGMA_MIN)

            alpha = 0.7  # momentum toward new estimate
            mean_coeffs[i] = alpha * new_mean_c + (1 - alpha) * mean_coeffs[i]
            std_coeffs[i] = alpha * new_std_c + (1 - alpha) * std_coeffs[i]

            # Track best actions (not coefficients — actual clamped actions)
            best_k = elite_idx[0]
            if route_costs[best_k] < best_costs[i]:
                best_costs[i] = route_costs[best_k]
                best_actions[i] = candidates[i, best_k]

        # Also update mean actions from best coefficients
        mean = (mean + torch.einsum("nb,bt->nt", mean_coeffs, basis)).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        mean_coeffs.zero_()  # reset coefficients (they're relative to updated mean)

        dt = time.time() - t0
        mean_cost = costs_2d[:, 0].mean()  # cost of current mean
        mean_best = np.mean(best_costs)
        mean_std = std.mean().item()
        print(
            f"  iter {it:3d}  mean={mean_cost:.1f}  best={mean_best:.1f}  "
            f"σ={mean_std:.4f}  ⏱{dt:.0f}s"
        )

        # Save checkpoint
        if (it + 1) % 10 == 0 or it == CEM_ITERS - 1:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "best_actions": best_actions.cpu(),
                    "best_costs": best_costs,
                    "iter": it,
                },
                SAVE_DIR / "latest.pt",
            )

    # Final evaluation of best actions
    print(f"\nFinal: verifying best actions...")
    final_costs = evaluate_actions(all_csv, best_actions, mdl_path, ort_sess, csv_cache)
    print(f"  mean={np.mean(final_costs):.1f}")
    for i in range(N):
        print(f"  [{i}] {all_csv[i].name}  cost={final_costs[i]:.1f}")

    print(f"\nDone. Best mean: {np.mean(final_costs):.1f}")


if __name__ == "__main__":
    main()
