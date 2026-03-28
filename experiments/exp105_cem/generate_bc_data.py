#!/usr/bin/env python3
"""Generate CEM-optimized actions for BC training.

Runs CEM in batches of BATCH_ROUTES routes, accumulates results.
Saves (obs, action) pairs for behavioral cloning.
"""

import numpy as np, os, sys, time, torch
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

TOTAL_ROUTES = int(os.getenv("TOTAL_ROUTES", "1000"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "100"))
CEM_K = int(os.getenv("CEM_K", "128"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "100"))
CEM_ELITE = float(os.getenv("CEM_ELITE", "0.1"))
CEM_SIGMA = float(os.getenv("CEM_SIGMA", "0.1"))
CEM_SIGMA_MIN = float(os.getenv("CEM_SIGMA_MIN", "0.005"))
N_BASIS = int(os.getenv("N_BASIS", "40"))
SAVE_PATH = Path(
    os.getenv("SAVE_PATH", str(Path(__file__).resolve().parent / "cem_actions.pt"))
)


def evaluate_actions(csv_files, actions, mdl_path, ort_sess, csv_cache):
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
    from experiments.exp055_batch_of_batch.train import (
        ActorCritic,
        _precompute_future_windows,
        fill_obs,
        HIST_LEN,
        OBS_DIM,
        DEL_T as DT,
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
        ei = err_sum * (DT / HIST_LEN)
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

    sim.rollout(ctrl)
    return stored


def cem_optimize(csv_files, mdl_path, ort_sess, csv_cache):
    """Run CEM on a batch of routes. Returns (best_actions, best_costs)."""
    N = len(csv_files)
    K = CEM_K
    n_elite = max(1, int(K * CEM_ELITE))

    # Warm start from policy
    mean = get_policy_actions(csv_files, mdl_path, ort_sess, csv_cache)

    # Cosine basis
    t_grid = torch.linspace(0, 1, N_CTRL, device="cuda")
    basis = torch.stack([torch.cos(np.pi * k * t_grid) for k in range(N_BASIS)])
    basis = basis / basis.norm(dim=1, keepdim=True)

    mean_coeffs = torch.zeros((N, N_BASIS), dtype=torch.float32, device="cuda")
    std_coeffs = torch.full((N, N_BASIS), CEM_SIGMA, dtype=torch.float32, device="cuda")
    best_actions = mean.clone()
    best_costs = np.full(N, float("inf"))

    for it in range(CEM_ITERS):
        noise = torch.randn(N, K, N_BASIS, device="cuda") * std_coeffs.unsqueeze(1)
        cand_coeffs = mean_coeffs.unsqueeze(1) + noise
        cand_coeffs[:, 0] = mean_coeffs
        perturbation = torch.einsum("nkb,bt->nkt", cand_coeffs, basis)
        candidates = (mean.unsqueeze(1) + perturbation).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        flat_actions = candidates.reshape(N * K, N_CTRL)
        flat_csvs = [f for f in csv_files for _ in range(K)]
        flat_costs = evaluate_actions(
            flat_csvs, flat_actions, mdl_path, ort_sess, csv_cache
        )
        costs_2d = flat_costs.reshape(N, K)

        for i in range(N):
            route_costs = costs_2d[i]
            elite_idx = np.argsort(route_costs)[:n_elite]
            elite_coeffs = cand_coeffs[i, elite_idx]
            alpha = 0.7
            mean_coeffs[i] = (
                alpha * elite_coeffs.mean(dim=0) + (1 - alpha) * mean_coeffs[i]
            )
            std_coeffs[i] = (
                alpha * elite_coeffs.std(dim=0).clamp(min=CEM_SIGMA_MIN)
                + (1 - alpha) * std_coeffs[i]
            )
            best_k = elite_idx[0]
            if route_costs[best_k] < best_costs[i]:
                best_costs[i] = route_costs[best_k]
                best_actions[i] = candidates[i, best_k]

        mean = (mean + torch.einsum("nb,bt->nt", mean_coeffs, basis)).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        mean_coeffs.zero_()

        if (it + 1) % 25 == 0:
            print(
                f"      iter {it + 1}/{CEM_ITERS}  mean={costs_2d[:, 0].mean():.1f}  best={np.mean(best_costs):.1f}"
            )

    return best_actions, best_costs


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:TOTAL_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    print(
        f"Generating CEM actions for {TOTAL_ROUTES} routes in batches of {BATCH_ROUTES}"
    )
    print(f"  K={CEM_K} iters={CEM_ITERS} basis={N_BASIS}")

    all_actions = []
    all_costs = []

    for batch_start in range(0, TOTAL_ROUTES, BATCH_ROUTES):
        batch_end = min(batch_start + BATCH_ROUTES, TOTAL_ROUTES)
        batch_csv = all_csv[batch_start:batch_end]
        print(f"\n  Batch {batch_start}-{batch_end} ({len(batch_csv)} routes):")
        t0 = time.time()
        actions, costs = cem_optimize(batch_csv, mdl_path, ort_sess, csv_cache)
        dt = time.time() - t0
        print(f"    mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")
        all_actions.append(actions.cpu())
        all_costs.append(costs)

    # Concatenate and save
    all_actions = torch.cat(all_actions, dim=0)
    all_costs = np.concatenate(all_costs)
    print(f"\nTotal: {len(all_actions)} routes, mean cost={np.mean(all_costs):.1f}")

    torch.save(
        {
            "actions": all_actions,
            "costs": all_costs,
            "csv_files": [str(f) for f in all_csv],
        },
        SAVE_PATH,
    )
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
