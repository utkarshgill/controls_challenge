# exp099 — Full-trajectory CEM through real TRT sim with PPO warm start
#
# For each segment: CEM in smooth perturbation space (40 basis functions).
# K candidates per CEM iteration, R iterations. All segments batched in parallel.
# Uses the REAL TRT sim with exact RNG replay. No surrogate mismatch.
# PPO policy provides the base trajectory. CEM finds smooth corrections.

import numpy as np, os, sys, time, random, torch, torch.nn.functional as F
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
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
    FUTURE_K,
)

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

# ── Config ────────────────────────────────────────────────────
MPC_K = int(os.getenv("MPC_K", "128"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "15"))
N_BASIS = int(os.getenv("N_BASIS", "40"))
BASIS_WIDTH = int(os.getenv("BASIS_WIDTH", "20"))
PERTURB_STD = float(os.getenv("PERTURB_STD", "0.02"))
ELITE_FRAC = float(os.getenv("ELITE_FRAC", "0.125"))
N_SEGS = int(os.getenv("N_SEGS", "100"))
SEG_BATCH = int(os.getenv("SEG_BATCH", "10"))  # segments per CEM batch
DELTA_SCALE = 0.25

EXP_DIR = Path(__file__).parent
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


def make_basis():
    centers = torch.linspace(0, N_CTRL - 1, N_BASIS, device=DEV)
    t = torch.arange(N_CTRL, dtype=torch.float32, device=DEV)
    basis = torch.exp(
        -0.5 * ((t.unsqueeze(0) - centers.unsqueeze(1)) / BASIS_WIDTH) ** 2
    )
    basis = basis / basis.max(dim=1, keepdim=True).values
    return basis  # (N_BASIS, N_CTRL)


def cem_optimize(
    seg_files, ac, basis, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """CEM over smooth perturbations for R segments × K candidates.

    All R×K rollouts run in one batched sim call per CEM iteration.
    """
    R = len(seg_files)
    K = MPC_K
    n_elite = max(1, int(K * ELITE_FRAC))

    # CEM state per segment
    cem_mean = torch.zeros(R, N_BASIS, device=DEV)
    cem_std = torch.full((R, N_BASIS), PERTURB_STD, device=DEV)

    best_cost = torch.full((R,), float("inf"), device=DEV)
    best_actions = torch.zeros((R, N_CTRL), dtype=torch.float64, device=DEV)

    for cem_iter in range(CEM_ITERS):
        # Sample perturbation coefficients: (R, K, N_BASIS)
        coeffs = torch.randn(R, K, N_BASIS, device=DEV) * cem_std.unsqueeze(
            1
        ) + cem_mean.unsqueeze(1)
        coeffs[:, 0, :] = cem_mean  # candidate 0 = current CEM mean

        # Convert to smooth perturbations: (R*K, N_CTRL)
        flat_coeffs = coeffs.view(R * K, N_BASIS)
        perturbations = flat_coeffs @ basis  # (R*K, N_CTRL)

        # Tile segment files: [s0c0, s0c1, ..., s0cK, s1c0, ...]
        csv_tiled = [f for f in seg_files for _ in range(K)]
        data, rng = csv_cache.slice(csv_tiled)
        sim = BatchedSimulator(
            str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
        )
        N = sim.N  # R * K
        dg = sim.data_gpu
        future = _precompute_future_windows(dg)

        h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
        h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
        obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
        hist_head = HIST_LEN - 1

        all_actions = torch.zeros((N, N_CTRL), dtype=torch.float64, device="cuda")

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

            with torch.inference_mode():
                logits = ac.actor(obs_buf)
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            mean_raw = 2.0 * a_p / (a_p + b_p) - 1.0

            ci = step_idx - CONTROL_START_IDX
            if ci < N_CTRL:
                raw = (mean_raw + perturbations[:, ci]).clamp(-1.0, 1.0)
            else:
                raw = mean_raw

            delta = raw.to(h_act.dtype) * ds
            action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
            h_act[:, next_head] = action
            h_act32[:, next_head] = action.float()
            h_lat[:, next_head] = cur32
            hist_head = next_head

            if 0 <= ci < N_CTRL:
                all_actions[:, ci] = action
            return action

        costs_np = sim.rollout(ctrl)["total_cost"]
        costs_t = torch.from_numpy(costs_np).to(DEV).float().view(R, K)

        # CEM update: keep elites, refit distribution
        _, elite_idx = costs_t.topk(n_elite, dim=1, largest=False)
        elite_coeffs = torch.gather(
            coeffs, 1, elite_idx.unsqueeze(-1).expand(-1, -1, N_BASIS)
        )
        cem_mean = elite_coeffs.mean(dim=1)
        cem_std = elite_coeffs.std(dim=1).clamp(min=0.001)

        # Track best
        iter_best_cost, iter_best_idx = costs_t.min(dim=1)
        improved = iter_best_cost < best_cost
        if improved.any():
            best_cost[improved] = iter_best_cost[improved]
            flat_best = torch.arange(R, device=DEV) * K + iter_best_idx
            best_actions[improved] = all_actions[flat_best[improved]]

        if cem_iter == 0 or cem_iter == CEM_ITERS - 1:
            print(
                f"      iter {cem_iter}: best={best_cost.mean().item():.1f}"
                f"  elite={costs_t[torch.arange(R).unsqueeze(1), elite_idx].mean().item():.1f}"
                f"  std={cem_std.mean().item():.4f}"
            )

    return best_cost.cpu().numpy(), best_actions.cpu().numpy()


def main():
    print(f"exp099 — Full-trajectory CEM on TRT sim")
    print(f"  K={MPC_K}  iters={CEM_ITERS}  basis={N_BASIS}  width={BASIS_WIDTH}")
    print(f"  σ={PERTURB_STD}  elite={ELITE_FRAC}  segs={N_SEGS}  batch={SEG_BATCH}")

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy (Δs={ds})")

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_SEGS]
    csv_cache = CSVCache([str(f) for f in all_csv])
    basis = make_basis()

    EXP_DIR.mkdir(exist_ok=True)
    all_costs = []
    all_actions = []
    all_paths = []

    for i in range(0, len(all_csv), SEG_BATCH):
        batch = all_csv[i : i + SEG_BATCH]
        print(f"\n  Segments {i}..{i + len(batch) - 1}")
        t0 = time.time()
        costs, actions = cem_optimize(
            batch, ac, basis, mdl_path, ort_sess, csv_cache, ds=ds
        )
        dt = time.time() - t0
        all_costs.extend(costs.tolist())
        all_actions.append(actions)
        all_paths.extend([str(p) for p in batch])
        print(f"    cost={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    mc = np.mean(all_costs)
    print(f"\nDone. {len(all_csv)} segments. Mean cost: {mc:.1f}")

    np.savez(
        EXP_DIR / "optimized_actions.npz",
        actions=np.concatenate(all_actions),
        costs=np.array(all_costs),
        paths=all_paths,
    )
    print(f"Saved to {EXP_DIR / 'optimized_actions.npz'}")


if __name__ == "__main__":
    main()
