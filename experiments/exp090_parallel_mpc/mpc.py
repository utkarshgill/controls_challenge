# exp090 — Massively parallel MPC using tinyphysics_batched
#
# The simplest possible MPC:
#   1. Tile each route K times
#   2. Each copy runs the policy with a per-copy perturbation
#   3. All K*R rollouts run in ONE batched sim call (TRT GPU)
#   4. Pick the lowest-cost copy per route
#   5. Output the winning actions + obs for BC distillation
#
# Perturbation options:
#   - Per-copy constant offset (shifts the policy mean)
#   - Per-copy temperature (samples from wider distribution)
#   - Per-copy smooth perturbation (basis functions)
#   - Combination
#
# This is MPC. N routes × K candidates. Evaluated by the real cost function.
# Using the exact same TRT batched sim that PPO uses. Nothing special.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F
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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

# ── Config ────────────────────────────────────────────────────
MPC_K = int(os.getenv("MPC_K", "128"))  # candidates per route
N_ROUTES = int(os.getenv("N_ROUTES", "100"))  # routes per batch
TOTAL_BS = int(os.getenv("TOTAL_BS", "5000"))  # max total rollouts per sim call
DELTA_SCALE = 0.25
SMOOTH = int(os.getenv("SMOOTH", "0"))  # 0=stochastic, 1=smooth perturbations
BIAS = int(os.getenv("BIAS", "0"))  # 0=off, 1=per-copy constant bias
BIAS_RANGE = float(os.getenv("BIAS_RANGE", "0.02"))  # max bias in raw delta space
N_BASIS = int(os.getenv("N_BASIS", "40"))
BASIS_WIDTH = int(os.getenv("BASIS_WIDTH", "20"))
PERTURB_STD = float(os.getenv("PERTURB_STD", "0.02"))

BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)
EXP_DIR = Path(__file__).parent


def mpc_search(route_files, ac, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE):
    """Run MPC: R routes × K candidates in one batched sim call.

    Copy 0 per route = deterministic policy mean (baseline).
    Copies 1..K-1 = policy samples (stochastic, from the actual Beta distribution).

    The only difference from normal PPO rollout: K copies instead of SAMPLES_PER_ROUTE,
    and we PICK THE BEST instead of computing advantages.

    Returns dict with costs, winning obs, winning raw actions.
    """
    R = len(route_files)
    K = MPC_K

    # Tile routes K times
    csv_tiled = [f for f in route_files for _ in range(K)]

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

    S = N_CTRL
    all_obs = torch.empty((S, N, OBS_DIM), dtype=torch.float32, device="cuda")
    all_raw = torch.empty((S, N), dtype=torch.float32, device="cuda")

    si = 0
    hist_head = HIST_LEN - 1

    # Per-copy index: which copy (0..K-1) is each rollout?
    copy_idx = torch.arange(N, device="cuda") % K

    # Precompute per-copy constant biases if enabled
    if BIAS:
        biases = torch.linspace(-BIAS_RANGE, BIAS_RANGE, K, device=DEV)  # (K,)
        biases[0] = 0.0  # copy 0 = no bias
        per_copy_bias = biases[copy_idx]  # (N*K,)

    # Precompute smooth perturbations if enabled
    if SMOOTH:
        centers = torch.linspace(0, N_CTRL - 1, N_BASIS, device=DEV)
        t_axis = torch.arange(N_CTRL, dtype=torch.float32, device=DEV)
        basis = torch.exp(
            -0.5 * ((t_axis.unsqueeze(0) - centers.unsqueeze(1)) / BASIS_WIDTH) ** 2
        )
        basis = basis / basis.max(dim=1, keepdim=True).values  # (N_BASIS, N_CTRL)
        # Sample coefficients per rollout: (N, N_BASIS)
        coeffs = torch.randn(N, N_BASIS, device=DEV) * PERTURB_STD
        coeffs[copy_idx == 0] = 0.0  # copy 0 = no perturbation
        smooth_perturb = coeffs @ basis  # (N, N_CTRL)

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
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

        # Copy 0: deterministic mean. Copies 1..K-1: perturbed.
        mean_raw = 2.0 * a_p / (a_p + b_p) - 1.0
        if BIAS:
            raw = (mean_raw + per_copy_bias).clamp(-1.0, 1.0)
        elif SMOOTH:
            ci = step_idx - CONTROL_START_IDX
            if 0 <= ci < N_CTRL:
                raw = (mean_raw + smooth_perturb[:, ci]).clamp(-1.0, 1.0)
            else:
                raw = mean_raw
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            sampled_raw = 2.0 * dist.sample() - 1.0
            raw = torch.where(copy_idx == 0, mean_raw, sampled_raw)

        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_raw[si] = raw.float()
            si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]  # (R*K,)
    S_actual = si

    # Reshape and find winners
    costs_2d = torch.from_numpy(costs).to("cuda").float().view(R, K)
    winners = costs_2d.argmin(dim=1)  # (R,)
    winner_flat = torch.arange(R, device="cuda") * K + winners
    baseline_flat = torch.arange(R, device="cuda") * K  # copy 0

    baseline_costs = costs_2d[:, 0]
    winner_costs = costs_2d.gather(1, winners.unsqueeze(1)).squeeze(1)

    # Extract winning trajectories
    win_obs = all_obs[:S_actual, winner_flat, :]  # (S, R, OBS_DIM)
    win_raw = all_raw[:S_actual, winner_flat]  # (S, R)
    base_obs = all_obs[:S_actual, baseline_flat, :]
    base_raw = all_raw[:S_actual, baseline_flat]

    improved = winner_costs < baseline_costs
    improve = (baseline_costs - winner_costs).mean().item()
    frac = (winners != 0).float().mean().item()

    return dict(
        costs_2d=costs_2d,
        baseline_costs=baseline_costs,
        winner_costs=winner_costs,
        improve=improve,
        frac=frac,
        # Winner data for BC
        win_obs=win_obs.permute(1, 0, 2).reshape(-1, OBS_DIM),
        win_raw=win_raw.T.reshape(-1),
        # Baseline data for anchor
        base_obs=base_obs.permute(1, 0, 2).reshape(-1, OBS_DIM),
        base_raw=base_raw.T.reshape(-1),
    )


def main():
    print(f"exp090 — Massively parallel MPC")
    print(f"  K={MPC_K}  routes={N_ROUTES}  total_bs={TOTAL_BS}")

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy (Δs={ds})")

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Process in batches that fit in TRT
    routes_per_batch = min(N_ROUTES, TOTAL_BS // MPC_K)
    routes = all_csv[:N_ROUTES]

    all_baseline = []
    all_winner = []
    all_win_obs = []
    all_win_raw = []
    all_base_obs = []
    all_base_raw = []

    for i in range(0, len(routes), routes_per_batch):
        batch = routes[i : i + routes_per_batch]
        t0 = time.time()
        res = mpc_search(batch, ac, mdl_path, ort_sess, csv_cache, ds=ds)
        dt = time.time() - t0

        all_baseline.extend(res["baseline_costs"].cpu().tolist())
        all_winner.extend(res["winner_costs"].cpu().tolist())
        all_win_obs.append(res["win_obs"].cpu())
        all_win_raw.append(res["win_raw"].cpu())
        all_base_obs.append(res["base_obs"].cpu())
        all_base_raw.append(res["base_raw"].cpu())

        mb = res["baseline_costs"].mean().item()
        mw = res["winner_costs"].mean().item()
        print(
            f"  [{i}..{i + len(batch) - 1}]  base={mb:.1f}  win={mw:.1f}"
            f"  Δ={res['improve']:.1f}  f={res['frac']:.2f}  ⏱{dt:.1f}s"
        )

    mb = np.mean(all_baseline)
    mw = np.mean(all_winner)
    print(
        f"\nTotal: {len(routes)} routes  base={mb:.1f}  win={mw:.1f}  Δ={mb - mw:.1f}"
    )

    out_path = EXP_DIR / "mpc_data.pt"
    torch.save(
        {
            "win_obs": torch.cat(all_win_obs),
            "win_raw": torch.cat(all_win_raw),
            "base_obs": torch.cat(all_base_obs),
            "base_raw": torch.cat(all_base_raw),
            "baseline_costs": np.array(all_baseline),
            "winner_costs": np.array(all_winner),
        },
        out_path,
    )
    print(f"Saved to {out_path}  ({torch.cat(all_win_obs).shape[0]} samples)")


if __name__ == "__main__":
    main()
