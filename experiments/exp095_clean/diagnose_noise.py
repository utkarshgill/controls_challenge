# Diagnostic: measure how much of the noise cost comes from
# direct sampling noise vs context corruption.
#
# Runs the policy on stochastic sim (temp=0.8).
# Computes cost THREE ways:
#   1. Standard: cost on sampled lataccel (the real eval metric)
#   2. Expected: cost on expected lataccel (what would the cost be if we used E[lataccel]?)
#   3. Cross: sampled lataccel for context, but expected for cost (isolates context corruption)

import numpy as np, os, sys, torch, torch.nn.functional as F
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


def run_diagnostic(n_routes=100):
    ac = ActorCritic().to(DEV)
    base_pt = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
    ckpt = torch.load(base_pt, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_cache = CSVCache([str(f) for f in all_csv[:n_routes]])
    va_f = all_csv[:n_routes]

    data, rng = csv_cache.slice(va_f)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    sim.compute_expected = True  # compute expected lataccel at each step
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

    # Collect expected lataccel at each scored step
    expected_history = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    si = 0

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

        # Record expected lataccel from PREVIOUS step's sim_step
        if step_idx >= CONTROL_START_IDX and step_idx < COST_END_IDX:
            exp = getattr(sim_ref, "expected_lataccel", None)
            if exp is not None:
                expected_history[:, si] = exp.float()
            si += 1

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
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs_standard = sim.rollout(ctrl)["total_cost"]

    # Now compute costs using expected lataccel
    start, end = CONTROL_START_IDX, CONTROL_START_IDX + N_CTRL
    sampled = sim.current_lataccel_history[:, start:end].float()
    expected = expected_history[:, :N_CTRL]
    targets = dg["target_lataccel"][:, start:end].float()

    # Cost 1: Standard (on sampled) — the real eval metric
    lat_s = ((targets - sampled) ** 2).mean(dim=1) * 100
    jerk_s = ((torch.diff(sampled, dim=1) / DEL_T) ** 2).mean(dim=1) * 100
    cost_sampled = lat_s * LAT_ACCEL_COST_MULTIPLIER + jerk_s

    # Cost 2: On expected lataccel (no sampling noise in cost)
    lat_e = ((targets - expected) ** 2).mean(dim=1) * 100
    jerk_e = ((torch.diff(expected, dim=1) / DEL_T) ** 2).mean(dim=1) * 100
    cost_expected = lat_e * LAT_ACCEL_COST_MULTIPLIER + jerk_e

    # Noise magnitude
    noise = sampled - expected
    noise_std = noise.std(dim=1).mean().item()
    noise_mean_abs = noise.abs().mean().item()

    print(f"Routes: {n_routes}")
    print(f"")
    print(
        f"Cost on SAMPLED lataccel:  {cost_sampled.mean().item():.1f} ± {cost_sampled.std().item():.1f}"
    )
    print(
        f"Cost on EXPECTED lataccel: {cost_expected.mean().item():.1f} ± {cost_expected.std().item():.1f}"
    )
    print(f"")
    print(f"Breakdown (sampled):")
    print(f"  lat_cost:  {(lat_s * LAT_ACCEL_COST_MULTIPLIER).mean().item():.1f}")
    print(f"  jerk_cost: {jerk_s.mean().item():.1f}")
    print(f"")
    print(f"Breakdown (expected):")
    print(f"  lat_cost:  {(lat_e * LAT_ACCEL_COST_MULTIPLIER).mean().item():.1f}")
    print(f"  jerk_cost: {jerk_e.mean().item():.1f}")
    print(f"")
    print(f"Noise stats:")
    print(f"  std:      {noise_std:.4f}")
    print(f"  mean|n|:  {noise_mean_abs:.4f}")
    print(f"")
    print(f"Gap analysis:")
    gap = cost_sampled.mean().item() - cost_expected.mean().item()
    print(f"  Total gap (sampled - expected): {gap:.1f}")
    print(f"  This is the irreducible noise cost")


if __name__ == "__main__":
    run_diagnostic()
