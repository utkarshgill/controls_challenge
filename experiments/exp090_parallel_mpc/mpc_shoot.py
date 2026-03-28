# exp090 — Massively parallel shooting MPC (pure GPU)
#
# Uses BatchedPhysicsModel directly for the MPC lookahead.
# All GPU. No CPU transfers. IOBinding via TRT.
#
# At each control step, for all N routes:
#   1. Policy proposes mean action + samples K candidates
#   2. Simulate H steps forward using a SECOND BatchedPhysicsModel (expected-value)
#   3. Score by tracking error against the known future plan
#   4. Pick best candidate per route
#   5. Execute in the real stochastic sim

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
    FUTURE_K,
    S_LAT,
    S_STEER,
    S_VEGO,
    S_AEGO,
    S_ROLL,
    S_CURV,
    C,
    H1,
    H2,
    F_LAT,
    F_ROLL,
    F_V,
    F_A,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH

MPC_K = int(os.getenv("MPC_K", "16"))
MPC_H = int(os.getenv("MPC_H", "5"))
MPC_SIGMA = float(
    os.getenv("MPC_SIGMA", "0.0")
)  # 0 = sample from policy Beta, >0 = Gaussian around mean
N_ROUTES = int(os.getenv("N_ROUTES", "100"))
DELTA_SCALE = 0.25
EVAL_N = 100

BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)

BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)
EXP_DIR = Path(__file__).parent


class MpcPhysics:
    """Wrapper around BatchedPhysicsModel for MPC lookahead.
    Stochastic sampling with shared RNG per route (fair comparison across candidates).
    All GPU."""

    def __init__(self, mdl_path, ort_session):
        self.model = BatchedPhysicsModel(str(mdl_path), ort_session=ort_session)
        self.bins = BINS

    def stochastic_step(self, states_gpu, tokens_gpu, rng_u):
        """Run one physics step with stochastic sampling. All GPU.
        states_gpu: (B, 20, 4) float32
        tokens_gpu: (B, 20) int64
        rng_u: (B,) float64 — uniform random draws for sampling
        Returns: (B,) float64 sampled lataccel
        """
        samples = self.model._predict_gpu(
            {"states": states_gpu, "tokens": tokens_gpu}, temperature=0.8, rng_u=rng_u
        )
        # samples are token indices (B,) int64
        return self.bins[samples].double()

    def expected_step(self, states_gpu, tokens_gpu):
        """Run one physics step, return expected lataccel. All GPU."""
        self.model._predict_gpu(
            {"states": states_gpu, "tokens": tokens_gpu}, temperature=0.8, rng_u=None
        )
        probs = self.model._last_probs_gpu
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)


def mpc_rollout(
    csv_files,
    ac,
    mpc_phys,
    mdl_path,
    ort_session,
    csv_cache,
    ds=DELTA_SCALE,
    collect=False,
):
    """Run batched sim with per-step shooting MPC. All GPU."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    K, H = MPC_K, MPC_H
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    S = N_CTRL
    if collect:
        all_obs = torch.empty((S, N, OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw = torch.empty((S, N), dtype=torch.float32, device="cuda")
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum, si
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

        # Sample K candidates per route
        mean_raw = 2.0 * a_p / (a_p + b_p) - 1.0
        if MPC_SIGMA > 0:
            # Gaussian around policy mean
            noise = torch.randn(N, K, device="cuda") * MPC_SIGMA
            raw_cand = (mean_raw.unsqueeze(1) + noise).clamp(-1.0, 1.0)
        else:
            # Sample from policy Beta
            dist = torch.distributions.Beta(
                a_p.unsqueeze(1).expand(-1, K), b_p.unsqueeze(1).expand(-1, K)
            )
            raw_cand = 2.0 * dist.sample() - 1.0  # (N, K)
        raw_cand[:, 0] = mean_raw  # candidate 0 = always the mean

        prev_steer = h_act[:, hist_head].float()
        action_cand = (prev_steer.unsqueeze(1) + raw_cand * ds).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )  # (N, K)

        # Build context from real sim (all GPU tensors)
        start = max(0, step_idx - CL + 1)
        act_hist = sim.action_history[:, start:step_idx].float()
        state_hist = sim.state_history[:, start : step_idx + 1, :3].float()
        pred_hist = sim.current_lataccel_history[
            :, max(0, step_idx - CL) : step_idx
        ].float()

        pa = CL - 1 - act_hist.shape[1]
        ps = CL - state_hist.shape[1]
        pp = CL - pred_hist.shape[1]
        if pa > 0:
            act_hist = F.pad(act_hist, (pa, 0))
        if ps > 0:
            state_hist = F.pad(state_hist, (0, 0, ps, 0))
        if pp > 0:
            pred_hist = F.pad(pred_hist, (pp, 0))

        cur_la = sim.current_lataccel_history[:, max(0, step_idx - 1)].float()

        # Tile for K candidates: (N*K, ...)
        NK = N * K
        act_h = act_hist.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        st_h = state_hist.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
        pr_h = pred_hist.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
        cand_flat = action_cand.reshape(NK)

        # Get the RNG draws for the main sim at this step range
        # Use the SAME random draw for all K candidates of each route
        # so the stochastic comparison is fair
        rng_base = sim._rng_all_gpu[
            step_idx - CL : step_idx - CL + H, :
        ]  # (H, N) float64

        # Track history for policy rollout inside lookahead
        mpc_h_act = h_act32.unsqueeze(1).expand(-1, K, -1).reshape(NK, HIST_LEN).clone()
        mpc_h_lat = h_lat.unsqueeze(1).expand(-1, K, -1).reshape(NK, HIST_LEN).clone()
        mpc_h_error = (
            h_error.unsqueeze(1).expand(-1, K, -1).reshape(NK, HIST_LEN).clone()
        )
        mpc_err_sum = err_sum.unsqueeze(1).expand(-1, K).reshape(NK).clone()
        mpc_hist_head = hist_head
        mpc_obs = torch.empty((NK, OBS_DIM), dtype=torch.float32, device="cuda")

        # Precompute future windows for the tiled routes
        # dg has shape (N, T, ...). We need (NK, T, ...)
        # But fill_obs indexes by [:, step_idx] which returns (NK,) if we tile dg
        # Instead, we'll index dg with route indices and tile manually

        # Shoot H steps forward (stochastic, policy-rolled)
        costs = torch.zeros(NK, device="cuda")
        prev_la = cur_la_k.clone()

        for h in range(H):
            s = step_idx + h
            if s >= dg["target_lataccel"].shape[1]:
                break

            # Get action for this step
            if h == 0:
                # First step: use candidate actions
                cur_steer = cand_flat
            else:
                # Steps 1+: policy rollout — query policy with predicted state
                s_idx = min(s, dg["target_lataccel"].shape[1] - 1)
                tgt_h = (
                    dg["target_lataccel"][:, s_idx]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK)
                )
                roll_h = (
                    dg["roll_lataccel"][:, s_idx]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK)
                )
                v_h = (
                    dg["v_ego"][:, s_idx].float().unsqueeze(1).expand(-1, K).reshape(NK)
                )
                a_h = (
                    dg["a_ego"][:, s_idx].float().unsqueeze(1).expand(-1, K).reshape(NK)
                )
                cur_la_f = cur_la_k

                # Update error tracking
                error_h = (tgt_h - cur_la_f).float()
                mpc_next = (mpc_hist_head + 1) % HIST_LEN
                old_e = mpc_h_error[:, mpc_next]
                mpc_h_error[:, mpc_next] = error_h
                mpc_err_sum = mpc_err_sum + error_h - old_e
                ei_h = mpc_err_sum * (DEL_T / HIST_LEN)

                # Build obs for policy (inline, handles NK batch)
                v2_h = torch.clamp(v_h * v_h, min=1.0)
                k_tgt_h = (tgt_h - roll_h) / v2_h
                k_cur_h = (cur_la_f - roll_h) / v2_h
                fp0_h = dg["target_lataccel"][
                    :, min(s + 1, dg["target_lataccel"].shape[1] - 1)
                ].float()
                fp0_h = fp0_h.unsqueeze(1).expand(-1, K).reshape(NK)
                fric_h = torch.sqrt(cur_la_f**2 + a_h**2) / 7.0
                pa_h = mpc_h_act[:, mpc_hist_head]
                pa2_h = mpc_h_act[:, (mpc_hist_head - 1) % HIST_LEN]
                pl_h = mpc_h_lat[:, mpc_hist_head]
                mpc_obs[:, 0] = tgt_h / S_LAT
                mpc_obs[:, 1] = cur_la_f / S_LAT
                mpc_obs[:, 2] = (tgt_h - cur_la_f) / S_LAT
                mpc_obs[:, 3] = k_tgt_h / S_CURV
                mpc_obs[:, 4] = k_cur_h / S_CURV
                mpc_obs[:, 5] = (k_tgt_h - k_cur_h) / S_CURV
                mpc_obs[:, 6] = v_h / S_VEGO
                mpc_obs[:, 7] = a_h / S_AEGO
                mpc_obs[:, 8] = roll_h / S_ROLL
                mpc_obs[:, 9] = pa_h / S_STEER
                mpc_obs[:, 10] = ei_h / S_LAT
                mpc_obs[:, 11] = (fp0_h - tgt_h) / DEL_T / S_LAT
                mpc_obs[:, 12] = (cur_la_f - pl_h) / DEL_T / S_LAT
                mpc_obs[:, 13] = (pa_h - pa2_h) / DEL_T / S_STEER
                mpc_obs[:, 14] = fric_h
                mpc_obs[:, 15] = torch.clamp(1.0 - fric_h, min=0.0)
                # History rings
                for j in range(HIST_LEN):
                    idx_j = (mpc_hist_head + 1 + j) % HIST_LEN
                    mpc_obs[:, C + j] = mpc_h_act[:, idx_j] / S_STEER
                    mpc_obs[:, H1 + j] = mpc_h_lat[:, idx_j] / S_LAT
                # Future plan (tile from N to NK)
                for key, start, scale in [
                    ("target_lataccel", F_LAT, S_LAT),
                    ("roll_lataccel", F_ROLL, S_ROLL),
                    ("v_ego", F_V, S_VEGO),
                    ("a_ego", F_A, S_AEGO),
                ]:
                    fp = future[key][:, min(s, future[key].shape[1] - 1)]
                    mpc_obs[:, start : start + FUTURE_K] = (
                        fp.unsqueeze(1).expand(-1, K, -1).reshape(NK, FUTURE_K) / scale
                    )
                mpc_obs.clamp_(-5.0, 5.0)

                with torch.inference_mode():
                    logits_h = ac.actor(mpc_obs)
                a_p_h = F.softplus(logits_h[..., 0]) + 1.0
                b_p_h = F.softplus(logits_h[..., 1]) + 1.0
                raw_h = 2.0 * a_p_h / (a_p_h + b_p_h) - 1.0  # deterministic mean
                delta_h = raw_h * ds
                cur_steer = (mpc_h_act[:, mpc_hist_head] + delta_h).clamp(
                    STEER_RANGE[0], STEER_RANGE[1]
                )

            # Action context: append current steer
            a_ctx = torch.cat([act_h, cur_steer.unsqueeze(1)], dim=1)

            # State context
            s_idx = min(s, dg["roll_lataccel"].shape[1] - 1)
            new_st = torch.stack(
                [
                    dg["roll_lataccel"][:, s_idx]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK),
                    dg["v_ego"][:, s_idx]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK),
                    dg["a_ego"][:, s_idx]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK),
                ],
                dim=-1,
            )
            s_ctx = torch.cat([st_h[:, 1:], new_st.unsqueeze(1)], dim=1)

            # Full states: (NK, CL, 4)
            full_states = torch.cat([a_ctx.unsqueeze(-1), s_ctx], dim=-1)

            # Tokens
            tokens = (
                torch.bucketize(pr_h.clamp(-5, 5), BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .long()
            )

            # Stochastic physics step — SAME RNG for all K copies of each route
            if h < rng_base.shape[0]:
                rng_h = rng_base[h]
                rng_tiled = rng_h.unsqueeze(1).expand(-1, K).reshape(NK)
            else:
                rng_tiled = torch.rand(NK, device="cuda", dtype=torch.float64)

            pred_la = mpc_phys.stochastic_step(full_states, tokens, rng_tiled)
            pred_la = pred_la.clamp(
                cur_la_k.double() - MAX_ACC_DELTA, cur_la_k.double() + MAX_ACC_DELTA
            )

            # Cost against future plan
            if CONTROL_START_IDX <= s < COST_END_IDX:
                tgt = (
                    dg["target_lataccel"][:, s]
                    .double()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK)
                )
                costs += (tgt - pred_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
                if h > 0:
                    costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100

            # Advance context
            prev_la = pred_la
            act_h = a_ctx[:, 1:]
            st_h = s_ctx
            pr_h = torch.cat([pr_h[:, 1:], pred_la.float().unsqueeze(1)], dim=1)
            cur_la_k = pred_la.float()

            # Update MPC internal histories for policy rollout
            mpc_next = (mpc_hist_head + 1) % HIST_LEN
            mpc_h_act[:, mpc_next] = cur_steer.float()
            mpc_h_lat[:, mpc_next] = cur_la_k
            mpc_hist_head = mpc_next

        # Pick best per route
        best_idx = costs.view(N, K).argmin(dim=1)
        best_action = action_cand[torch.arange(N, device="cuda"), best_idx]
        best_raw = raw_cand[torch.arange(N, device="cuda"), best_idx]

        # Collect obs and raw for BC distillation
        if collect and step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_raw[si] = best_raw
            si += 1

        h_act[:, next_head] = best_action.double()
        h_act32[:, next_head] = best_action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return best_action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    if collect:
        return (
            costs,
            all_obs[:si].permute(1, 0, 2).reshape(-1, OBS_DIM).cpu(),
            all_raw[:si].T.reshape(-1).cpu(),
        )
    return costs


def main():
    print(f"exp090 — Massively parallel shooting MPC (pure GPU)")
    print(f"  K={MPC_K}  H={MPC_H}  routes={N_ROUTES}")

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy (Δs={ds})")

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    # MPC physics: separate BatchedPhysicsModel, same TRT session
    mpc_phys = MpcPhysics(mdl_path, ort_sess)

    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    routes = all_csv[: max(N_ROUTES, EVAL_N)]
    csv_cache = CSVCache([str(f) for f in routes])  # only cache what we need

    # Baseline
    print("\nBaseline (no MPC)...")
    from experiments.exp055_batch_of_batch.train import rollout as exp055_rollout

    base_res = exp055_rollout(
        va_f, ac, mdl_path, ort_sess, csv_cache, deterministic=True, ds=ds
    )
    print(f"  Baseline: {np.mean(base_res):.1f}")

    # MPC on all routes in batches of 100, with collection
    BATCH = 100
    all_routes = all_csv[:N_ROUTES]
    all_mpc_obs = []
    all_mpc_raw = []
    all_mpc_costs = []

    print(f"\nShooting MPC on {N_ROUTES} routes (batch={BATCH})...")
    t0_total = time.time()
    for i in range(0, len(all_routes), BATCH):
        batch = all_routes[i : i + BATCH]
        t0 = time.time()
        costs, obs, raw = mpc_rollout(
            batch, ac, mpc_phys, mdl_path, ort_sess, csv_cache, ds=ds, collect=True
        )
        dt = time.time() - t0
        all_mpc_costs.extend(
            costs.tolist() if hasattr(costs, "tolist") else list(costs)
        )
        all_mpc_obs.append(obs)
        all_mpc_raw.append(raw)
        print(f"  [{i}..{i + len(batch) - 1}]  cost={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    dt_total = time.time() - t0_total
    mc = np.mean(all_mpc_costs)
    print(
        f"\nTotal: {N_ROUTES} routes  MPC={mc:.1f}  baseline={np.mean(base_res):.1f}"
        f"  Δ={np.mean(base_res) - mc:.1f}  ⏱{dt_total:.0f}s"
    )

    out_path = EXP_DIR / "mpc_shoot_data.pt"
    torch.save(
        {
            "mpc_obs": torch.cat(all_mpc_obs),
            "mpc_raw": torch.cat(all_mpc_raw),
            "mpc_costs": np.array(all_mpc_costs),
            "baseline": np.mean(base_res),
        },
        out_path,
    )
    print(f"Saved {torch.cat(all_mpc_obs).shape[0]} samples to {out_path}")


if __name__ == "__main__":
    main()
