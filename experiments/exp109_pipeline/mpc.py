#!/usr/bin/env python3
"""exp109 — Combined MPC + CEM pipeline

Pipeline:
  1. Per-step policy MPC (exp101 style): K candidates, H-step lookahead, pick best → ~34
  2. Full-trajectory smooth CEM (exp105 style): cosine basis, refine stored actions → ~27
  3. Repeat CEM passes with shrinking sigma → push lower

All deterministic RNG, all GPU batched.
"""

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

# ── Config ──
N_ROUTES = int(os.getenv("N_ROUTES", "10"))
# Pass 1: per-step MPC
MPC_K = int(os.getenv("MPC_K", "128"))
MPC_H = int(os.getenv("MPC_H", "50"))
MPC_SIGMA = float(os.getenv("MPC_SIGMA", "0.01"))
# Pass 2+: full-trajectory CEM
CEM_K = int(os.getenv("CEM_K", "128"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "100"))
CEM_PASSES = int(os.getenv("CEM_PASSES", "5"))
CEM_SIGMA_START = float(os.getenv("CEM_SIGMA_START", "0.1"))
CEM_SIGMA_END = float(os.getenv("CEM_SIGMA_END", "0.01"))
CEM_ELITE = float(os.getenv("CEM_ELITE", "0.1"))
N_BASIS = int(os.getenv("N_BASIS", "40"))
# Pass 3+: per-step coord descent refinement
CD_PASSES = int(os.getenv("CD_PASSES", "3"))
CD_K = int(os.getenv("CD_K", "128"))
CD_SIGMA_START = float(os.getenv("CD_SIGMA_START", "0.005"))
CD_SIGMA_END = float(os.getenv("CD_SIGMA_END", "0.001"))
SAVE_DIR = Path(
    os.getenv("SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints"))
)


# ═══════════════════════════════════════════════════════════
# Pass 1: Per-step policy MPC (from exp101, with state context fix)
# ═══════════════════════════════════════════════════════════


def mpc_pass(csv_files, mdl_path, ort_sess, csv_cache):
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
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1
    K = MPC_K
    H = MPC_H

    # Pre-fill stored actions by running policy deterministically
    stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    _h_act_pre = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    _h_act32_pre = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _h_lat_pre = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _h_error_pre = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _err_sum_pre = torch.zeros(N, dtype=torch.float32, device="cuda")
    _hh_pre = HIST_LEN - 1

    data_pre, rng_pre = csv_cache.slice(csv_files)
    sim_pre = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data_pre, cached_rng=rng_pre
    )
    dg_pre = sim_pre.data_gpu
    future_pre = _precompute_future_windows(dg_pre)

    def ctrl_pre(step_idx, sim_ref):
        nonlocal _hh_pre, _err_sum_pre
        target = dg_pre["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        error = (target - current).float()
        nh = (_hh_pre + 1) % HIST_LEN
        old = _h_error_pre[:, nh]
        _h_error_pre[:, nh] = error
        _err_sum_pre = _err_sum_pre + error - old
        ei = _err_sum_pre * (DEL_T / HIST_LEN)
        if step_idx < CONTROL_START_IDX:
            _h_act_pre[:, nh] = 0.0
            _h_act32_pre[:, nh] = 0.0
            _h_lat_pre[:, nh] = cur32
            _hh_pre = nh
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        fill_obs(
            obs_buf,
            target.float(),
            cur32,
            dg_pre["roll_lataccel"][:, step_idx].float(),
            dg_pre["v_ego"][:, step_idx].float(),
            dg_pre["a_ego"][:, step_idx].float(),
            _h_act32_pre,
            _h_lat_pre,
            _hh_pre,
            ei,
            future_pre,
            step_idx,
        )
        with torch.no_grad():
            logits = ac.actor(obs_buf)
        a_p = tF.softplus(logits[..., 0]) + 1.0
        b_p = tF.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(_h_act_pre.dtype) * ds
        action = (_h_act_pre[:, _hh_pre] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        if ci < N_CTRL:
            stored[:, ci] = action.float()
        _h_act_pre[:, nh] = action
        _h_act32_pre[:, nh] = action.float()
        _h_lat_pre[:, nh] = cur32
        _hh_pre = nh
        return action

    sim_pre.rollout(ctrl_pre)
    print(f"    (policy pre-fill done)")

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
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        # Policy mean as center
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
        a_p = tF.softplus(logits[..., 0]) + 1.0
        b_p = tF.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        policy_act = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # K candidates around policy mean
        noise = torch.randn(N, K, device="cuda") * MPC_SIGMA
        cand = (policy_act.float().unsqueeze(1) + noise).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        cand[:, 0] = policy_act.float()

        # H-step lookahead with deterministic RNG
        NK = N * K
        start = max(0, step_idx - CL + 1)
        act_h = sim_ref.action_history[:, start:step_idx].float()
        st_h_prev = sim_ref.state_history[:, start:step_idx, :3].float()
        cur_state = torch.stack(
            [
                dg["roll_lataccel"][:, step_idx].float(),
                dg["v_ego"][:, step_idx].float(),
                dg["a_ego"][:, step_idx].float(),
            ],
            dim=-1,
        )
        st_h = torch.cat([st_h_prev, cur_state.unsqueeze(1)], dim=1)
        pr_h = sim_ref.current_lataccel_history[
            :, max(0, step_idx - CL) : step_idx
        ].float()

        pa = CL - 1 - act_h.shape[1]
        ps = CL - st_h.shape[1]
        pp = CL - pr_h.shape[1]
        if pa > 0:
            act_h = tF.pad(act_h, (pa, 0))
        if ps > 0:
            st_h = tF.pad(st_h, (0, 0, ps, 0))
        if pp > 0:
            pr_h = tF.pad(pr_h, (pp, 0))

        cur_la = sim_ref.current_lataccel.float()
        act_h = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        st_h = st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
        pr_h = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        costs = torch.zeros(NK, device="cuda")
        prev_la = cur_la_k.clone()
        for h_i in range(H):
            s = step_idx + h_i
            if s >= dg["target_lataccel"].shape[1]:
                break
            s_ci = s - CONTROL_START_IDX
            if h_i == 0:
                cur_steer = cand.reshape(NK)
            else:
                if 0 <= s_ci < N_CTRL:
                    cur_steer = stored[:, s_ci].unsqueeze(1).expand(-1, K).reshape(NK)
                else:
                    cur_steer = torch.zeros(NK, device="cuda")

            a_ctx = torch.cat([act_h, cur_steer.unsqueeze(1)], dim=1)
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
            full_states = torch.cat([a_ctx.unsqueeze(-1), s_ctx], dim=-1)
            tokens = (
                torch.bucketize(pr_h.clamp(-5, 5), BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .long()
            )
            if h_i < rng_base.shape[0]:
                rng_h = rng_base[h_i].unsqueeze(1).expand(-1, K).reshape(NK)
            else:
                rng_h = torch.rand(NK, device="cuda", dtype=torch.float64)
            sampled = mpc_phys._predict_gpu(
                {"states": full_states, "tokens": tokens}, temperature=0.8, rng_u=rng_h
            )
            pred_la = (
                BINS[sampled]
                .float()
                .clamp(cur_la_k - MAX_ACC_DELTA, cur_la_k + MAX_ACC_DELTA)
            )
            if CONTROL_START_IDX <= s < COST_END_IDX:
                tgt = (
                    dg["target_lataccel"][:, s]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK)
                )
                costs += (tgt - pred_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
                if h_i > 0:
                    costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100
            prev_la = pred_la
            act_h = a_ctx[:, 1:]
            st_h = s_ctx
            pr_h = torch.cat([pr_h[:, 1:], pred_la.unsqueeze(1)], dim=1)
            cur_la_k = pred_la

        cost_2d = costs.view(N, K)
        best_idx = cost_2d.argmin(dim=1)
        best_action = cand[torch.arange(N, device="cuda"), best_idx]
        stored[:, ci] = best_action

        h_act[:, next_head] = best_action.double()
        h_act32[:, next_head] = best_action
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return best_action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, stored


# ═══════════════════════════════════════════════════════════
# Pass 2+: Full-trajectory CEM with smooth basis
# ═══════════════════════════════════════════════════════════


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


def cem_pass(csv_files, stored_actions, mdl_path, ort_sess, csv_cache, sigma, n_iters):
    N = len(csv_files)
    K = CEM_K
    n_elite = max(1, int(K * CEM_ELITE))

    t_grid = torch.linspace(0, 1, N_CTRL, device="cuda")
    basis = torch.stack([torch.cos(np.pi * k * t_grid) for k in range(N_BASIS)])
    basis = basis / basis.norm(dim=1, keepdim=True)

    mean = stored_actions.clone()
    mean_coeffs = torch.zeros((N, N_BASIS), dtype=torch.float32, device="cuda")
    std_coeffs = torch.full((N, N_BASIS), sigma, dtype=torch.float32, device="cuda")
    best_actions = stored_actions.clone()
    best_costs = np.full(N, float("inf"))

    for it in range(n_iters):
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
                alpha * elite_coeffs.std(dim=0).clamp(min=0.005)
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

    return best_actions, best_costs


# ═══════════════════════════════════════════════════════════
# Pass 3+: Per-step coord descent (from exp101, true sequential)
# ═══════════════════════════════════════════════════════════


def coord_descent_pass(csv_files, stored_actions, mdl_path, ort_sess, csv_cache, sigma):
    """True sequential coord descent: perturb one step at a time, execute best."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    K = CD_K
    H = MPC_H
    dg = sim.data_gpu
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)
    orig_actions = stored_actions
    live_actions = stored_actions.clone()
    improved_count = 0

    def ctrl(step_idx, sim_ref):
        nonlocal improved_count
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        base_action = orig_actions[:, ci]
        noise = torch.randn(N, K, device="cuda") * sigma
        action_cand = (base_action.unsqueeze(1) + noise).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        action_cand[:, 0] = base_action

        NK = N * K
        start = max(0, step_idx - CL + 1)
        act_h = sim_ref.action_history[:, start:step_idx].float()
        st_h_prev = sim_ref.state_history[:, start:step_idx, :3].float()
        cur_state = torch.stack(
            [
                dg["roll_lataccel"][:, step_idx].float(),
                dg["v_ego"][:, step_idx].float(),
                dg["a_ego"][:, step_idx].float(),
            ],
            dim=-1,
        )
        st_h = torch.cat([st_h_prev, cur_state.unsqueeze(1)], dim=1)
        pr_h = sim_ref.current_lataccel_history[
            :, max(0, step_idx - CL) : step_idx
        ].float()

        pa = CL - 1 - act_h.shape[1]
        ps = CL - st_h.shape[1]
        pp = CL - pr_h.shape[1]
        if pa > 0:
            act_h = tF.pad(act_h, (pa, 0))
        if ps > 0:
            st_h = tF.pad(st_h, (0, 0, ps, 0))
        if pp > 0:
            pr_h = tF.pad(pr_h, (pp, 0))

        cur_la = sim_ref.current_lataccel.float()
        act_h = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        st_h = st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
        pr_h = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        costs = torch.zeros(NK, device="cuda")
        prev_la = cur_la_k.clone()
        for h_i in range(H):
            s = step_idx + h_i
            if s >= dg["target_lataccel"].shape[1]:
                break
            s_ci = s - CONTROL_START_IDX
            if h_i == 0:
                cur_steer = action_cand.reshape(NK)
            else:
                if 0 <= s_ci < N_CTRL:
                    cur_steer = (
                        live_actions[:, s_ci].unsqueeze(1).expand(-1, K).reshape(NK)
                    )
                else:
                    cur_steer = torch.zeros(NK, device="cuda")

            a_ctx = torch.cat([act_h, cur_steer.unsqueeze(1)], dim=1)
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
            full_states = torch.cat([a_ctx.unsqueeze(-1), s_ctx], dim=-1)
            tokens = (
                torch.bucketize(pr_h.clamp(-5, 5), BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .long()
            )
            if h_i < rng_base.shape[0]:
                rng_h = rng_base[h_i].unsqueeze(1).expand(-1, K).reshape(NK)
            else:
                rng_h = torch.rand(NK, device="cuda", dtype=torch.float64)
            sampled = mpc_phys._predict_gpu(
                {"states": full_states, "tokens": tokens}, temperature=0.8, rng_u=rng_h
            )
            pred_la = (
                BINS[sampled]
                .float()
                .clamp(cur_la_k - MAX_ACC_DELTA, cur_la_k + MAX_ACC_DELTA)
            )
            if CONTROL_START_IDX <= s < COST_END_IDX:
                tgt = (
                    dg["target_lataccel"][:, s]
                    .float()
                    .unsqueeze(1)
                    .expand(-1, K)
                    .reshape(NK)
                )
                costs += (tgt - pred_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
                if h_i > 0:
                    costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100
            prev_la = pred_la
            act_h = a_ctx[:, 1:]
            st_h = s_ctx
            pr_h = torch.cat([pr_h[:, 1:], pred_la.unsqueeze(1)], dim=1)
            cur_la_k = pred_la

        cost_2d = costs.view(N, K)
        best_idx = cost_2d.argmin(dim=1)
        best_action = action_cand[torch.arange(N, device="cuda"), best_idx]
        changed = best_idx != 0
        if changed.any():
            live_actions[changed, ci] = best_action[changed]
            improved_count += changed.sum().item()

        return live_actions[:, ci].double()

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, live_actions


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    print(f"exp109 — MPC + CEM pipeline: {N_ROUTES} routes")

    # Pass 1: per-step MPC
    print(f"\n  Pass 1: per-step MPC (K={MPC_K}, H={MPC_H})")
    t0 = time.time()
    costs, stored = mpc_pass(all_csv, mdl_path, ort_sess, csv_cache)
    dt = time.time() - t0
    print(f"    mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    # CEM passes
    sigmas = np.geomspace(CEM_SIGMA_START, CEM_SIGMA_END, CEM_PASSES)
    for p, sig in enumerate(sigmas):
        print(
            f"\n  CEM pass {p + 2}: σ={sig:.4f}, {CEM_ITERS} iters (K={CEM_K}, basis={N_BASIS})"
        )
        t0 = time.time()
        stored, costs = cem_pass(
            all_csv, stored, mdl_path, ort_sess, csv_cache, sigma=sig, n_iters=CEM_ITERS
        )
        dt = time.time() - t0
        print(f"    mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    # Coord descent passes: fine per-step refinement
    if CD_PASSES > 0:
        cd_sigmas = np.geomspace(CD_SIGMA_START, CD_SIGMA_END, CD_PASSES)
        for p, sig in enumerate(cd_sigmas):
            print(f"\n  CD pass {p + 1}: σ={sig:.4f} (K={CD_K}, H={MPC_H})")
            t0 = time.time()
            costs, stored = coord_descent_pass(
                all_csv, stored, mdl_path, ort_sess, csv_cache, sigma=sig
            )
            dt = time.time() - t0
            print(f"    mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    # Final verification
    print(f"\n  Final verification:")
    final_costs = evaluate_actions(all_csv, stored, mdl_path, ort_sess, csv_cache)
    print(f"    mean={np.mean(final_costs):.1f}")
    for i, c in enumerate(final_costs):
        print(f"    [{i}] {all_csv[i].name}  cost={c:.1f}")

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"actions": stored.cpu(), "costs": np.array(final_costs)}, SAVE_DIR / "best.pt"
    )
    print(f"\n  Saved to {SAVE_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
