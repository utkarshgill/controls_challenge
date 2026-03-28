#!/usr/bin/env python3
"""exp108 — Per-step CEM-MPC with RNG-aware deterministic lookahead

At each control step:
  1. Policy proposes a mean action for the next W steps
  2. CEM samples K smooth perturbations around the W-step mean
  3. Each candidate is evaluated with an H-step lookahead using the EXACT RNG
  4. Elites refine the mean. After CEM_ITERS, execute the first action.
  5. Advance one step, repeat.

The lookahead is fully deterministic: known context + known action + known RNG → exact token.
No noise in the cost evaluation. Pure search.
"""

import numpy as np, os, sys, time, torch, torch.nn.functional as F
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
MPC_K = int(os.getenv("MPC_K", "256"))  # candidates per step
MPC_H = int(os.getenv("MPC_H", "50"))  # lookahead horizon
MPC_W = int(os.getenv("MPC_W", "5"))  # CEM window: jointly optimize W actions
CEM_ITERS = int(os.getenv("CEM_ITERS", "3"))  # CEM refinement iterations per step
CEM_ELITE = float(os.getenv("CEM_ELITE", "0.1"))
CEM_SIGMA = float(os.getenv("CEM_SIGMA", "0.05"))


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Load policy for warm-start
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    print(f"exp108 — Per-step CEM-MPC")
    print(f"  K={MPC_K} H={MPC_H} W={MPC_W} CEM_iters={CEM_ITERS} σ={CEM_SIGMA}")
    print(f"  {N_ROUTES} routes")

    # Setup sim
    data, rng = csv_cache.slice(all_csv)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    future_windows = _precompute_future_windows(dg)
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)

    # Policy state for warm-start
    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    # Pre-generate policy actions for all steps (warm start for lookahead)
    policy_actions = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")

    n_elite = max(1, int(MPC_K * CEM_ELITE))
    stored_actions = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    _prev_time = [time.time()]

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

        # Get policy action as warm start
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
            future_windows,
            step_idx,
        )
        with torch.no_grad():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        policy_action = (h_act[:, hist_head] + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        # CEM over W actions starting at current step
        W = min(MPC_W, N_CTRL - ci)
        H = min(MPC_H, N_CTRL - ci)
        K = MPC_K

        # Mean action for W steps: policy action for step 0, stored/policy for steps 1+
        mean_w = torch.zeros((N, W), dtype=torch.float32, device="cuda")
        mean_w[:, 0] = policy_action.float()
        for w in range(1, W):
            if ci + w < N_CTRL and stored_actions[:, ci + w].abs().sum() > 0:
                mean_w[:, w] = stored_actions[:, ci + w]
            else:
                mean_w[:, w] = policy_action.float()  # fallback

        std_w = torch.full((N, W), CEM_SIGMA, dtype=torch.float32, device="cuda")

        # Build lookahead context (same as exp101 but with state fix)
        start = max(0, step_idx - CL + 1)
        act_h = sim_ref.action_history[:, start:step_idx].float()
        # State: read CL-1 valid entries + append current step's data
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
            act_h = F.pad(act_h, (pa, 0))
        if ps > 0:
            st_h = F.pad(st_h, (0, 0, ps, 0))
        if pp > 0:
            pr_h = F.pad(pr_h, (pp, 0))

        cur_la = sim_ref.current_lataccel.float()
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        # CEM iterations
        for cem_iter in range(CEM_ITERS):
            # Sample K candidate W-step action sequences per route
            noise = torch.randn(N, K, W, device="cuda") * std_w.unsqueeze(1)
            cand_w = (mean_w.unsqueeze(1) + noise).clamp(STEER_RANGE[0], STEER_RANGE[1])
            cand_w[:, 0] = mean_w  # candidate 0 = current mean

            # Expand context for K candidates
            NK = N * K
            act_hk = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
            st_hk = st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
            pr_hk = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
            cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)

            # H-step deterministic lookahead
            costs = torch.zeros(NK, device="cuda")
            prev_la = cur_la_k.clone()

            for h in range(H):
                s = step_idx + h
                if s >= dg["target_lataccel"].shape[1]:
                    break
                s_ci = s - CONTROL_START_IDX

                # Action: from CEM candidates for h < W, from stored/policy for h >= W
                if h < W:
                    cur_steer = cand_w[:, :, h].reshape(NK)
                else:
                    if 0 <= s_ci < N_CTRL and stored_actions[:, s_ci].abs().sum() > 0:
                        cur_steer = (
                            stored_actions[:, s_ci]
                            .unsqueeze(1)
                            .expand(-1, K)
                            .reshape(NK)
                        )
                    else:
                        cur_steer = (
                            policy_action.float().unsqueeze(1).expand(-1, K).reshape(NK)
                        )

                a_ctx = torch.cat([act_hk, cur_steer.unsqueeze(1)], dim=1)
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
                s_ctx = torch.cat([st_hk[:, 1:], new_st.unsqueeze(1)], dim=1)
                full_states = torch.cat([a_ctx.unsqueeze(-1), s_ctx], dim=-1)
                tokens = (
                    torch.bucketize(pr_hk.clamp(-5, 5), BINS, right=False)
                    .clamp(0, VOCAB_SIZE - 1)
                    .long()
                )

                # Deterministic: use the REAL RNG value
                if h < rng_base.shape[0]:
                    rng_h = rng_base[h].unsqueeze(1).expand(-1, K).reshape(NK)
                else:
                    rng_h = torch.rand(NK, device="cuda", dtype=torch.float64)

                sampled = mpc_phys._predict_gpu(
                    {"states": full_states, "tokens": tokens},
                    temperature=0.8,
                    rng_u=rng_h,
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
                    if h > 0:
                        costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100

                prev_la = pred_la
                act_hk = a_ctx[:, 1:]
                st_hk = s_ctx
                pr_hk = torch.cat([pr_hk[:, 1:], pred_la.unsqueeze(1)], dim=1)
                cur_la_k = pred_la

            # CEM update: select elites, refine mean and std
            cost_2d = costs.view(N, K)
            for i in range(N):
                elite_idx = torch.argsort(cost_2d[i])[:n_elite]
                elite_actions = cand_w[i, elite_idx]  # (n_elite, W)
                mean_w[i] = elite_actions.mean(dim=0)
                if cem_iter < CEM_ITERS - 1:
                    std_w[i] = elite_actions.std(dim=0).clamp(min=0.001)

        # Execute the best first action
        best_idx = cost_2d.argmin(dim=1)
        best_action = cand_w[torch.arange(N, device="cuda"), best_idx, 0]
        stored_actions[:, ci] = best_action

        # Store future actions from best candidate for warm start
        for w in range(1, W):
            if ci + w < N_CTRL:
                stored_actions[:, ci + w] = cand_w[
                    torch.arange(N, device="cuda"), best_idx, w
                ]

        # Update policy history
        h_act[:, next_head] = best_action.double()
        h_act32[:, next_head] = best_action
        h_lat[:, next_head] = cur32
        hist_head = next_head

        # Progress
        if (ci + 1) % 100 == 0:
            now = time.time()
            dt = now - _prev_time[0]
            _prev_time[0] = now
            print(f"      step {ci + 1}/{N_CTRL}  ⏱{dt:.1f}s", flush=True)

        return best_action.double()

    print("  Running...")
    t0 = time.time()
    costs = sim.rollout(ctrl)["total_cost"]
    dt = time.time() - t0
    print(f"\n  mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")
    for i, c in enumerate(costs):
        print(f"  [{i}] {all_csv[i].name}  cost={c:.1f}")


if __name__ == "__main__":
    main()
