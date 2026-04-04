#!/usr/bin/env python3
"""exp110 — True receding-horizon MPC with per-step CEM

At each step t, optimize a W-step action window [t:t+W] using smooth-basis CEM.
Evaluate each candidate over H-step horizon with exact RNG.
Execute action[t], warm-start actions[t+1:t+W] for the next step.

This is true MPC: joint multi-step optimization, receding horizon, re-plan every step.
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

N_ROUTES = int(os.getenv("N_ROUTES", "10"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "500"))  # routes per GPU batch
USE_CUDA_GRAPH = int(os.getenv("USE_CUDA_GRAPH", "0"))
MPC_K = int(os.getenv("MPC_K", "64"))
MPC_H = int(os.getenv("MPC_H", "50"))  # evaluation horizon
MPC_W = int(os.getenv("MPC_W", "20"))  # optimization window
CEM_ITERS = int(os.getenv("CEM_ITERS", "10"))
CEM_SIGMA = float(
    os.getenv("CEM_SIGMA", "0.1")
)  # default lower for warm-start refinement
CEM_ELITE_FRAC = float(os.getenv("CEM_ELITE", "0.15"))
N_BASIS_W = int(os.getenv("N_BASIS_W", "8"))  # basis functions over the W-step window


class FastMPCPredictor:
    """Optimized physics model for MPC lookahead.
    Pre-binds IOBinding once for a fixed (NK, CL) shape.
    Reuses bindings across all calls — zero per-call overhead.
    """

    def __init__(self, mdl_path, ort_session, NK, CL=CONTEXT_LENGTH):
        self.NK = NK
        self.CL = CL
        # Pre-allocate all GPU buffers
        self.states_buf = torch.empty((NK, CL, 4), dtype=torch.float32, device="cuda")
        self.tokens_buf = torch.empty((NK, CL), dtype=torch.int64, device="cuda")
        self.out_buf = torch.empty(
            (NK, CL, VOCAB_SIZE), dtype=torch.float32, device="cuda"
        )
        self.probs_buf = torch.empty(
            (NK, VOCAB_SIZE), dtype=torch.float32, device="cuda"
        )

        # Pre-bind IOBinding ONCE
        self._io = ort_session.io_binding()
        self._io.bind_input(
            "states", "cuda", 0, np.float32, [NK, CL, 4], self.states_buf.data_ptr()
        )
        self._io.bind_input(
            "tokens", "cuda", 0, np.int64, [NK, CL], self.tokens_buf.data_ptr()
        )
        out_name = ort_session.get_outputs()[0].name
        self._io.bind_output(
            out_name,
            "cuda",
            0,
            np.float32,
            [NK, CL, VOCAB_SIZE],
            self.out_buf.data_ptr(),
        )
        self._sess = ort_session
        self._bins = BINS

    def predict(self, states, tokens, temperature, rng_u):
        """states: (NK, CL, 4), tokens: (NK, CL) → sampled token indices (NK,)
        Writes directly into pre-allocated buffers. No rebinding."""
        # Copy data into pre-bound buffers
        self.states_buf.copy_(states)
        self.tokens_buf.copy_(tokens)
        # Run model (no rebinding needed!)
        self._sess.run_with_iobinding(self._io)
        # Softmax + sample
        torch.div(self.out_buf[:, -1, :], temperature, out=self.probs_buf)
        torch.softmax(self.probs_buf, dim=-1, out=self.probs_buf)
        # CDF + searchsorted
        cdf = torch.cumsum(self.probs_buf, dim=1)
        cdf.div_(cdf[:, -1:])
        u = rng_u.unsqueeze(1) if rng_u.dim() == 1 else rng_u
        samples = (
            torch.searchsorted(cdf.double(), u.double())
            .squeeze(1)
            .clamp(0, VOCAB_SIZE - 1)
        )
        return samples


WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")  # path to actions.npz for warm start


def run_mpc_batch(
    batch_csv,
    mdl_path,
    ort_sess,
    csv_cache,
    ac,
    ds,
    mpc_phys=None,
    mpc_ort=None,
    warm_actions_dict=None,
):
    """Run receding-horizon MPC on a batch of routes. Returns (actions, costs, mpc_phys)."""
    all_csv = batch_csv
    N = len(all_csv)

    # Warm start: from npz if available, else from policy rollout
    if warm_actions_dict is not None:
        policy_stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
        for i, f in enumerate(all_csv):
            if f.name in warm_actions_dict:
                policy_stored[i] = (
                    torch.from_numpy(warm_actions_dict[f.name]).float().to("cuda")
                )
        print(
            f"  Warm-start from npz ({sum(1 for f in all_csv if f.name in warm_actions_dict)}/{N} found)"
        )
    else:
        # Pre-compute policy actions for warm start
        data_pre, rng_pre = csv_cache.slice(all_csv)
        sim_pre = BatchedSimulator(
            str(mdl_path),
            ort_session=ort_sess,
            cached_data=data_pre,
            cached_rng=rng_pre,
        )
        N = sim_pre.N
        dg_pre = sim_pre.data_gpu
        future_pre = _precompute_future_windows(dg_pre)
        policy_stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")

    _h_a = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    _h_a32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _h_l = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _h_e = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    _es = torch.zeros(N, dtype=torch.float32, device="cuda")
    _ob = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    _hh = HIST_LEN - 1

    def ctrl_pre(step_idx, sim_ref):
        nonlocal _hh, _es
        tgt = dg_pre["target_lataccel"][:, step_idx]
        cur = sim_ref.current_lataccel
        c32 = cur.float()
        err = (tgt - cur).float()
        nh = (_hh + 1) % HIST_LEN
        oe = _h_e[:, nh]
        _h_e[:, nh] = err
        _es = _es + err - oe
        ei = _es * (DEL_T / HIST_LEN)
        if step_idx < CONTROL_START_IDX:
            _h_a[:, nh] = 0.0
            _h_a32[:, nh] = 0.0
            _h_l[:, nh] = c32
            _hh = nh
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        fill_obs(
            _ob,
            tgt.float(),
            c32,
            dg_pre["roll_lataccel"][:, step_idx].float(),
            dg_pre["v_ego"][:, step_idx].float(),
            dg_pre["a_ego"][:, step_idx].float(),
            _h_a32,
            _h_l,
            _hh,
            ei,
            future_pre,
            step_idx,
        )
        with torch.no_grad():
            logits = ac.actor(_ob)
        ap = tF.softplus(logits[..., 0]) + 1.0
        bp = tF.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * ap / (ap + bp) - 1.0
        action = (_h_a[:, _hh] + raw.to(_h_a.dtype) * ds).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        ci = step_idx - CONTROL_START_IDX
        if ci < N_CTRL:
            policy_stored[:, ci] = action.float()
        _h_a[:, nh] = action
        _h_a32[:, nh] = action.float()
        _h_l[:, nh] = c32
        _hh = nh
        return action

    if warm_actions_dict is None:
        sim_pre.rollout(ctrl_pre)
        print(f"  Policy warm-start done")

    # Build per-window cosine basis (W_max steps)
    W_max = MPC_W
    t_grid_w = torch.linspace(0, 1, W_max, device="cuda")
    basis_w = torch.stack([torch.cos(np.pi * k * t_grid_w) for k in range(N_BASIS_W)])
    basis_w = basis_w / basis_w.norm(dim=1, keepdim=True)  # (N_BASIS_W, W_max)

    K = MPC_K
    n_elite = max(1, int(K * CEM_ELITE_FRAC))

    print(f"exp110 — Receding-horizon MPC")
    print(f"  K={K} H={MPC_H} W={MPC_W} CEM_iters={CEM_ITERS} basis={N_BASIS_W}")
    print(f"  σ={CEM_SIGMA} elite={n_elite}")

    # Main sim
    data, rng = csv_cache.slice(all_csv)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    dg = sim.data_gpu
    NK = N * MPC_K
    # Reuse MPC predictor if provided, else create one
    if mpc_phys is None or mpc_phys.NK != NK:
        if mpc_ort is None:
            mpc_ort = make_ort_session(mdl_path)
        mpc_phys = FastMPCPredictor(str(mdl_path), mpc_ort, NK)

    # Warm-start window: actions[t:t+W] initialized from policy
    warm_w = policy_stored.clone()  # (N, N_CTRL) — full stored actions
    stored_final = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    _prev_time = [time.time()]

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        W = min(MPC_W, N_CTRL - ci)
        H = min(MPC_H, N_CTRL - ci)
        basis_w_cur = basis_w[:, :W]  # (N_BASIS_W, W)

        # Mean for CEM: warm_w actions for the window
        mean_w = warm_w[:, ci : ci + W].clone()  # (N, W)

        # Build lookahead context
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
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        # ── Precompute constants for the H-step lookahead (done ONCE per step) ──
        NK = N * K
        # Expand base context to NK — done once, reused across CEM iters
        act_h_base = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1).contiguous()
        st_h_base = (
            st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3).contiguous()
        )
        pr_h_base = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1).contiguous()
        cur_la_base = cur_la.unsqueeze(1).expand(-1, K).reshape(NK).contiguous()

        # Precompute state data for all H lookahead steps (constant, doesn't change)
        H_actual = min(H, dg["target_lataccel"].shape[1] - step_idx)
        state_data_h = torch.empty(
            (H_actual, NK, 3), dtype=torch.float32, device="cuda"
        )
        tgt_data_h = torch.empty((H_actual, NK), dtype=torch.float32, device="cuda")
        rng_data_h = torch.empty((H_actual, NK), dtype=torch.float64, device="cuda")
        for h_i in range(H_actual):
            s_idx = min(step_idx + h_i, dg["roll_lataccel"].shape[1] - 1)
            state_data_h[h_i, :, 0] = (
                dg["roll_lataccel"][:, s_idx]
                .float()
                .unsqueeze(1)
                .expand(-1, K)
                .reshape(NK)
            )
            state_data_h[h_i, :, 1] = (
                dg["v_ego"][:, s_idx].float().unsqueeze(1).expand(-1, K).reshape(NK)
            )
            state_data_h[h_i, :, 2] = (
                dg["a_ego"][:, s_idx].float().unsqueeze(1).expand(-1, K).reshape(NK)
            )
            tgt_data_h[h_i] = (
                dg["target_lataccel"][:, step_idx + h_i]
                .float()
                .unsqueeze(1)
                .expand(-1, K)
                .reshape(NK)
            )
            if h_i < rng_base.shape[0]:
                rng_data_h[h_i] = rng_base[h_i].unsqueeze(1).expand(-1, K).reshape(NK)
            else:
                rng_data_h[h_i] = torch.rand(NK, device="cuda", dtype=torch.float64)

        # Precompute warm actions for h >= W (constant across CEM iters)
        warm_steer_h = torch.zeros((H_actual, NK), dtype=torch.float32, device="cuda")
        for h_i in range(W, H_actual):
            s_ci = step_idx + h_i - CONTROL_START_IDX
            if 0 <= s_ci < N_CTRL:
                warm_steer_h[h_i] = (
                    warm_w[:, s_ci].unsqueeze(1).expand(-1, K).reshape(NK)
                )

        # Pre-allocate lookahead buffers (reused across CEM iters)
        act_hk = torch.empty_like(act_h_base)
        st_hk = torch.empty_like(st_h_base)
        pr_hk = torch.empty_like(pr_h_base)
        full_states = torch.empty((NK, CL, 4), dtype=torch.float32, device="cuda")

        # CEM loop
        mean_coeffs = torch.zeros((N, N_BASIS_W), dtype=torch.float32, device="cuda")
        std_coeffs = torch.full(
            (N, N_BASIS_W), CEM_SIGMA, dtype=torch.float32, device="cuda"
        )

        best_w = mean_w.clone()
        best_cost = torch.full((N,), float("inf"), device="cuda")
        arange_N = torch.arange(N, device="cuda")

        for cem_it in range(CEM_ITERS):
            # Sample candidates in basis space
            noise = torch.randn(N, K, N_BASIS_W, device="cuda") * std_coeffs.unsqueeze(
                1
            )
            cand_coeffs = mean_coeffs.unsqueeze(1) + noise
            cand_coeffs[:, 0] = mean_coeffs

            # Project to action space
            perturbation = torch.einsum("nkb,bw->nkw", cand_coeffs, basis_w_cur)
            cand_w = (mean_w.unsqueeze(1) + perturbation).clamp(
                STEER_RANGE[0], STEER_RANGE[1]
            )

            # Reset lookahead state from precomputed base
            act_hk.copy_(act_h_base)
            st_hk.copy_(st_h_base)
            pr_hk.copy_(pr_h_base)
            cur_la_k = cur_la_base.clone()

            costs = torch.zeros(NK, device="cuda")
            prev_la = cur_la_k.clone()

            for h_i in range(H_actual):
                s = step_idx + h_i

                # Action source
                if h_i < W:
                    cur_steer = cand_w[:, :, h_i].reshape(NK)
                else:
                    cur_steer = warm_steer_h[h_i]

                # Build full_states: shift context left, append new
                # actions context (CL): act_hk is (NK, CL-1), append cur_steer
                a_ctx = torch.cat([act_hk, cur_steer.unsqueeze(1)], dim=1)  # (NK, CL)
                # states context: shift st_hk left, append new state
                s_ctx = torch.cat(
                    [st_hk[:, 1:], state_data_h[h_i].unsqueeze(1)], dim=1
                )  # (NK, CL, 3)
                # combine
                full_states[:, :, 0] = a_ctx
                full_states[:, :, 1:] = s_ctx

                tokens = (
                    torch.bucketize(pr_hk.clamp(-5, 5), BINS, right=False)
                    .clamp(0, VOCAB_SIZE - 1)
                    .long()
                )

                sampled = mpc_phys.predict(
                    full_states,
                    tokens,
                    temperature=0.8,
                    rng_u=rng_data_h[h_i],
                )
                pred_la = (
                    BINS[sampled]
                    .float()
                    .clamp(cur_la_k - MAX_ACC_DELTA, cur_la_k + MAX_ACC_DELTA)
                )

                if CONTROL_START_IDX <= s < COST_END_IDX:
                    costs += (
                        (tgt_data_h[h_i] - pred_la) ** 2
                        * 100
                        * LAT_ACCEL_COST_MULTIPLIER
                    )
                    if h_i > 0:
                        costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100

                prev_la = pred_la
                # Slide context for next step
                act_hk = a_ctx[:, 1:]  # (NK, CL-1)
                st_hk = s_ctx  # (NK, CL, 3)
                pr_hk = torch.cat([pr_hk[:, 1:], pred_la.unsqueeze(1)], dim=1)
                cur_la_k = pred_la

            # CEM elite selection — fully vectorized on GPU
            cost_2d = costs.view(N, K)
            _, sorted_idx = cost_2d.sort(dim=1)
            elite_idx = sorted_idx[:, :n_elite]

            elite_coeffs = cand_coeffs.gather(
                1, elite_idx.unsqueeze(-1).expand(-1, -1, N_BASIS_W)
            )
            new_mean = elite_coeffs.mean(dim=1)
            mean_coeffs = 0.7 * new_mean + 0.3 * mean_coeffs
            if cem_it < CEM_ITERS - 1:
                new_std = elite_coeffs.std(dim=1).clamp(min=0.01)
                std_coeffs = 0.7 * new_std + 0.3 * std_coeffs

            # Track best per route
            best_k = sorted_idx[:, 0]
            best_k_cost = cost_2d[arange_N, best_k]
            improved = best_k_cost < best_cost
            if improved.any():
                best_cost[improved] = best_k_cost[improved]
                best_w[improved] = cand_w[improved, best_k[improved]]

        # Execute best action at step t
        action = best_w[:, 0]
        stored_final[:, ci] = action

        # Update warm-start: use optimized future actions
        w_end = min(W, N_CTRL - ci)
        if w_end > 1:
            warm_w[:, ci + 1 : ci + w_end] = best_w[:, 1:w_end]

        if (ci + 1) % 100 == 0:
            now = time.time()
            dt = now - _prev_time[0]
            _prev_time[0] = now
            print(f"      step {ci + 1}/{N_CTRL}  ⏱{dt:.1f}s", flush=True)

        return action.double()

    print("  Running MPC...")
    t0 = time.time()
    costs = sim.rollout(ctrl)["total_cost"]
    dt = time.time() - t0
    print(f"  batch mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")
    return stored_final, costs, mpc_phys


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Load policy once
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    print(
        f"exp110 — Receding-horizon MPC: {N_ROUTES} routes in batches of {BATCH_ROUTES}"
    )
    print(f"  K={MPC_K} H={MPC_H} W={MPC_W} CEM_iters={CEM_ITERS} basis={N_BASIS_W}")

    # Load warm-start from previous run if available
    warm_actions_dict = None
    if WARM_START_NPZ and Path(WARM_START_NPZ).exists():
        warm_data = np.load(WARM_START_NPZ)
        warm_actions_dict = {k: warm_data[k] for k in warm_data.files}
        print(
            f"  Loaded warm-start from {WARM_START_NPZ} ({len(warm_actions_dict)} routes)"
        )

    save_dir = Path(__file__).resolve().parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    actions_dict = {}
    all_costs = []
    mpc_phys_cached = None
    mpc_ort_cached = make_ort_session(mdl_path)  # create ONCE, reuse across batches

    for batch_start in range(0, N_ROUTES, BATCH_ROUTES):
        batch_end = min(batch_start + BATCH_ROUTES, N_ROUTES)
        batch_csv = all_csv[batch_start:batch_end]
        print(f"\n  Batch {batch_start}-{batch_end} ({len(batch_csv)} routes):")
        t0 = time.time()
        actions, costs, mpc_phys_cached = run_mpc_batch(
            batch_csv,
            mdl_path,
            ort_sess,
            csv_cache,
            ac,
            ds,
            mpc_phys=mpc_phys_cached,
            mpc_ort=mpc_ort_cached,
            warm_actions_dict=warm_actions_dict,
        )
        dt = time.time() - t0

        # Accumulate
        actions_np = actions.cpu().numpy()
        for i, f in enumerate(batch_csv):
            actions_dict[f.name] = actions_np[i]
            all_costs.append(costs[i])

        # Save incrementally
        np.savez(save_dir / "actions.npz", **actions_dict)
        running_mean = np.mean(all_costs)
        print(
            f"  running mean={running_mean:.1f}  ({len(all_costs)}/{N_ROUTES} routes)  ⏱{dt:.0f}s"
        )

    print(f"\nFinal: {len(all_costs)} routes, mean={np.mean(all_costs):.1f}")
    print(f"Saved to {save_dir / 'actions.npz'}")


if __name__ == "__main__":
    main()
