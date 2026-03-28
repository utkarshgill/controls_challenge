# exp101 — Per-step MPC with multi-pass coordinate descent
#
# Pass 1: Per-step MPC (policy samples K candidates, H-step lookahead, pick best)
# Pass 2+: Coordinate descent on stored actions:
#   - Replay stored actions exactly (reproduces pass N-1 trajectory)
#   - At each step: try K perturbations, simulate H steps using stored actions for 1+
#   - If better, update stored action at that step
#   - Continue left to right
#
# The sim replays stored actions. The lookahead uses stored actions.
# Everything is consistent. The RNG is deterministic per segment.

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

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH
BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)
BINS_F32 = BINS.float()  # for expected value computation

MPC_K = int(os.getenv("MPC_K", "64"))
MPC_H = int(os.getenv("MPC_H", "50"))
MPC_SIGMA = float(os.getenv("MPC_SIGMA", "0.01"))
NUM_CD = int(os.getenv("NUM_CD", "20"))  # number of coord descent passes
SIGMA_START = float(os.getenv("SIGMA_START", "0.01"))
SIGMA_END = float(os.getenv("SIGMA_END", "0.0002"))
PASS_SIGMAS = os.getenv("PASS_SIGMAS", "")  # override: explicit comma-separated
N_ROUTES = int(os.getenv("N_ROUTES", "10"))
DEBUG = int(os.getenv("DEBUG", "0"))
VERIFY = int(os.getenv("VERIFY", "0"))
RESUME = int(os.getenv("RESUME", "0"))
SAVE_DIR = Path(
    os.getenv("SAVE_DIR", str(Path(__file__).resolve().parent / "checkpoints"))
)
DELTA_SCALE = 0.25

BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ═══════════════════════════════════════════════════════════
# Pass 1: Per-step MPC with policy
# ═══════════════════════════════════════════════════════════


def mpc_pass1(csv_files, ac, mdl_path, ort_session, csv_cache, ds, sigma):
    """Policy-based MPC. Returns costs and stored absolute steer actions."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N = sim.N
    K = MPC_K
    H = MPC_H
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_session)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1
    stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    _t = {"costs": torch.zeros(N, device="cuda"), "prev_la": None, "t": time.time()}

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
        prev_steer = h_act[:, hist_head].float()

        # Sample K candidates in raw space around policy mean
        noise = torch.randn(N, K, device="cuda") * sigma
        raw_cand = (mean_raw.unsqueeze(1) + noise).clamp(-1.0, 1.0)
        raw_cand[:, 0] = mean_raw
        action_cand = (prev_steer.unsqueeze(1) + raw_cand * ds).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        # H-step lookahead (policy rollout for steps 1+)
        NK = N * K
        start = max(0, step_idx - CL + 1)
        # NOTE: state_history[:, step_idx] not yet written at controller call time.
        # Read CL-1 valid entries, append data[step_idx] manually.
        act_h = sim_ref.action_history[:, start:step_idx].float()  # CL-1
        st_h_prev = sim_ref.state_history[:, start:step_idx, :3].float()  # CL-1
        cur_state = torch.stack(
            [
                dg["roll_lataccel"][:, step_idx].float(),
                dg["v_ego"][:, step_idx].float(),
                dg["a_ego"][:, step_idx].float(),
            ],
            dim=-1,
        )  # (N, 3)
        st_h = torch.cat([st_h_prev, cur_state.unsqueeze(1)], dim=1)  # CL entries
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
        act_h = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        st_h = st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
        pr_h = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        costs = torch.zeros(NK, device="cuda")
        prev_la = cur_la_k.clone()
        cand_flat = action_cand.reshape(NK)
        for h in range(H):
            s = step_idx + h
            if s >= dg["target_lataccel"].shape[1]:
                break
            # h==0: candidate action. h>0: zero-order hold (same action held)
            cur_steer = cand_flat
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
            if h < rng_base.shape[0]:
                rng_h = rng_base[h].unsqueeze(1).expand(-1, K).reshape(NK)
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
                if h > 0:
                    costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100
            prev_la = pred_la
            act_h = a_ctx[:, 1:]
            st_h = s_ctx
            pr_h = torch.cat([pr_h[:, 1:], pred_la.unsqueeze(1)], dim=1)
            cur_la_k = pred_la

        best_idx = costs.view(N, K).argmin(dim=1)
        best_action = action_cand[torch.arange(N, device="cuda"), best_idx]
        ci = step_idx - CONTROL_START_IDX
        h_act[:, next_head] = best_action.double()
        h_act32[:, next_head] = best_action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        if 0 <= ci < N_CTRL:
            stored[:, ci] = best_action.float()
            cur_pred = sim_ref.current_lataccel.float()
            tgt_val = dg["target_lataccel"][:, step_idx].float()
            _t["costs"] += (tgt_val - cur_pred) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            if _t["prev_la"] is not None:
                _t["costs"] += ((cur_pred - _t["prev_la"]) / DEL_T) ** 2 * 100
            _t["prev_la"] = cur_pred
            if (ci + 1) % 100 == 0:
                now = time.time()
                dt = now - _t["t"]
                _t["t"] = now
                avg = (_t["costs"] / (ci + 1)).cpu()
                print(
                    f"      step {ci + 1:3d}/{N_CTRL}  mean={avg.mean().item():.1f}  med={avg.median().item():.1f}  ⏱{dt:.1f}s",
                    flush=True,
                )
        return best_action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, stored


# ═══════════════════════════════════════════════════════════
# Pass 2+: Coordinate descent on stored actions
# ═══════════════════════════════════════════════════════════


def coord_descent_pass(
    csv_files, stored_actions, mdl_path, ort_session, csv_cache, sigma
):
    """True sequential coordinate descent: perturb one step at a time, execute best.

    At each step ci:
      1. The sim state reflects all previous updates (truly sequential)
      2. Try K perturbations of the action at ci, with H-step lookahead
      3. Lookahead uses ORIGINAL stored_actions for future steps (h>0)
      4. Execute the best candidate (may differ from stored_actions[ci])
      5. The sim state at ci+1 reflects the updated action

    The key difference from "batch propose": each update is immediately executed
    in the sim, so subsequent steps see the correct updated state. The lookahead
    still uses the original stored future actions (not yet updated), which is
    conservative — it assumes the future won't change.
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N = sim.N
    K = MPC_K
    H = MPC_H
    dg = sim.data_gpu
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_session)
    orig_actions = stored_actions  # (N, N_CTRL) — original, used for lookahead h>0
    live_actions = stored_actions.clone()  # updated in-place as we go
    improved_count = 0
    _t = {"costs": torch.zeros(N, device="cuda"), "prev_la": None, "t": time.time()}

    def ctrl(step_idx, sim_ref):
        nonlocal improved_count
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        base_action = orig_actions[:, ci]  # original stored action

        # Generate K candidates around the original stored action
        noise = torch.randn(N, K, device="cuda") * sigma
        action_cand = (base_action.unsqueeze(1) + noise).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        action_cand[:, 0] = base_action  # candidate 0 = original (no change)

        # H-step lookahead using ORIGINAL stored actions for steps 1+
        NK = N * K
        start = max(0, step_idx - CL + 1)
        # NOTE: at controller call time, state_history[:, step_idx] has NOT
        # been written yet (it's written AFTER the controller returns).
        # So we read CL-1 valid entries and manually append data[step_idx].
        act_h = sim_ref.action_history[:, start:step_idx].float()  # CL-1
        st_h_prev = sim_ref.state_history[:, start:step_idx, :3].float()  # CL-1
        # Append the current step's state from data (what sim will write)
        cur_state = torch.stack(
            [
                dg["roll_lataccel"][:, step_idx].float(),
                dg["v_ego"][:, step_idx].float(),
                dg["a_ego"][:, step_idx].float(),
            ],
            dim=-1,
        )  # (N, 3)
        st_h = torch.cat([st_h_prev, cur_state.unsqueeze(1)], dim=1)  # CL entries
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
        act_h = act_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        st_h = st_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
        pr_h = pr_h.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
        cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
        rng_base = sim_ref._rng_all_gpu[step_idx - CL : step_idx - CL + H, :]

        costs = torch.zeros(NK, device="cuda")
        prev_la = cur_la_k.clone()
        for h in range(H):
            s = step_idx + h
            if s >= dg["target_lataccel"].shape[1]:
                break
            s_ci = s - CONTROL_START_IDX
            if h == 0:
                cur_steer = action_cand.reshape(NK)
            else:
                # Use LIVE actions for future steps (reflects earlier updates this pass)
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
            if h < rng_base.shape[0]:
                rng_h = rng_base[h].unsqueeze(1).expand(-1, K).reshape(NK)
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
                if h > 0:
                    costs += ((pred_la - prev_la) / DEL_T) ** 2 * 100
            prev_la = pred_la
            act_h = a_ctx[:, 1:]
            st_h = s_ctx
            pr_h = torch.cat([pr_h[:, 1:], pred_la.unsqueeze(1)], dim=1)
            cur_la_k = pred_la

        # Pick best — execute it immediately (true coordinate descent)
        cost_2d = costs.view(N, K)
        best_idx = cost_2d.argmin(dim=1)
        best_action = action_cand[torch.arange(N, device="cuda"), best_idx]
        changed = best_idx != 0
        if changed.any():
            live_actions[changed, ci] = best_action[changed]
            improved_count += changed.sum().item()

        # Execute the best action (may differ from original)
        final_action = live_actions[:, ci]

        # Progress tracking
        cur_pred = sim_ref.current_lataccel.float()
        tgt_val = dg["target_lataccel"][:, step_idx].float()
        _t["costs"] += (tgt_val - cur_pred) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
        if _t["prev_la"] is not None:
            _t["costs"] += ((cur_pred - _t["prev_la"]) / DEL_T) ** 2 * 100
        _t["prev_la"] = cur_pred
        if (ci + 1) % 100 == 0:
            now = time.time()
            dt = now - _t["t"]
            _t["t"] = now
            avg = (_t["costs"] / (ci + 1)).cpu()
            print(
                f"      step {ci + 1:3d}/{N_CTRL}  mean={avg.mean().item():.1f}  med={avg.median().item():.1f}  imp={improved_count}  ⏱{dt:.1f}s",
                flush=True,
            )

        return final_action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    print(f"      updated {improved_count}/{N * N_CTRL} steps")
    return costs, live_actions


# ═══════════════════════════════════════════════════════════
# Pure replay: just execute stored actions, measure cost
# ═══════════════════════════════════════════════════════════


def pure_replay(csv_files, stored_actions, mdl_path, ort_session, csv_cache):
    """Feed stored actions to a fresh sim. No MPC, no perturbation.
    Returns the actual trajectory cost. This is the ground truth."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N = sim.N
    actions = stored_actions  # (N, N_CTRL)

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")
        return actions[:, ci].double()

    costs = sim.rollout(ctrl)["total_cost"]
    return costs


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════


def save_checkpoint(stored_actions, costs, pass_num):
    """Save actions + costs to checkpoint dir."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"pass{pass_num:03d}.pt"
    torch.save(
        {"actions": stored_actions.cpu(), "costs": np.array(costs), "pass": pass_num},
        path,
    )
    # Also save as "latest" for easy resume
    latest = SAVE_DIR / "latest.pt"
    torch.save(
        {"actions": stored_actions.cpu(), "costs": np.array(costs), "pass": pass_num},
        latest,
    )


def load_checkpoint():
    """Load latest checkpoint. Returns (stored_actions, costs, pass_num) or None."""
    latest = SAVE_DIR / "latest.pt"
    if not latest.exists():
        return None
    ckpt = torch.load(latest, weights_only=False, map_location=DEV)
    print(
        f"  Resumed from {latest} (pass {ckpt['pass']}, mean={np.mean(ckpt['costs']):.1f})"
    )
    return ckpt["actions"].to(DEV), ckpt["costs"], ckpt["pass"]


def main():
    if PASS_SIGMAS:
        sigmas = [float(x) for x in PASS_SIGMAS.split(",")]
    else:
        # Pass 1 uses MPC_SIGMA, then NUM_CD coord descent passes with geometric decay
        cd_sigmas = np.geomspace(SIGMA_START, SIGMA_END, NUM_CD).tolist()
        sigmas = [MPC_SIGMA] + cd_sigmas

    print(
        f"exp101 — MPC + CD: K={MPC_K} H={MPC_H} passes={len(sigmas)} "
        f"σ=[{sigmas[0]:.4f}→{sigmas[-1]:.4f}]"
    )

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    start_pass = 0  # 0 = need pass 1, 1 = start from pass 2, etc.
    stored_actions = None
    costs = None

    # Try resume
    if RESUME:
        loaded = load_checkpoint()
        if loaded is not None:
            stored_actions, costs, start_pass = loaded

    # Pass 1: policy MPC (skip if resumed past it)
    if start_pass < 1:
        ac = ActorCritic().to(DEV)
        ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt["ac"])
        ac.eval()
        ds = float(ckpt.get("delta_scale", DELTA_SCALE))

        t0 = time.time()
        costs, stored_actions = mpc_pass1(
            all_csv, ac, mdl_path, ort_sess, csv_cache, ds=ds, sigma=sigmas[0]
        )
        dt = time.time() - t0
        print(f"  Pass  1  σ={sigmas[0]}  mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")
        save_checkpoint(stored_actions, costs, 1)
        if DEBUG:
            for i, c in enumerate(costs):
                print(f"    [{i}] {all_csv[i].name}  cost={c:.1f}")
        if VERIFY:
            replay_costs = pure_replay(
                all_csv, stored_actions, mdl_path, ort_sess, csv_cache
            )
            print(f"    verify: Δ={abs(np.mean(replay_costs) - np.mean(costs)):.2f}")
        start_pass = 1

    # Pass 2+: true sequential coordinate descent
    for p in range(max(1, start_pass), len(sigmas)):
        sig = sigmas[p]
        t0 = time.time()
        costs, stored_actions = coord_descent_pass(
            all_csv, stored_actions, mdl_path, ort_sess, csv_cache, sigma=sig
        )
        dt = time.time() - t0
        print(f"  Pass {p + 1:2d}  σ={sig}  mean={np.mean(costs):.1f}  ⏱{dt:.0f}s")
        save_checkpoint(stored_actions, costs, p + 1)
        if DEBUG:
            for i, c in enumerate(costs):
                print(f"    [{i}] {all_csv[i].name}  cost={c:.1f}")
        if VERIFY:
            verify_costs = pure_replay(
                all_csv, stored_actions, mdl_path, ort_sess, csv_cache
            )
            print(f"    verify: Δ={abs(np.mean(verify_costs) - np.mean(costs)):.2f}")

    print(f"\nFinal mean: {np.mean(costs):.1f}")


if __name__ == "__main__":
    main()
