# exp080 — MPC using diff physics for lookahead, real sim for execution
#
# At each step t:
#   1. Policy proposes mean action
#   2. Generate K candidates around the mean
#   3. For each candidate, simulate H steps forward using diff physics (expected value)
#   4. Pick the candidate with lowest H-step cost
#   5. Execute that action in the real ONNX sim
#   6. Record (obs, action) for BC distillation
#
# The diff physics model is fast (GPU batched). The real sim advances
# one step at a time. MPC replans every step, correcting for model drift.
#
# After collecting MPC-improved actions on many routes, BC-distill into policy.

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
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

TEMPERATURE = 0.8
N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

# ── Config ────────────────────────────────────────────────────
MPC_H = int(os.getenv("MPC_H", "10"))  # lookahead horizon
MPC_K = int(os.getenv("MPC_K", "128"))  # candidates per step
MPC_SIGMA = float(os.getenv("MPC_SIGMA", "0.03"))  # noise σ in raw delta space
N_ROUTES = int(os.getenv("N_ROUTES", "100"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "10"))  # routes in parallel

# ── Policy architecture (must match exp055) ───────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = 0.25
S_LAT, S_STEER, S_VEGO, S_AEGO, S_ROLL, S_CURV = 5.0, 2.0, 40.0, 4.0, 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "mpc_actions"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ══════════════════════════════════════════════════════════════
#  Diff physics model (for MPC lookahead)
# ══════════════════════════════════════════════════════════════


class DiffPhysics(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        import onnx2torch

        self.model = onnx2torch.convert(str(onnx_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(
            np.float32
        )
        self.register_buffer("bins", torch.from_numpy(bins))

    @torch.no_grad()
    def step(self, states, tokens):
        """Single physics step. states: (B,20,4), tokens: (B,20) -> E[lataccel] (B,)"""
        logits = self.model(states, tokens)
        probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)

    def tokenize(self, lataccel):
        clamped = lataccel.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(clamped, self.bins, right=False).clamp(0, VOCAB_SIZE - 1)


# ══════════════════════════════════════════════════════════════
#  MPC H-step lookahead evaluation
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def mpc_evaluate_candidates(
    diff_phys,
    candidate_first_actions,
    policy_ac,
    ds,
    # Current sim state for context:
    action_history,
    state_history,
    pred_history,
    current_la,
    target_future,
    data_future,
    step_idx,
):
    """Evaluate K candidate first-actions by simulating H steps with diff physics.

    candidate_first_actions: (K,) — raw delta candidates for step t
    Returns: costs (K,) — H-step cost for each candidate
    """
    K = candidate_first_actions.shape[0]
    H = MPC_H
    CL = CONTEXT_LENGTH

    # Simulate H steps for K candidates in parallel
    # action_history (CL,), state_history (CL, 3)=[roll,v,a], pred_history (CL,)
    act_hist = action_history.unsqueeze(0).expand(K, -1).clone()  # (K, CL)
    st_hist = state_history.unsqueeze(0).expand(K, -1, -1).clone()  # (K, CL, 3)
    pred_hist = pred_history.unsqueeze(0).expand(K, -1).clone()  # (K, CL)
    cur_la = current_la.unsqueeze(0).expand(K).clone()  # (K,)
    prev_action = act_hist[:, -1].clone()  # (K,)

    total_cost = torch.zeros(K, device=DEV)
    prev_la_for_jerk = cur_la.clone()

    for h in range(H):
        t = step_idx + h
        if t >= COST_END_IDX:
            break

        if h == 0:
            raw = candidate_first_actions
        else:
            raw = torch.zeros(K, device=DEV)  # hold action for future steps

        delta = raw * ds
        action = (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # Shift action context: drop oldest, append new
        act_ctx = torch.cat([act_hist[:, 1:], action.unsqueeze(1)], dim=1)  # (K, CL)

        # Shift state context (roll_la, v_ego, a_ego): append data values for step t
        ctrl_idx = t - CONTROL_START_IDX
        fut_idx = min(ctrl_idx, data_future.shape[0] - 1)
        new_st = data_future[fut_idx].unsqueeze(0).expand(K, -1)  # (K, 3)
        st_ctx = torch.cat(
            [st_hist[:, 1:, :], new_st.unsqueeze(1)], dim=1
        )  # (K, CL, 3)

        # Build full (K, CL, 4) states: [action, roll_la, v_ego, a_ego]
        full_states = torch.cat([act_ctx.unsqueeze(-1), st_ctx], dim=-1)  # (K, CL, 4)

        # Tokens from prediction history
        tokens = diff_phys.tokenize(pred_hist)  # (K, CL)

        # Physics prediction
        pred_la = diff_phys.step(full_states, tokens)
        pred_la = torch.clamp(pred_la, cur_la - MAX_ACC_DELTA, cur_la + MAX_ACC_DELTA)

        # Per-step cost
        tgt_idx = min(ctrl_idx, target_future.shape[0] - 1)
        target_la = target_future[tgt_idx]
        lat_err = (target_la - pred_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
        jerk = ((pred_la - prev_la_for_jerk) / DEL_T) ** 2 * 100
        total_cost += lat_err + jerk

        # Advance state
        prev_la_for_jerk = pred_la
        act_hist = act_ctx
        st_hist = st_ctx
        pred_hist = torch.cat([pred_hist[:, 1:], pred_la.unsqueeze(1)], dim=1)
        cur_la = pred_la
        prev_action = action

    return total_cost / min(H, COST_END_IDX - step_idx)


# ══════════════════════════════════════════════════════════════
#  Policy
# ══════════════════════════════════════════════════════════════


class PolicyActor(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)

    def get_delta(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        return 2.0 * a_p / (a_p + b_p) - 1.0


# ══════════════════════════════════════════════════════════════
#  Obs builder (for policy query during MPC rollout)
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _w(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        k: _w(dg[k]) for k in ("target_lataccel", "roll_lataccel", "v_ego", "a_ego")
    }


def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale
        return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale


def fill_obs(
    buf,
    target,
    current,
    roll_la,
    v_ego,
    a_ego,
    h_act,
    h_lat,
    hist_head,
    ei,
    future,
    step_idx,
):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    pa = h_act[:, hist_head]
    pa2 = h_act[:, (hist_head - 1) % HIST_LEN]
    pl = h_lat[:, hist_head]
    buf[:, 0] = target / S_LAT
    buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV
    buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO
    buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL
    buf[:, 9] = pa / S_STEER
    buf[:, 10] = ei / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - pl) / DEL_T / S_LAT
    buf[:, 13] = (pa - pa2) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)
    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  MPC rollout: real sim execution with MPC action selection
# ══════════════════════════════════════════════════════════════


def mpc_rollout(
    csv_files, ac, diff_phys, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """Run MPC on a batch of routes. At each control step, use diff physics
    to evaluate K candidates, pick the best, execute in real sim.

    Returns: costs, obs (for BC), raw actions (for BC)
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
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

    all_obs = torch.empty((N_CTRL, N, OBS_DIM), dtype=torch.float32, device="cuda")
    all_raw = torch.empty((N_CTRL, N), dtype=torch.float32, device="cuda")
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        roll_la = dg["roll_lataccel"][:, step_idx]
        v_ego = dg["v_ego"][:, step_idx]
        a_ego = dg["a_ego"][:, step_idx]

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
            roll_la.float(),
            v_ego.float(),
            a_ego.float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )

        # Get policy mean as baseline
        with torch.inference_mode():
            policy_raw = ac.get_delta(obs_buf)  # (N,)

        # MPC: for each route, evaluate K candidates using diff physics
        best_raw = torch.empty(N, device="cuda")

        for r in range(N):
            # Generate K candidates around policy mean
            mean = policy_raw[r].item()
            candidates = torch.randn(MPC_K, device="cuda") * MPC_SIGMA + mean
            candidates[0] = mean  # always include the mean
            candidates = candidates.clamp(-1.0, 1.0)

            # Get current context from real sim for this route
            CL = CONTEXT_LENGTH
            act_h = sim.action_history[r, max(0, step_idx - CL + 1) : step_idx].float()
            if act_h.shape[0] < CL:
                act_h = F.pad(act_h, (CL - act_h.shape[0], 0), value=0.0)

            state_h = sim.state_history[
                r, max(0, step_idx - CL + 1) : step_idx, :3
            ].float()
            if state_h.shape[0] < CL:
                state_h = F.pad(state_h, (0, 0, CL - state_h.shape[0], 0), value=0.0)

            pred_h = sim.current_lataccel_history[
                r, max(0, step_idx - CL) : step_idx
            ].float()
            if pred_h.shape[0] < CL:
                pred_h = F.pad(pred_h, (CL - pred_h.shape[0], 0), value=0.0)

            cur_la_r = sim.current_lataccel_history[r, step_idx - 1].float()

            # Target future for cost computation
            ctrl_idx = step_idx - CONTROL_START_IDX
            tgt_future = dg["target_lataccel"][
                r, CONTROL_START_IDX:COST_END_IDX
            ].float()

            # Data future (roll_la, v_ego, a_ego) for state context in lookahead
            data_future = torch.stack(
                [
                    dg["roll_lataccel"][r, CONTROL_START_IDX:COST_END_IDX].float(),
                    dg["v_ego"][r, CONTROL_START_IDX:COST_END_IDX].float(),
                    dg["a_ego"][r, CONTROL_START_IDX:COST_END_IDX].float(),
                ],
                dim=-1,
            )  # (400, 3)

            costs = mpc_evaluate_candidates(
                diff_phys,
                candidates,
                ac,
                ds,
                act_h,
                state_h,
                pred_h,
                cur_la_r,
                tgt_future,
                data_future,
                step_idx,
            )

            best_raw[r] = candidates[costs.argmin()]

        delta = best_raw * ds
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        if step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_raw[si] = best_raw
            si += 1

        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, all_obs[:si].cpu(), all_raw[:si].cpu()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def main():
    print(f"exp080 — MPC with diff physics lookahead + real sim execution")
    print(f"  H={MPC_H}  K={MPC_K}  σ={MPC_SIGMA}  routes={N_ROUTES}")

    # Load diff physics
    diff_phys = DiffPhysics(ROOT / "models" / "tinyphysics.onnx").to(DEV)
    print(f"Loaded diff physics on {DEV}")

    # Load policy
    ac = PolicyActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_keys = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    ac.load_state_dict(actor_keys, strict=False)
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy from {BASE_PT} (Δs={ds})")

    # Real sim
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_cache = CSVCache([str(f) for f in all_csv])

    # First get baseline policy costs
    print("\nBaseline policy rollout...")
    from tinyphysics_batched import BatchedSimulator

    va_files = all_csv[:N_ROUTES]

    routes = all_csv[:N_ROUTES]
    OUT_DIR.mkdir(exist_ok=True)

    all_mpc_costs = []
    all_mpc_obs = []
    all_mpc_raw = []

    for i in range(0, len(routes), BATCH_ROUTES):
        batch = routes[i : i + BATCH_ROUTES]
        print(f"\nRoutes {i}..{i + len(batch) - 1}")
        t0 = time.time()
        costs, obs, raw = mpc_rollout(
            batch, ac, diff_phys, mdl_path, ort_sess, csv_cache, ds=ds
        )
        dt = time.time() - t0
        print(
            f"  MPC costs: {[f'{c:.1f}' for c in costs]}  mean={np.mean(costs):.1f}  ⏱{dt:.0f}s"
        )

        all_mpc_costs.extend(costs.tolist())
        all_mpc_obs.append(obs)
        all_mpc_raw.append(raw)

        if (i + BATCH_ROUTES) % 50 == 0 or i + BATCH_ROUTES >= len(routes):
            torch.save(
                {
                    "obs": torch.cat(
                        [o.permute(1, 0, 2).reshape(-1, OBS_DIM) for o in all_mpc_obs],
                        dim=0,
                    ),
                    "raw": torch.cat([r.T.reshape(-1) for r in all_mpc_raw], dim=0),
                    "costs": np.array(all_mpc_costs),
                },
                OUT_DIR / "mpc_data.pt",
            )
            print(
                f"  [{len(all_mpc_costs)} routes] mean MPC cost: {np.mean(all_mpc_costs):.1f}"
            )

    print(f"\nDone. {len(routes)} routes. Mean MPC cost: {np.mean(all_mpc_costs):.1f}")


if __name__ == "__main__":
    main()
