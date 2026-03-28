# exp077 — TBPTT through differentiable physics
#
# exp055 exactly as-is (same arch, same obs, same single-step delta output),
# but trained with TBPTT through a differentiable PyTorch copy of the ONNX
# physics model.
#
# The gradient flows: cost → E[lataccel] → physics_model → steer_action → policy
#
# At each step the policy produces a steer delta. The delta enters the
# differentiable physics model which predicts E[lataccel]. The tracking +
# jerk cost on the diff-physics prediction is backpropped into the policy.
#
# The real ONNX sim still runs in parallel to advance the episode state.
# The diff-physics model is only used for gradient computation.
#
# Cost function matches eval exactly:
#   total = mean((target - pred)^2) * 100 * 50 + mean((diff(pred)/dt)^2) * 100
# Normalized per-step for gradient stability.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    ACC_G,
    LATACCEL_RANGE,
    VOCAB_SIZE,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture (must match exp055) ──────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))

# ── scaling (must match exp055) ───────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── obs layout (must match exp055) ────────────────────────────
C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
OBS_DIM = F_A + FUTURE_K  # 256

# ── training ──────────────────────────────────────────────────
LR = float(os.getenv("LR", "3e-5"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))
MAX_ACC_DELTA = 0.5
TEMPERATURE = 0.8

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "500"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = 100

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ══════════════════════════════════════════════════════════════
#  Policy (identical to exp055)
# ══════════════════════════════════════════════════════════════


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)
        for layer in self.actor[:-1]:
            _ortho(layer)
        _ortho(self.actor[-1], gain=0.01)

    def get_delta(self, obs):
        """Deterministic delta from Beta mean."""
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        return 2.0 * a_p / (a_p + b_p) - 1.0


# ══════════════════════════════════════════════════════════════
#  Differentiable Physics (onnx2torch, frozen)
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

    def expected_lataccel(self, states, tokens):
        """(B,20,4) float, (B,20) long -> (B,) float, differentiable w.r.t. states."""
        logits = self.model(states, tokens)
        probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)

    def tokenize(self, lataccel):
        """(B,) or (B,T) float -> long tokens."""
        clamped = lataccel.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(clamped, self.bins, right=False).clamp(0, VOCAB_SIZE - 1)


# ══════════════════════════════════════════════════════════════
#  Observation builder (identical to exp055)
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
#  TBPTT training rollout
# ══════════════════════════════════════════════════════════════


def tbptt_rollout(
    csv_files, ac, diff_phys, mdl_path, ort_session, csv_cache, optimizer
):
    """Run rollout with per-step TBPTT through differentiable physics.

    For each control step:
      1. Build obs from real sim state (detached)
      2. Policy outputs delta (with grad)
      3. Compute steer action (with grad)
      4. Build physics model context: 19 historical actions (detached) + current action (with grad)
      5. Diff physics predicts E[lataccel] (differentiable w.r.t. current action)
      6. Compute per-step cost, backprop, update policy
      7. Continue with detached action in real sim

    Each step is an independent optimization step. No multi-step unrolling.
    This avoids the exploding gradient issue entirely.
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    # Real sim stays stochastic — matches eval conditions.
    # Gradients come from diff_phys (expected-value) which is fine:
    # the gradient direction is correct, the sim just adds realistic noise.
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

    total_loss = 0.0
    n_steps = 0
    prev_diff_la = None  # previous step's diff-physics prediction, for jerk

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum, total_loss, n_steps, prev_diff_la

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

        # ── Policy forward (with gradients) ──
        obs_snap = obs_buf.clone()  # snapshot — fill_obs writes in-place
        raw = ac.get_delta(obs_snap)
        delta = raw * DELTA_SCALE
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        # ── Differentiable physics prediction ──
        # Context: 19 past actions (detached) + current action (differentiable)
        CL = CONTEXT_LENGTH
        start = max(0, step_idx - CL + 1)
        n_hist = step_idx - start  # up to 19
        pad = CL - 1 - n_hist

        act_hist = sim.action_history[:, start:step_idx].float().detach()
        if pad > 0:
            act_hist = F.pad(act_hist, (pad, 0), value=0.0)
        # Concatenate: (N, 19) detached + (N, 1) differentiable
        act_ctx = torch.cat([act_hist, action.unsqueeze(1)], dim=1)  # (N, 20)

        # States context (roll_la, v_ego, a_ego) — all from data, detached
        end = step_idx + 1
        state_ctx = sim.state_history[:, start:end].float().detach()
        if state_ctx.shape[1] < CL:
            state_ctx = F.pad(state_ctx, (0, 0, CL - state_ctx.shape[1], 0), value=0.0)

        # Full (N, 20, 4) = [steer, roll_la, v_ego, a_ego]
        full_states = torch.cat([act_ctx.unsqueeze(-1), state_ctx], dim=-1)

        # Tokens from past predictions (non-differentiable)
        pred_hist = sim.current_lataccel_history[:, start:step_idx].float().detach()
        if pred_hist.shape[1] < CL - 1:
            pred_hist = F.pad(pred_hist, (CL - 1 - pred_hist.shape[1], 0), value=0.0)
        # Last token: use current real lataccel (the one before this step's prediction)
        pred_ctx = torch.cat(
            [
                pred_hist,
                sim.current_lataccel_history[:, max(0, step_idx - 1) : step_idx]
                .float()
                .detach(),
            ],
            dim=1,
        )
        tokens = diff_phys.tokenize(pred_ctx)

        # E[lataccel] — differentiable w.r.t. action (through act_ctx -> full_states)
        pred_la = diff_phys.expected_lataccel(full_states, tokens)

        # Apply MAX_ACC_DELTA clamp
        prev_real_la = (
            sim.current_lataccel_history[:, max(0, step_idx - 1)].float().detach()
        )
        pred_la = torch.clamp(
            pred_la, prev_real_la - MAX_ACC_DELTA, prev_real_la + MAX_ACC_DELTA
        )

        # ── Per-step cost (matches eval exactly) ──
        # lat_accel_cost = mean((target - pred)^2) * 100  (accumulated, averaged at end)
        # jerk_cost = mean((diff(pred)/dt)^2) * 100       (accumulated, averaged at end)
        tgt = target.float()
        lat_cost = ((tgt - pred_la) ** 2).mean() * 100 * LAT_ACCEL_COST_MULTIPLIER

        if prev_diff_la is not None:
            jerk = (pred_la - prev_diff_la) / DEL_T
            jerk_cost = (jerk**2).mean() * 100
        else:
            jerk_cost = torch.tensor(0.0, device="cuda")

        step_cost = lat_cost + jerk_cost

        # Accumulate gradient (backward but NO optimizer step yet)
        # Divide by total control steps so the accumulated gradient is the mean
        max_ctrl_steps = COST_END_IDX - CONTROL_START_IDX
        (step_cost / max_ctrl_steps).backward()

        total_loss += step_cost.item()
        n_steps += 1
        prev_diff_la = pred_la.detach()

        # ── Update histories (all detached) ──
        act_d = action.detach()
        h_act[:, next_head] = act_d.double()
        h_act32[:, next_head] = act_d.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return act_d.double()

    optimizer.zero_grad(set_to_none=True)  # clear before rollout
    costs = sim.rollout(ctrl)["total_cost"]

    # One optimizer step per batch of routes
    nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, n_steps)
    return costs, avg_loss


# ══════════════════════════════════════════════════════════════
#  Eval (stochastic physics, no TBPTT)
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache):
    data, rng = csv_cache.slice(files)
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
            raw = ac.get_delta(obs_buf)
        delta = raw * DELTA_SCALE
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return float(np.mean(costs)), float(np.std(costs))


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════


def train():
    # ── Load policy from exp055 ──
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_keys = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    ac.load_state_dict(actor_keys, strict=False)
    print(f"Loaded policy from {BASE_PT}")

    ds = ckpt.get("delta_scale", None)
    if ds is not None:
        global DELTA_SCALE
        DELTA_SCALE = float(ds)
        print(f"  delta_scale={DELTA_SCALE:.4f}")

    print(f"  Actor params: {sum(p.numel() for p in ac.actor.parameters()):,}")

    # ── Differentiable physics ──
    diff_phys = DiffPhysics(ROOT / "models" / "tinyphysics.onnx").to(DEV)
    print(f"Loaded differentiable physics model")

    # ── Sim ──
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    # ── Data ──
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # ── Optimizer ──
    optimizer = optim.Adam(ac.actor.parameters(), lr=LR)

    # ── Baseline ──
    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, "init"
    print(f"\nBaseline: {vm:.1f} ± {vs:.1f}")
    print(f"\nTBPTT Training (single-step, exact cost function)")
    print(f"  lr={LR}  grad_clip={GRAD_CLIP}")
    print(f"  csvs/epoch={CSVS_EPOCH}  delta_scale={DELTA_SCALE}")
    print()

    def save_best():
        torch.save(
            {
                "ac": {f"actor.{k}": v for k, v in ac.actor.state_dict().items()},
                "delta_scale": DELTA_SCALE,
            },
            BEST_PT,
        )

    for epoch in range(MAX_EP):
        ac.train()
        t0 = time.time()
        batch = random.sample(all_csv, min(CSVS_EPOCH, len(all_csv)))
        costs, avg_loss = tbptt_rollout(
            batch, ac, diff_phys, mdl_path, ort_sess, csv_cache, optimizer
        )
        dt = time.time() - t0
        line = (
            f"E{epoch:3d}  train={np.mean(costs):6.1f}  loss={avg_loss:.2f}  ⏱{dt:.0f}s"
        )

        if epoch % EVAL_EVERY == 0:
            ac.eval()
            vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"

        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")


if __name__ == "__main__":
    train()
