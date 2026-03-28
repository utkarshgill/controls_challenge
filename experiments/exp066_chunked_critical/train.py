# exp066 — exp055 with chunked Beta PPO + critical-phase mining
#
# Minimal fork of exp055:
# - policy outputs CHUNK_K future deltas at once (still Beta-distributed)
# - PPO operates on chunk decisions instead of single steps
# - actor training focuses on high-criticality windows first, then all windows
# - optional initialization from exp055 checkpoint tiles the 1-step head into K steps

import numpy as np, pandas as pd, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path
from tqdm.contrib.concurrent import process_map

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
    State,
    FuturePlan,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS, C_LAYERS = 4, 4
CHUNK_K = int(os.getenv("CHUNK_K", "5"))
assert (COST_END_IDX - CONTROL_START_IDX) % CHUNK_K == 0, "CHUNK_K must divide scored horizon"
DELTA_SCALE_MAX = float(os.getenv("DELTA_SCALE_MAX", "0.25"))
DELTA_SCALE_MIN = float(os.getenv("DELTA_SCALE_MIN", "0.25"))
INIT_FROM_EXP055 = os.getenv("INIT_FROM_EXP055", "1") == "1"
RESIDUAL_SCALE = float(os.getenv("RESIDUAL_SCALE", "0.05"))
ACTOR_REF_INPUT = os.getenv("ACTOR_REF_INPUT", "1") == "1"

# ── critical-phase mining ─────────────────────────────────────
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", str(max(2 * CHUNK_K, 8))))
CRIT_TOP_FRAC = float(os.getenv("CRIT_TOP_FRAC", "0.35"))
CRIT_ONLY_EPOCHS = int(os.getenv("CRIT_ONLY_EPOCHS", "15"))
CRIT_WEIGHT = float(os.getenv("CRIT_WEIGHT", "2.0"))

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── PPO ───────────────────────────────────────────────────────
PI_LR = float(os.getenv("PI_LR", "3e-4"))
VF_LR = float(os.getenv("VF_LR", "3e-4"))
LR_MIN = 5e-5
GAMMA = float(os.getenv("GAMMA", "0.95"))
LAMDA = float(os.getenv("LAMDA", "0.9"))
K_EPOCHS = 4
EPS_CLIP = 0.2
VF_COEF = 1.0
ENT_COEF = float(os.getenv("ENT_COEF", "0.003"))
SIGMA_FLOOR = float(os.getenv("SIGMA_FLOOR", "0.01"))
SIGMA_FLOOR_COEF = float(os.getenv("SIGMA_FLOOR_COEF", "0.5"))
ACT_SMOOTH = float(os.getenv("ACT_SMOOTH", "0.0"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1.0"))
MINI_BS = int(os.getenv("MINI_BS", "25_000"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "20"))
BC_LR = float(os.getenv("BC_LR", "0.01"))
BC_BS = int(os.getenv("BC_BS", "2048"))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "10"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = 5
EVAL_N = 100
RESUME = os.getenv("RESUME", "0") == "1"
RESUME_OPT = os.getenv("RESUME_OPT", "1") == "1"
RESUME_DS = os.getenv("RESUME_DS", "0") == "1"
RESET_CRITIC = os.getenv("RESET_CRITIC", "0") == "1"
RESUME_WARMUP = os.getenv("RESUME_WARMUP", "0") == "1"
LR_DECAY = os.getenv("LR_DECAY", "1") == "1"
DELTA_SCALE_DECAY = os.getenv("DELTA_SCALE_DECAY", "0") == "1"
REWARD_RMS_NORM = os.getenv("REWARD_RMS_NORM", "1") == "1"
ADV_NORM = os.getenv("ADV_NORM", "1") == "1"
COMPILE = os.getenv("COMPILE", "0") == "1"
USE_BASE_REWARD_BASELINE = os.getenv("USE_BASE_REWARD_BASELINE", "1") == "1"
POS_ADV_ONLY = os.getenv("POS_ADV_ONLY", "1") == "1"
POS_ADV_MARGIN = float(os.getenv("POS_ADV_MARGIN", "0.0"))
RESIDUAL_BC_COEF = float(os.getenv("RESIDUAL_BC_COEF", "0.0"))


def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout offsets ────────────────────────────────────────
C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
OBS_DIM = F_A + FUTURE_K


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_actor = BaseActor()
        actor_in_dim = STATE_DIM + (CHUNK_K if ACTOR_REF_INPUT else 0)
        a = [nn.Linear(actor_in_dim, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2 * CHUNK_K))
        self.actor = nn.Sequential(*a)

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        for layer in self.actor[:-1]:
            _ortho(layer)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        for layer in self.critic[:-1]:
            _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

    def beta_params(self, obs):
        actor_in = obs
        if ACTOR_REF_INPUT:
            base_ref = self.base_raw(obs).unsqueeze(-1).expand(-1, CHUNK_K)
            actor_in = torch.cat([obs, base_ref], dim=-1)
        out = self.actor(actor_in)
        alpha = F.softplus(out[..., :CHUNK_K]) + 1.0
        beta = F.softplus(out[..., CHUNK_K:]) + 1.0
        return alpha, beta

    def base_raw(self, obs):
        return self.base_actor.raw(obs)


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)

        for layer in self.actor[:-1]:
            _ortho(layer)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

        for p in self.parameters():
            p.requires_grad_(False)

    def raw(self, obs):
        out = self.actor(obs)
        alpha = F.softplus(out[..., 0]) + 1.0
        beta = F.softplus(out[..., 1]) + 1.0
        return 2.0 * alpha / (alpha + beta) - 1.0


def maybe_init_from_exp055(ac):
    ckpt_path = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
    if not ckpt_path.exists():
        print("exp055 init skipped: checkpoint not found")
        return False

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    src = ckpt["ac"]
    dst = ac.state_dict()

    for key, value in src.items():
        if key.startswith("critic.") and key in dst and dst[key].shape == value.shape:
            dst[key] = value.clone()
        elif key.startswith("actor."):
            base_key = f"base_actor.{key}"
            if base_key in dst and dst[base_key].shape == value.shape:
                dst[base_key] = value.clone()
            if key in dst and dst[key].shape == value.shape:
                dst[key] = value.clone()

    head_w_key = f"actor.{len(ac.actor) - 1}.weight"
    head_b_key = f"actor.{len(ac.actor) - 1}.bias"
    if head_w_key in dst:
        dst[head_w_key].zero_()
    if head_b_key in dst:
        dst[head_b_key].zero_()

    ac.load_state_dict(dst)
    print(f"Initialized from exp055 checkpoint: {ckpt_path.name} (frozen base + zero residual)")
    return True


def load_resume_checkpoint(ac, ckpt):
    state = ckpt["ac"]
    has_base = any(k.startswith("base_actor.") for k in state)
    if has_base:
        try:
            missing, unexpected = ac.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"Resume checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
            return {"legacy": False}
        except RuntimeError:
            dst = ac.state_dict()
            loaded = 0
            for key, value in state.items():
                if key in dst and dst[key].shape == value.shape:
                    dst[key] = value.clone()
                    loaded += 1
            ac.load_state_dict(dst, strict=False)
            print(f"Checkpoint partially loaded after architecture change: reused {loaded} tensors with matching shape")
            return {"legacy": True}

    if not maybe_init_from_exp055(ac):
        raise RuntimeError("Legacy exp066 checkpoint cannot be migrated because exp055 checkpoint is unavailable")

    print("Legacy exp066 checkpoint detected: migrated to frozen-base residual init; old chunk actor ignored")
    return {"legacy": True}


# ══════════════════════════════════════════════════════════════
#  Observation builder (GPU, batched)
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _windows(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        "target_lataccel": _windows(dg["target_lataccel"]),
        "roll_lataccel": _windows(dg["roll_lataccel"]),
        "v_ego": _windows(dg["v_ego"]),
        "a_ego": _windows(dg["a_ego"]),
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
    error_integral,
    future,
    step_idx,
):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    prev_act = h_act[:, hist_head]
    prev_act2 = h_act[:, (hist_head - 1) % HIST_LEN]
    prev_lat = h_lat[:, hist_head]

    buf[:, 0] = target / S_LAT
    buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV
    buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO
    buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL
    buf[:, 9] = prev_act / S_STEER
    buf[:, 10] = error_integral / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - prev_lat) / DEL_T / S_LAT
    buf[:, 13] = (prev_act - prev_act2) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)

    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)

    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO

    buf.clamp_(-5.0, 5.0)


def _chunk_criticality(future_target, current):
    horizon = min(CRIT_LOOKAHEAD, future_target.shape[1])
    fp = future_target[:, :horizon]
    if horizon <= 1:
        return (fp[:, 0] - current).abs() / S_LAT

    slope = torch.diff(fp, dim=1, prepend=fp[:, :1]) / DEL_T
    mismatch = (fp[:, min(CHUNK_K - 1, horizon - 1)] - current).abs() / S_LAT
    span = (fp[:, -1] - fp[:, 0]).abs() / S_LAT
    peak_slope = slope.abs().amax(dim=1) / max(S_LAT / DEL_T, 1e-6)
    flip = ((fp[:, :-1] * fp[:, 1:]) < 0).any(dim=1).float()
    return mismatch + 0.5 * span + 0.5 * peak_slope + 0.5 * flip


# ══════════════════════════════════════════════════════════════
#  GPU Rollout
# ══════════════════════════════════════════════════════════════


def rollout(
    csv_files,
    ac,
    mdl_path,
    ort_session,
    csv_cache,
    deterministic=False,
    ds=DELTA_SCALE_MAX,
    base_only=False,
    return_rew=False,
):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng)
    N = sim.N
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    n_chunks = max_steps // CHUNK_K
    future = _precompute_future_windows(dg)
    collect_policy = (not deterministic) and (not base_only)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    if collect_policy:
        all_obs = torch.empty((n_chunks, N, OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw = torch.empty((n_chunks, N, CHUNK_K), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((n_chunks, N), dtype=torch.float32, device="cuda")
        all_val = torch.empty((n_chunks, N), dtype=torch.float32, device="cuda")
        all_crit = torch.empty((n_chunks, N), dtype=torch.float32, device="cuda")

    chunk_idx = 0
    hist_head = HIST_LEN - 1
    planned_raw = None
    step_in_chunk = 0

    def ctrl(step_idx, sim_ref):
        nonlocal chunk_idx, hist_head, err_sum, planned_raw, step_in_chunk

        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        roll_la = dg["roll_lataccel"][:, step_idx]
        v_ego = dg["v_ego"][:, step_idx]
        a_ego = dg["a_ego"][:, step_idx]

        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_error = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_error
        ei = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        ctrl_step = step_idx - CONTROL_START_IDX
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

        with torch.inference_mode():
            base_raw = ac.base_raw(obs_buf)
            if (not base_only) and ctrl_step < max_steps and ctrl_step % CHUNK_K == 0:
                alpha, beta = ac.beta_params(obs_buf)
                if collect_policy:
                    val = ac.critic(obs_buf).squeeze(-1)

        if (not base_only) and ctrl_step < max_steps and ctrl_step % CHUNK_K == 0:
            if deterministic:
                x = alpha / (alpha + beta)
                raw = 2.0 * x - 1.0
                planned_raw = raw
            else:
                dist = torch.distributions.Beta(alpha, beta)
                x = dist.sample()
                raw = 2.0 * x - 1.0
                logp = dist.log_prob(x).sum(dim=-1)
                planned_raw = raw

                if collect_policy:
                    all_obs[chunk_idx] = obs_buf
                    all_raw[chunk_idx] = raw
                    all_logp[chunk_idx] = logp
                    all_val[chunk_idx] = val
                    all_crit[chunk_idx] = _chunk_criticality(future["target_lataccel"][:, step_idx], cur32)
                    chunk_idx += 1

            step_in_chunk = 0

        if (not base_only) and planned_raw is not None and ctrl_step < max_steps and step_in_chunk < CHUNK_K:
            residual_raw = planned_raw[:, step_in_chunk]
            step_in_chunk += 1
        else:
            residual_raw = torch.zeros_like(base_raw)

        raw = (base_raw + residual_raw * RESIDUAL_SCALE).clamp(-1.0, 1.0)
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]

    if deterministic and not return_rew:
        return costs.tolist()

    start = CONTROL_START_IDX
    end = start + max_steps
    pred = sim.current_lataccel_history[:, start:end].float()
    target = dg["target_lataccel"][:, start:end].float()
    act = sim.action_history[:, start:end].float()

    lat_r = (target - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    rew_step = (-(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(REWARD_SCALE, 1e-8)).float()
    rew = rew_step.reshape(N, n_chunks, CHUNK_K).sum(dim=-1)
    if deterministic and return_rew:
        return dict(rew=rew, costs=costs)

    dones = torch.zeros((N, chunk_idx), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    return dict(
        obs=all_obs[:chunk_idx].permute(1, 0, 2).reshape(-1, OBS_DIM),
        raw=all_raw[:chunk_idx].permute(1, 0, 2).reshape(-1, CHUNK_K),
        old_logp=all_logp[:chunk_idx].T.reshape(-1),
        val_2d=all_val[:chunk_idx].T,
        rew=rew[:, :chunk_idx],
        done=dones,
        crit=all_crit[:chunk_idx].T.reshape(-1),
        costs=costs,
    )


# ══════════════════════════════════════════════════════════════
#  BC Pretrain
# ══════════════════════════════════════════════════════════════


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    if vals is not None and len(vals) > 0:
        arr = np.asarray(vals, np.float32)
        return np.pad(arr, (0, k - len(arr)), "edge")
    return np.full(k, fallback, dtype=np.float32)


def _build_obs_bc(target, current, state, fplan, hist_act, hist_lat):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    flat = getattr(fplan, "lataccel", None)
    fp0 = flat[0] if (flat and len(flat) > 0) else target
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0

    core = np.array(
        [
            target / S_LAT,
            current / S_LAT,
            (target - current) / S_LAT,
            k_tgt / S_CURV,
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            state.v_ego / S_VEGO,
            state.a_ego / S_AEGO,
            state.roll_lataccel / S_ROLL,
            hist_act[-1] / S_STEER,
            0.0,
            (fp0 - target) / DEL_T / S_LAT,
            (current - hist_lat[-1]) / DEL_T / S_LAT,
            (hist_act[-1] - hist_act[-2]) / DEL_T / S_STEER,
            fric,
            max(0.0, 1.0 - fric),
        ],
        dtype=np.float32,
    )

    obs = np.concatenate(
        [
            core,
            np.array(hist_act, np.float32) / S_STEER,
            np.array(hist_lat, np.float32) / S_LAT,
            _future_raw(fplan, "lataccel", target) / S_LAT,
            _future_raw(fplan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
            _future_raw(fplan, "v_ego", state.v_ego) / S_VEGO,
            _future_raw(fplan, "a_ego", state.a_ego) / S_AEGO,
        ]
    )
    return np.clip(obs, -5.0, 5.0)


def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values
    steer = -df["steerCommand"].values

    obs_list, raw_list = [], []
    h_act = [0.0] * HIST_LEN
    h_lat = [0.0] * HIST_LEN

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        target_la = tgt[step_idx]
        state = State(
            roll_lataccel=roll_la[step_idx],
            v_ego=v_ego[step_idx],
            a_ego=a_ego[step_idx],
        )
        fplan = FuturePlan(
            lataccel=tgt[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
        )

        if step_idx + CHUNK_K <= CONTROL_START_IDX and (step_idx - CONTEXT_LENGTH) % CHUNK_K == 0:
            obs = _build_obs_bc(target_la, target_la, state, fplan, h_act, h_lat)
            prev_action = h_act[-1]
            raw_chunk = []
            for j in range(CHUNK_K):
                next_action = steer[step_idx + j]
                raw_chunk.append(np.clip((next_action - prev_action) / DELTA_SCALE_MAX, -1.0, 1.0))
                prev_action = next_action
            obs_list.append(obs)
            raw_list.append(raw_chunk)

        h_act = h_act[1:] + [steer[step_idx]]
        h_lat = h_lat[1:] + [tgt[step_idx]]

    return np.array(obs_list, np.float32), np.array(raw_list, np.float32)


def pretrain_bc(ac, all_csvs):
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(_bc_worker, [str(f) for f in all_csvs], max_workers=10, chunksize=50, disable=False)
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    N = len(all_obs)
    print(f"BC pretrain: {N} chunk samples, {BC_EPOCHS} epochs")

    obs_t = torch.FloatTensor(all_obs).to(DEV)
    raw_t = torch.FloatTensor(all_raw).to(DEV)
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            alpha, beta = ac.beta_params(obs_t[idx])
            x = ((raw_t[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(alpha, beta).log_prob(x).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        print(f"  BC epoch {ep}: loss={total / nb:.6f}  lr={opt.param_groups[0]['lr']:.1e}")
    print("BC pretrain done.\n")


# ══════════════════════════════════════════════════════════════
#  PPO
# ══════════════════════════════════════════════════════════════


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / tot
        ) / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(ac.actor.parameters(), lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._rms = RunningMeanStd()

    @staticmethod
    def _beta_sigma_raw(alpha, beta):
        return 2.0 * torch.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

    def _gae(self, rew, val, done):
        if REWARD_RMS_NORM:
            with torch.no_grad():
                flat = rew.reshape(-1)
                self._rms.update(flat.mean().item(), flat.var().item(), flat.numel())
            rew = rew / max(self._rms.std, 1e-8)
        N, S = rew.shape
        adv = torch.empty_like(rew)
        g = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(S - 1, -1, -1):
            nv = val[:, t + 1] if t < S - 1 else g
            mask = 1.0 - done[:, t]
            g = (rew[:, t] + GAMMA * nv * mask - val[:, t]) + GAMMA * LAMDA * mask * g
            adv[:, t] = g
        return adv.reshape(-1), (adv + val).reshape(-1)

    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX, epoch=0):
        obs = gd["obs"]
        raw = gd["raw"]
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])
        if SAMPLES_PER_ROUTE > 1 and not gd.get("base_relative", False):
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_2d = adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)

        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)
        sigma_pen = torch.tensor(0.0, device="cuda")

        crit = gd.get("crit", None)
        if crit is None:
            crit = torch.ones(len(obs), dtype=torch.float32, device="cuda")
        crit = crit.float()
        crit_norm = crit / (crit.mean() + 1e-8)
        crit_norm = crit_norm.clamp(min=0.0, max=5.0)
        if CRIT_TOP_FRAC < 1.0 and epoch < CRIT_ONLY_EPOCHS:
            threshold = torch.quantile(crit, max(0.0, 1.0 - CRIT_TOP_FRAC))
            actor_keep = crit >= threshold
        else:
            threshold = crit.min()
            actor_keep = torch.ones_like(crit, dtype=torch.bool)

        n_vf = 0
        n_actor = 0
        vf_sum = 0.0
        pi_sum = 0.0
        ent_sum = 0.0
        sigma_pen_sum = 0.0
        crit_sum = 0.0
        bc_sum = 0.0

        with torch.no_grad():
            if "old_logp" in gd:
                old_lp = gd["old_logp"]
            else:
                a_old, b_old = self.ac.beta_params(obs)
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t).sum(dim=-1)

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device="cuda").split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                val = self.ac.critic(obs[idx]).squeeze(-1)
                vf_loss = F.mse_loss(val, ret_t[idx], reduction="mean")
                bs = int(idx.numel())
                vf_sum += vf_loss.detach().item() * bs
                n_vf += bs

                actor_mask = actor_keep[idx]
                if POS_ADV_ONLY:
                    actor_mask = actor_mask & (adv_t[idx] > POS_ADV_MARGIN)
                actor_idx = idx[actor_mask] if not critic_only else idx[:0]
                if critic_only or actor_idx.numel() == 0:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                    continue

                a_c, b_c = self.ac.beta_params(obs[actor_idx])
                dist = torch.distributions.Beta(a_c, b_c)
                lp = dist.log_prob(x_t[actor_idx]).sum(dim=-1)
                mean_raw = 2.0 * a_c / (a_c + b_c) - 1.0
                mb_adv_actor = adv_t[actor_idx]
                if ADV_NORM:
                    mb_adv_actor = (mb_adv_actor - mb_adv_actor.mean()) / (mb_adv_actor.std() + 1e-8)
                ratio = (lp - old_lp[actor_idx]).exp()
                weights = 1.0 + CRIT_WEIGHT * torch.clamp(crit_norm[actor_idx] - 1.0, min=0.0, max=4.0)
                surr1 = ratio * mb_adv_actor
                surr2 = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv_actor
                pi_loss = -((torch.min(surr1, surr2)) * weights).sum() / weights.sum().clamp_min(1.0)
                ent = dist.entropy().mean()
                sigma_raw_mb = self._beta_sigma_raw(a_c, b_c).mean()
                sigma_eff_mb = sigma_raw_mb * ds * RESIDUAL_SCALE
                sigma_pen = F.relu(SIGMA_FLOOR * RESIDUAL_SCALE - sigma_eff_mb)
                # Anchor the current residual mean to the frozen base policy.
                resid_bc = mean_raw.pow(2).mean()
                loss = (
                    pi_loss
                    + VF_COEF * vf_loss
                    - ENT_COEF * ent
                    + SIGMA_FLOOR_COEF * sigma_pen
                    + RESIDUAL_BC_COEF * resid_bc
                )

                self.pi_opt.zero_grad(set_to_none=True)
                self.vf_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                self.pi_opt.step()
                self.vf_opt.step()

                n_actor += int(actor_idx.numel())
                pi_sum += pi_loss.detach().item() * int(actor_idx.numel())
                ent_sum += ent.detach().item() * int(actor_idx.numel())
                sigma_pen_sum += sigma_pen.detach().item() * int(actor_idx.numel())
                crit_sum += crit[actor_idx].mean().detach().item() * int(actor_idx.numel())
                bc_sum += resid_bc.detach().item() * int(actor_idx.numel())

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs[:1000])
            sigma_raw = self._beta_sigma_raw(a_d, b_d).mean().item()
            sigma_eff = sigma_raw * ds * RESIDUAL_SCALE

        effective_keep = actor_keep
        if POS_ADV_ONLY:
            effective_keep = effective_keep & (adv_t > POS_ADV_MARGIN)
        keep_frac = effective_keep.float().mean().item()
        return dict(
            pi=(pi_sum / max(1, n_actor)) if not critic_only else 0.0,
            vf=(vf_sum / max(1, n_vf)),
            ent=(ent_sum / max(1, n_actor)) if not critic_only else 0.0,
            σ=sigma_eff,
            σraw=sigma_raw,
            σpen=(sigma_pen_sum / max(1, n_actor)) if not critic_only else 0.0,
            crit=(crit_sum / max(1, n_actor)) if not critic_only else crit.mean().item(),
            bc=(bc_sum / max(1, n_actor)) if not critic_only else 0.0,
            keep=keep_frac,
            thr=float(threshold.item()),
            lr=self.pi_opt.param_groups[0]["lr"],
        )


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
    return float(np.mean(costs)), float(np.std(costs))


def train():
    ac = ActorCritic().to(DEV)
    ppo = PPO(ac)
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    all_csv = sorted((ROOT / "data").glob("*.csv"))
    tr_f = all_csv
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    if BEST_PT.exists() and not RESUME:
        print(f"Note: {BEST_PT.name} exists but RESUME=0, so this run starts fresh from init and ignores the saved exp066 checkpoint")

    warmup_off = 0
    resumed_ds = None
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        resume_info = load_resume_checkpoint(ac, ckpt)
        legacy_resume = resume_info["legacy"]
        if RESUME_OPT and not legacy_resume and "pi_opt" in ckpt:
            ppo.pi_opt.load_state_dict(ckpt["pi_opt"])
            ppo.vf_opt.load_state_dict(ckpt["vf_opt"])
            if "ret_rms" in ckpt:
                r = ckpt["ret_rms"]
                ppo._rms.mean, ppo._rms.var, ppo._rms.count = r["mean"], r["var"], r["count"]
        elif RESUME_OPT and legacy_resume:
            print("RESUME_OPT=1 but checkpoint is legacy; optimizer and RMS use fresh state after migration")
        elif RESUME_OPT:
            print("RESUME_OPT=1 but optimizer state missing in checkpoint; using fresh optimizer/RMS state")
        warmup_off = 0 if (RESUME_WARMUP or legacy_resume) else CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
        if RESUME_OPT and not legacy_resume:
            print("RESUME_OPT=1: optimizer state, LR/eps, and RMS restored from checkpoint")
        elif not RESUME_OPT:
            print("RESUME_OPT=0: resumed weights only; optimizer and RMS use fresh state")
        else:
            print("Legacy checkpoint migration: resumed weights only; optimizer and RMS use fresh state")
        if RESUME_DS:
            ds_ckpt = ckpt.get("delta_scale", None)
            if ds_ckpt is not None:
                resumed_ds = float(ds_ckpt)
                print(f"Resumed delta_scale={resumed_ds:.6f} from checkpoint")
            else:
                print("RESUME_DS=1 but checkpoint has no delta_scale; using schedule/env")
        if RESET_CRITIC:
            for layer in ac.critic[:-1]:
                if isinstance(layer, nn.Linear):
                    _ortho(layer)
            if isinstance(ac.critic[-1], nn.Linear):
                _ortho(ac.critic[-1], gain=1.0)
            ppo.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic, vf_opt, and ret_rms reset; critic warmup re-enabled")
    else:
        if not (INIT_FROM_EXP055 and maybe_init_from_exp055(ac)):
            pretrain_bc(ac, all_csv)

    if COMPILE:
        ac.actor = torch.compile(ac.actor, mode="max-autotune-no-cudagraphs", dynamic=True)
        ac.critic = torch.compile(ac.critic, mode="max-autotune-no-cudagraphs", dynamic=True)

    ds_max_run = DELTA_SCALE_MAX
    ds_min_run = DELTA_SCALE_MIN
    if RESUME_DS and resumed_ds is not None:
        ds_max_run = resumed_ds
        ds_min_run = min(ds_min_run, ds_max_run)

    baseline_ds = (
        ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(0.0))
        if DELTA_SCALE_DECAY
        else ds_max_run
    )
    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=baseline_ds)
    best, best_ep = vm, "init"
    print(f"Baseline: {vm:.1f} ± {vs:.1f}  (Δs={baseline_ds:.4f})")

    cur_ds = ds_max_run

    def save_best():
        torch.save(
            {
                "ac": ac.state_dict(),
                "pi_opt": ppo.pi_opt.state_dict(),
                "vf_opt": ppo.vf_opt.state_dict(),
                "ret_rms": {"mean": ppo._rms.mean, "var": ppo._rms.var, "count": ppo._rms.count},
                "delta_scale": cur_ds,
                "chunk_k": CHUNK_K,
                "residual_scale": RESIDUAL_SCALE,
            },
            BEST_PT,
        )

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    n_r = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
    print(f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {n_r} routes × {SAMPLES_PER_ROUTE} = {n_r * SAMPLES_PER_ROUTE} rollouts/epoch")
    print(
        f"  chunk={CHUNK_K}  init55={'on' if INIT_FROM_EXP055 else 'off'}"
        f"  residual_scale={RESIDUAL_SCALE:g}"
        f"  ref_input={'on' if ACTOR_REF_INPUT else 'off'}"
        f"  crit(top={CRIT_TOP_FRAC:.2f}, warm={CRIT_ONLY_EPOCHS}, w={CRIT_WEIGHT:.2f}, look={CRIT_LOOKAHEAD})"
    )
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
        f"  rew_scale={REWARD_SCALE:g}"
        f"  base_rew_baseline={'on' if USE_BASE_REWARD_BASELINE else 'off'}"
        f"  pos_adv_only={'on' if POS_ADV_ONLY else 'off'}>{POS_ADV_MARGIN:g}"
        f"  resid_bc={RESIDUAL_BC_COEF:g}"
        f"  lr_decay={'on' if LR_DECAY else 'off'}"
        f"  resume_opt={'on' if RESUME_OPT else 'off'}"
        f"  reset_critic={'on' if RESET_CRITIC else 'off'}"
        f"  resume_warmup={'on' if RESUME_WARMUP else 'off'}"
        f"  σfloor_eff={SIGMA_FLOOR * RESIDUAL_SCALE:g} coef={SIGMA_FLOOR_COEF}"
        f"  rew_rms_norm={'on' if REWARD_RMS_NORM else 'off'}"
        f"  adv_norm={'on' if ADV_NORM else 'off'}"
        f"  compile={'on' if COMPILE else 'off'}"
        f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}  K={K_EPOCHS}  dim={STATE_DIM}\n"
    )

    for epoch in range(MAX_EP):
        if DELTA_SCALE_DECAY:
            ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(np.pi * epoch / MAX_EP))
        else:
            ds = ds_max_run
        cur_ds = ds
        if RESUME and RESUME_OPT and epoch == 0:
            pi_lr = ppo.pi_opt.param_groups[0]["lr"]
            vf_lr = ppo.vf_opt.param_groups[0]["lr"]
        elif LR_DECAY:
            pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
            vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
        else:
            pi_lr = PI_LR
            vf_lr = VF_LR
        for pg in ppo.pi_opt.param_groups:
            pg["lr"] = pi_lr
        for pg in ppo.vf_opt.param_groups:
            pg["lr"] = vf_lr

        t0 = time.time()
        n_routes = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
        route_batch = random.sample(tr_f, max(n_routes, 1))
        batch = [f for f in route_batch for _ in range(SAMPLES_PER_ROUTE)]
        res = rollout(batch, ac, mdl_path, ort_sess, csv_cache, deterministic=False, ds=ds)
        if USE_BASE_REWARD_BASELINE:
            base_res = rollout(route_batch, ac, mdl_path, ort_sess, csv_cache, deterministic=True, ds=ds, base_only=True, return_rew=True)
            res["rew"] = res["rew"] - base_res["rew"].repeat_interleave(SAMPLES_PER_ROUTE, dim=0)
            res["base_relative"] = True
        t1 = time.time()

        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co, ds=ds, epoch=epoch)
        tu = time.time() - t1

        phase = "  [critic warmup]" if co else ""
        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}  σraw={info['σraw']:.4f}"
            f"  σpen={info['σpen']:.4f}  bc={info['bc']:.4f}  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  C={info['crit']:.2f}  keep={info['keep']:.2f}"
            f"  Δs={ds:.4f}  lr={info['lr']:.1e}  ⏱{t1 - t0:.0f}+{tu:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    torch.save(
        {"ac": ac.state_dict(), "delta_scale": cur_ds, "chunk_k": CHUNK_K, "residual_scale": RESIDUAL_SCALE},
        EXP_DIR / "final_model.pt",
    )


if __name__ == "__main__":
    train()
