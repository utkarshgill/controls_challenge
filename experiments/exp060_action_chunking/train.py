# exp060 — Action-Chunking PPO
#
# Key change from exp059: policy outputs CHUNK_K=5 deltas at once.
# Re-plans every 5 steps. This gives the policy a planning horizon,
# reduces per-step myopia, and produces temporally coherent actions.
#
# 400 control steps / 5 = 80 decision points per episode.
# Actor: obs(258) → (mu_1..mu_5, log_sigma_1..log_sigma_5) = 10 outputs
# Each delta_i = tanh(raw_i) * DELTA_SCALE, applied sequentially.
# Log-prob of chunk = sum of per-step log-probs.
# Value is estimated once per chunk.
# GAE operates on chunk-level (80 steps, not 400).
#
# Between re-plan points, history buffers are still updated for obs accuracy.

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
CHUNK_K = int(os.getenv("CHUNK_K", "5"))  # actions per chunk
STATE_DIM = 258  # obs dim (same as exp059)
HIDDEN = 256
A_LAYERS, C_LAYERS = 4, 4
DELTA_SCALE_MAX = float(os.getenv("DELTA_SCALE_MAX", "0.25"))
DELTA_SCALE_MIN = float(os.getenv("DELTA_SCALE_MIN", "0.25"))

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
LOG_SIGMA_MIN = float(os.getenv("LOG_SIGMA_MIN", "-4.0"))
LOG_SIGMA_MAX = float(os.getenv("LOG_SIGMA_MAX", "0.0"))
ACT_SMOOTH = float(os.getenv("ACT_SMOOTH", "0.5"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1.0"))
MINI_BS = int(os.getenv("MINI_BS", "5_000"))  # fewer chunks than steps
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))
ELITE_BC_COEF = float(os.getenv("ELITE_BC_COEF", "0.01"))
ELITE_BC_FRAC = float(os.getenv("ELITE_BC_FRAC", "0.2"))

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "20"))
BC_LR = float(os.getenv("BC_LR", "0.01"))
BC_BS = int(os.getenv("BC_BS", "2048"))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "20"))
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


def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout offsets ────────────────────────────────────────
C = 18
H1 = C + HIST_LEN  # 38
H2 = H1 + HIST_LEN  # 58
F_LAT = H2  # 58
F_ROLL = F_LAT + FUTURE_K  # 108
F_V = F_ROLL + FUTURE_K  # 158
F_A = F_V + FUTURE_K  # 208
OBS_DIM = F_A + FUTURE_K  # 258

MAX_CONTROL_STEPS = COST_END_IDX - CONTROL_START_IDX  # 400
N_CHUNKS = MAX_CONTROL_STEPS // CHUNK_K  # 80

assert MAX_CONTROL_STEPS % CHUNK_K == 0, (
    f"CHUNK_K={CHUNK_K} must divide {MAX_CONTROL_STEPS} evenly"
)


# ══════════════════════════════════════════════════════════════
#  Model — Chunked Actor
# ══════════════════════════════════════════════════════════════


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor outputs 2*CHUNK_K values: (mu_1..mu_K, log_sigma_1..log_sigma_K)
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
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
        _ortho(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]:
            _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

    def get_mu_sigma(self, obs):
        """Return (mu, sigma) each of shape (..., CHUNK_K)."""
        out = self.actor(obs)  # (..., 2*CHUNK_K)
        mu = out[..., :CHUNK_K]  # (..., K)
        log_sigma = out[..., CHUNK_K:].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        sigma = log_sigma.exp()
        return mu, sigma


# ══════════════════════════════════════════════════════════════
#  Squashed Gaussian helpers (K-dimensional)
# ══════════════════════════════════════════════════════════════


def _squashed_gaussian_sample_k(mu, sigma):
    """Sample K-dim tanh-squashed Gaussian. Returns (raw, squashed, logp_sum).
    raw, squashed: (..., K).  logp_sum: (...) — sum over K."""
    eps = torch.randn_like(mu)
    raw = mu + sigma * eps
    squashed = torch.tanh(raw)
    logp_per = (
        -0.5 * eps.pow(2)
        - 0.5 * np.log(2 * np.pi)
        - sigma.log()
        - torch.log(1.0 - squashed.pow(2) + 1e-6)
    )
    return raw, squashed, logp_per.sum(dim=-1)  # sum over K dims


def _squashed_gaussian_logprob_k(raw, mu, sigma):
    """Log-prob of K-dim pre-tanh sample. Returns (...) scalar."""
    squashed = torch.tanh(raw)
    logp_per = (
        -0.5 * ((raw - mu) / sigma).pow(2)
        - 0.5 * np.log(2 * np.pi)
        - sigma.log()
        - torch.log(1.0 - squashed.pow(2) + 1e-6)
    )
    return logp_per.sum(dim=-1)


def _squashed_gaussian_entropy_k(sigma):
    """Approximate entropy of K-dim tanh-squashed Gaussian. Returns (...) scalar."""
    gauss_ent = 0.5 * torch.log(2 * np.pi * np.e * sigma.pow(2))
    correction = -2.0 * sigma.pow(2) / (1.0 + sigma.pow(2))
    return (gauss_ent + correction).sum(dim=-1)


# ══════════════════════════════════════════════════════════════
#  Observation builder (GPU, batched) — same as exp059
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

    ctrl_step = max(0, step_idx - CONTROL_START_IDX)
    progress = ctrl_step / MAX_CONTROL_STEPS
    buf_valid = min(ctrl_step, HIST_LEN) / HIST_LEN

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
    buf[:, 16] = progress
    buf[:, 17] = buf_valid

    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)

    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO

    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  GPU Rollout with Action Chunking
# ══════════════════════════════════════════════════════════════


def rollout(
    csv_files,
    ac,
    mdl_path,
    ort_session,
    csv_cache,
    deterministic=False,
    ds=DELTA_SCALE_MAX,
):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N, T = sim.N, sim.T
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    # Storage: one entry per chunk (not per step)
    if not deterministic:
        all_obs = torch.empty(
            (N_CHUNKS, N, OBS_DIM), dtype=torch.float32, device="cuda"
        )
        all_raw = torch.empty(
            (N_CHUNKS, N, CHUNK_K), dtype=torch.float32, device="cuda"
        )
        all_logp = torch.empty((N_CHUNKS, N), dtype=torch.float32, device="cuda")
        all_val = torch.empty((N_CHUNKS, N), dtype=torch.float32, device="cuda")

    chunk_idx = 0
    hist_head = HIST_LEN - 1

    # Pre-planned actions for current chunk
    planned_squashed = None  # (N, CHUNK_K) — tanh-squashed deltas
    planned_raw = None  # (N, CHUNK_K) — pre-tanh raw values
    planned_logp = None  # (N,) — sum of log-probs
    planned_val = None  # (N,) — value at plan time
    planned_obs = None  # (N, OBS_DIM) — obs at plan time
    step_in_chunk = 0  # 0..CHUNK_K-1

    def ctrl(step_idx, sim_ref):
        nonlocal chunk_idx, hist_head, err_sum
        nonlocal planned_squashed, planned_raw, planned_logp, planned_val, planned_obs
        nonlocal step_in_chunk

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

        # ── Plan at start of each chunk ──
        ctrl_step = step_idx - CONTROL_START_IDX
        in_cost_window = step_idx < COST_END_IDX

        if ctrl_step % CHUNK_K == 0 and in_cost_window:
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
                mu, sigma = ac.get_mu_sigma(obs_buf)  # (N, K), (N, K)
                val = ac.critic(obs_buf).squeeze(-1)  # (N,)

            if deterministic:
                planned_squashed = torch.tanh(mu)
                planned_raw = None
                planned_logp = None
                planned_val = None
                planned_obs = None
            else:
                raw, squashed, logp_sum = _squashed_gaussian_sample_k(mu, sigma)
                planned_squashed = squashed
                planned_raw = raw
                planned_logp = logp_sum
                planned_val = val
                planned_obs = obs_buf.clone()

                # Store chunk data
                all_obs[chunk_idx] = planned_obs
                all_raw[chunk_idx] = planned_raw
                all_logp[chunk_idx] = planned_logp
                all_val[chunk_idx] = planned_val
                chunk_idx += 1

            step_in_chunk = 0

        # ── Apply the pre-planned delta for this step ──
        if planned_squashed is not None and step_in_chunk < CHUNK_K:
            delta = planned_squashed[:, step_in_chunk].to(h_act.dtype) * ds
            action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        else:
            # Past cost window — hold last action
            action = h_act[:, hist_head]

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        step_in_chunk += 1

        return action

    costs = sim.rollout(ctrl)["total_cost"]

    if deterministic:
        return costs.tolist()

    # ── Build chunk-level rewards ──
    # Sum per-step rewards within each chunk → (N, N_CHUNKS)
    start = CONTROL_START_IDX
    end = COST_END_IDX
    pred = sim.current_lataccel_history[:, start:end].float()  # (N, 400)
    target_la = dg["target_lataccel"][:, start:end].float()
    act = sim.action_history[:, start:end].float()

    lat_r = (target_la - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    step_rew = -(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(
        REWARD_SCALE, 1e-8
    )

    # Reshape (N, 400) → (N, N_CHUNKS, CHUNK_K) → sum over K → (N, N_CHUNKS)
    chunk_rew = step_rew.reshape(N, N_CHUNKS, CHUNK_K).sum(dim=2)

    dones = torch.zeros((N, N_CHUNKS), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    CI = chunk_idx  # should equal N_CHUNKS
    return dict(
        obs=all_obs[:CI].permute(1, 0, 2).reshape(-1, OBS_DIM),  # (N*CI, OBS_DIM)
        raw=all_raw[:CI].permute(1, 0, 2).reshape(-1, CHUNK_K),  # (N*CI, K)
        old_logp=all_logp[:CI].T.reshape(-1),  # (N*CI,)
        val_2d=all_val[:CI].T,  # (N, CI)
        rew=chunk_rew,  # (N, N_CHUNKS)
        done=dones,
        costs=costs,
    )


# ══════════════════════════════════════════════════════════════
#  BC Pretrain — extract K-step chunks from CSV steer data
# ══════════════════════════════════════════════════════════════


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), "edge")
    return np.full(k, fallback, dtype=np.float32)


def _build_obs_bc(target, current, state, fplan, hist_act, hist_lat, ctrl_step):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    _flat = getattr(fplan, "lataccel", None)
    fp0 = _flat[0] if (_flat and len(_flat) > 0) else target
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0
    progress = ctrl_step / MAX_CONTROL_STEPS
    buf_valid = min(ctrl_step, HIST_LEN) / HIST_LEN

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
            progress,
            buf_valid,
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
    """Extract (obs, raw_chunk) pairs from CSV warmup data.
    We extract overlapping K-step chunks from the warmup window."""
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values
    steer = -df["steerCommand"].values

    obs_list, raw_list = [], []
    h_act = [0.0] * HIST_LEN
    h_lat = [0.0] * HIST_LEN

    # Extract from warmup window with sliding K-step chunks
    warmup_start = CONTEXT_LENGTH
    warmup_end = CONTROL_START_IDX
    for step_idx in range(warmup_start, warmup_end - CHUNK_K + 1):
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

        ctrl_step = max(0, step_idx - CONTROL_START_IDX)
        obs = _build_obs_bc(target_la, target_la, state, fplan, h_act, h_lat, ctrl_step)

        # Target: K consecutive deltas from CSV steer data
        raw_chunk = np.zeros(CHUNK_K, dtype=np.float32)
        prev = h_act[-1]
        for k in range(CHUNK_K):
            si = step_idx + k
            if si < len(steer):
                delta_target = np.clip(
                    (steer[si] - prev) / DELTA_SCALE_MAX, -0.999, 0.999
                )
                raw_chunk[k] = np.clip(np.arctanh(delta_target), -3.0, 3.0)
                prev = steer[si]

        obs_list.append(obs)
        raw_list.append(raw_chunk)

        # Advance history by 1 step (sliding window)
        h_act = h_act[1:] + [steer[step_idx]]
        h_lat = h_lat[1:] + [tgt[step_idx]]

    return (np.array(obs_list, np.float32), np.array(raw_list, np.float32))


def pretrain_bc(ac, all_csvs):
    print(f"BC pretrain: extracting K={CHUNK_K} chunks from {len(all_csvs)} CSVs ...")
    results = process_map(
        _bc_worker,
        [str(f) for f in all_csvs],
        max_workers=10,
        chunksize=50,
        disable=False,
    )
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    N = len(all_obs)
    print(f"BC pretrain: {N} samples (each with {CHUNK_K} targets), {BC_EPOCHS} epochs")

    obs_t = torch.FloatTensor(all_obs).to(DEV)
    raw_t = torch.FloatTensor(all_raw).to(DEV)  # (N, K)
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            mu, sigma = ac.get_mu_sigma(obs_t[idx])  # (B, K), (B, K)
            # Gaussian NLL on pre-tanh space, summed over K
            loss = (
                (0.5 * ((raw_t[idx] - mu) / sigma).pow(2) + sigma.log())
                .sum(dim=-1)
                .mean()
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        print(
            f"  BC epoch {ep}: loss={total / nb:.6f}  lr={opt.param_groups[0]['lr']:.1e}"
        )
    print("BC pretrain done.\n")


# ══════════════════════════════════════════════════════════════
#  PPO — chunk-level
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

    def _gae(self, rew, val, done):
        """GAE on chunk-level. rew/val/done: (N, N_CHUNKS)."""
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

    @staticmethod
    def _elite_bc_weights(gd):
        if ELITE_BC_COEF <= 0.0 or SAMPLES_PER_ROUTE <= 1:
            return None, 0.0
        n_total = gd["rew"].shape[0]
        S = gd["rew"].shape[1]
        if n_total % SAMPLES_PER_ROUTE != 0:
            return None, 0.0
        n_routes = n_total // SAMPLES_PER_ROUTE
        frac = float(np.clip(ELITE_BC_FRAC, 0.0, 1.0))
        if frac <= 0.0:
            return None, 0.0
        k_elite = max(1, min(SAMPLES_PER_ROUTE, int(np.ceil(SAMPLES_PER_ROUTE * frac))))
        costs = torch.as_tensor(gd["costs"], device="cuda", dtype=torch.float32)
        costs = costs.view(n_routes, SAMPLES_PER_ROUTE)
        elite_idx = torch.topk(-costs, k=k_elite, dim=1).indices
        weights = torch.zeros(
            (n_routes, SAMPLES_PER_ROUTE, S), dtype=torch.float32, device="cuda"
        )
        weights.scatter_(1, elite_idx.unsqueeze(-1).expand(-1, -1, S), 1.0)
        return weights.view(-1), (k_elite / SAMPLES_PER_ROUTE)

    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX):
        obs = gd["obs"]  # (N*CI, OBS_DIM)
        raw = gd["raw"]  # (N*CI, CHUNK_K)
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])

        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_2d = adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)

        elite_bc_w, elite_frac = self._elite_bc_weights(gd)

        n_vf = 0
        n_actor = 0
        n_elite = 0
        vf_sum = 0.0
        pi_sum = 0.0
        ent_sum = 0.0
        sigma_sum = 0.0
        elite_bc_sum = 0.0

        with torch.no_grad():
            if "old_logp" in gd:
                old_lp = gd["old_logp"]
            else:
                mu_old, sigma_old = self.ac.get_mu_sigma(obs)
                old_lp = _squashed_gaussian_logprob_k(raw, mu_old, sigma_old)

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

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                else:
                    mu, sigma = self.ac.get_mu_sigma(obs[idx])  # (B, K)
                    lp = _squashed_gaussian_logprob_k(raw[idx], mu, sigma)
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv, ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv
                    ).mean()
                    ent = _squashed_gaussian_entropy_k(sigma).mean()

                    elite_bc_loss = torch.tensor(0.0, device="cuda")
                    if elite_bc_w is not None:
                        mb_elite = elite_bc_w[idx]
                        elite_mass = mb_elite.sum()
                        if elite_mass.item() > 0:
                            elite_bc_loss = -(lp * mb_elite).sum() / elite_mass
                            elite_bc_sum += (
                                elite_bc_loss.detach().item() * elite_mass.item()
                            )
                            n_elite += int(elite_mass.item())

                    loss = (
                        pi_loss
                        + VF_COEF * vf_loss
                        - ENT_COEF * ent
                        + ELITE_BC_COEF * elite_bc_loss
                    )
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.pi_opt.step()
                    self.vf_opt.step()
                    pi_sum += pi_loss.detach().item() * bs
                    ent_sum += ent.detach().item() * bs
                    sigma_sum += sigma.mean().detach().item() * bs
                    n_actor += bs

        with torch.no_grad():
            mu_d, sigma_d = self.ac.get_mu_sigma(obs[:1000])
            sigma_raw = sigma_d.mean().item()
            sigma_eff = sigma_raw * ds
        return dict(
            pi=(pi_sum / max(1, n_actor)) if not critic_only else 0.0,
            vf=(vf_sum / max(1, n_vf)),
            ent=(ent_sum / max(1, n_actor)) if not critic_only else 0.0,
            σ=sigma_eff,
            σraw=sigma_raw,
            ebc=(elite_bc_sum / max(1, n_elite)) if not critic_only else 0.0,
            efrac=elite_frac if not critic_only else 0.0,
            lr=self.pi_opt.param_groups[0]["lr"],
        )


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(
        files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds
    )
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

    warmup_off = 0
    resumed_ds = None
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt["ac"])
        if RESUME_OPT and "pi_opt" in ckpt:
            ppo.pi_opt.load_state_dict(ckpt["pi_opt"])
            ppo.vf_opt.load_state_dict(ckpt["vf_opt"])
            if "ret_rms" in ckpt:
                r = ckpt["ret_rms"]
                ppo._rms.mean, ppo._rms.var, ppo._rms.count = (
                    r["mean"],
                    r["var"],
                    r["count"],
                )
        elif RESUME_OPT:
            print("RESUME_OPT=1 but optimizer state missing; using fresh")
        warmup_off = 0 if RESUME_WARMUP else CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
        if RESUME_DS:
            ds_ckpt = ckpt.get("delta_scale", None)
            if ds_ckpt is not None:
                resumed_ds = float(ds_ckpt)
                print(f"Resumed delta_scale={resumed_ds:.6f}")
        if RESET_CRITIC:
            for layer in ac.critic[:-1]:
                if isinstance(layer, nn.Linear):
                    _ortho(layer)
            if isinstance(ac.critic[-1], nn.Linear):
                _ortho(ac.critic[-1], gain=1.0)
            ppo.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic reset")
    else:
        pretrain_bc(ac, all_csv)

    if COMPILE:
        ac.actor = torch.compile(
            ac.actor, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        ac.critic = torch.compile(
            ac.critic, mode="max-autotune-no-cudagraphs", dynamic=True
        )

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
                "ret_rms": {
                    "mean": ppo._rms.mean,
                    "var": ppo._rms.var,
                    "count": ppo._rms.count,
                },
                "delta_scale": cur_ds,
            },
            BEST_PT,
        )

    print(
        f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  chunk_k={CHUNK_K}  n_chunks={N_CHUNKS}  dev={DEV}"
    )
    _n_r = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
    print(
        f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {_n_r} routes × {SAMPLES_PER_ROUTE} = {_n_r * SAMPLES_PER_ROUTE} rollouts/epoch"
    )
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
        f"  rew_scale={REWARD_SCALE:g}"
        f"  lr_decay={'on' if LR_DECAY else 'off'}"
        f"  log_σ=[{LOG_SIGMA_MIN},{LOG_SIGMA_MAX}]"
        f"  elite_bc={ELITE_BC_COEF} frac={ELITE_BC_FRAC}"
        f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}"
        f"  dim={STATE_DIM}  chunk={CHUNK_K}\n"
    )

    for epoch in range(MAX_EP):
        if DELTA_SCALE_DECAY:
            ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (
                1 + np.cos(np.pi * epoch / MAX_EP)
            )
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
        batch = random.sample(tr_f, max(n_routes, 1))
        batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]
        res = rollout(
            batch, ac, mdl_path, ort_sess, csv_cache, deterministic=False, ds=ds
        )

        t1 = time.time()
        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co, ds=ds)
        tu = time.time() - t1

        phase = "  [critic warmup]" if co else ""
        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}  σraw={info['σraw']:.4f}"
            f"  π={info['pi']:+.4f}  ebc={info['ebc']:.4f}"
            f"  vf={info['vf']:.1f}  H={info['ent']:.2f}"
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
    torch.save({"ac": ac.state_dict()}, EXP_DIR / "final_model.pt")


if __name__ == "__main__":
    train()
