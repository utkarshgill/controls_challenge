# exp104 — Custom physics-informed controller architecture
#
# Three specialized perception modules augment the exp055 MLP backbone:
#
#   1. STATE FILTER: learned weighted average over 20-step error/action/jerk history.
#      Outputs: filtered_error, error_trend, noise_estimate, action_effect (~8 features)
#      Physical role: Kalman-like filter that estimates true plant state from noisy obs.
#
#   2. PLANT MODEL: small MLP on core features + filtered signals.
#      Outputs: ~8 features encoding expected plant response.
#      Physical role: captures how the plant will respond given the estimated state.
#
#   3. PLANNER: state-conditioned weighted average over 50-step future plan.
#      Outputs: ~8 anticipation features.
#      Physical role: "given where I am, what upcoming road features matter?"
#
# Total: 256 (original MLP obs) + 24 (3 modules) = 280 input dims.
# Warm-started from exp055. Modules zero-initialized so starting perf = exp055.

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
MLP_OBS_DIM = 256  # original exp055 obs dim
FILTER_OUT = 8  # state filter output features
PLANT_OUT = 8  # plant model output features
PLAN_OUT = 8  # planner output features
EXTRA_DIM = FILTER_OUT + PLANT_OUT + PLAN_OUT  # 24
STATE_DIM = MLP_OBS_DIM + EXTRA_DIM  # 280
HIDDEN = 256
A_LAYERS, C_LAYERS = 4, 4
HIST_TOKEN_DIM = 5  # per-step history: error, action, jerk, target, current (scaled)
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
K_EPOCHS = int(os.getenv("K_EPOCHS", "4"))
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


def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout ────────────────────────────────────────────────
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


class StateFilter(nn.Module):
    """Learned weighted average over history tokens.

    At each position, a small MLP scores relevance. Softmax over positions
    gives attention weights. Weighted sum produces filtered features.
    Also computes trend (weighted by position) and variance estimate.

    This is a single-head linear attention — O(T) not O(T²).
    """

    def __init__(self, token_dim, out_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(token_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.value = nn.Linear(token_dim, out_dim)
        # Trend: position-weighted features
        self.trend_value = nn.Linear(token_dim, out_dim // 2)
        self.out_dim = out_dim

    def forward(self, tokens):
        """tokens: (B, T, token_dim) → (B, out_dim)"""
        B, T, _ = tokens.shape
        # Attention weights from scoring each position
        scores = self.score(tokens).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)

        # Weighted average: "filtered" signal
        values = self.value(tokens)  # (B, T, out_dim)
        filtered = (weights.unsqueeze(-1) * values).sum(dim=1)  # (B, out_dim)

        # Position-weighted trend: recent positions weighted more for trend
        pos_weight = torch.linspace(-1, 1, T, device=tokens.device)  # [-1, 1]
        trend_vals = self.trend_value(tokens)  # (B, T, out_dim//2)
        trend = (
            pos_weight.unsqueeze(0).unsqueeze(-1) * weights.unsqueeze(-1) * trend_vals
        ).sum(dim=1)

        return torch.cat([filtered[:, : self.out_dim // 2], trend], dim=-1)


class PlantModel(nn.Module):
    """Small MLP that combines core state features with filtered signals
    to produce plant response features."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Planner(nn.Module):
    """State-conditioned weighted average over future plan.

    The "query" is the current plant state (from PlantModel).
    The attention weights depend on both the query and each future step.
    This lets the planner focus on the relevant part of the future
    depending on the current state.
    """

    def __init__(self, fut_token_dim, query_dim, out_dim):
        super().__init__()
        self.fut_proj = nn.Linear(fut_token_dim, 16)
        self.query_proj = nn.Linear(query_dim, 16)
        self.score = nn.Linear(16, 1)
        self.value = nn.Linear(fut_token_dim, out_dim)

    def forward(self, fut_tokens, query):
        """fut_tokens: (B, T, fut_dim), query: (B, query_dim) → (B, out_dim)"""
        B, T, _ = fut_tokens.shape
        # Score each future position conditioned on query
        f = self.fut_proj(fut_tokens)  # (B, T, 16)
        q = self.query_proj(query).unsqueeze(1).expand(-1, T, -1)  # (B, T, 16)
        scores = self.score(torch.tanh(f + q)).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)

        values = self.value(fut_tokens)  # (B, T, out_dim)
        return (weights.unsqueeze(-1) * values).sum(dim=1)  # (B, out_dim)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Three perception modules
        self.state_filter = StateFilter(HIST_TOKEN_DIM, FILTER_OUT)
        self.plant_model = PlantModel(
            C + FILTER_OUT, PLANT_OUT
        )  # core features + filter
        self.planner = Planner(4, PLANT_OUT, PLAN_OUT)  # future tokens, plant query

        # Same MLP as exp055, but wider input (280 vs 256)
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
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

    def _augment(self, obs, hist_tokens, fut_tokens):
        """obs: (B, 256), hist_tokens: (B, 20, 5), fut_tokens: (B, 50, 4) → (B, 280)"""
        # Step 1: Filter history
        filt = self.state_filter(hist_tokens)  # (B, 8)
        # Step 2: Plant model — core features from obs[:, :C] + filter output
        plant_in = torch.cat([obs[:, :C], filt], dim=-1)  # (B, C+8=24)
        plant = self.plant_model(plant_in)  # (B, 8)
        # Step 3: Planner — future tokens conditioned on plant state
        plan = self.planner(fut_tokens, plant)  # (B, 8)
        return torch.cat([obs, filt, plant, plan], dim=-1)  # (B, 280)

    def beta_params(self, obs, hist_tokens, fut_tokens):
        aug = self._augment(obs, hist_tokens, fut_tokens)
        out = self.actor(aug)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0


# ══════════════════════════════════════════════════════════════
#  Observation builder
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
#  Rollout
# ══════════════════════════════════════════════════════════════


def rollout(
    csv_files,
    ac,
    mdl_path,
    ort_session,
    csv_cache,
    deterministic=False,
    ds=DELTA_SCALE_MAX,
    sim_temp=None,
):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    if sim_temp is None:
        sim_temp = float(os.getenv("SIM_TEMP", "0.8"))
    if sim_temp != 0.8:
        sim.sim_model.sim_temperature = sim_temp
    N = sim.N
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    # History token ring buffer for state filter
    hist_tok_buf = torch.zeros(
        (N, HIST_LEN, HIST_TOKEN_DIM), dtype=torch.float32, device="cuda"
    )
    hist_tok_head = HIST_LEN - 1
    prev_lat = torch.zeros(N, dtype=torch.float32, device="cuda")

    if not deterministic:
        all_obs = torch.empty(
            (max_steps, N, OBS_DIM), dtype=torch.float32, device="cuda"
        )
        all_hist = torch.empty(
            (max_steps, N, HIST_LEN, HIST_TOKEN_DIM), dtype=torch.float32, device="cuda"
        )
        all_fut = torch.empty(
            (max_steps, N, FUTURE_K, 4), dtype=torch.float32, device="cuda"
        )
        all_raw = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_val = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")

    si = 0
    hist_head = HIST_LEN - 1

    EXPECTED_REWARD = int(os.getenv("EXPECTED_REWARD", "0"))
    if EXPECTED_REWARD:
        sim.compute_expected = True
        expected_history = torch.zeros(
            (N, max_steps), dtype=torch.float32, device="cuda"
        )

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum, hist_tok_head, prev_lat
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        tgt32 = target.float()

        # Collect expected lataccel for expected-reward training
        if EXPECTED_REWARD and si > 0:
            exp = getattr(sim_ref, "expected_lataccel", None)
            if exp is not None:
                expected_history[:, si - 1] = exp.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_err = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_err
        ei = err_sum * (DEL_T / HIST_LEN)

        # Update history token ring buffer
        next_th = (hist_tok_head + 1) % HIST_LEN
        jerk = (cur32 - prev_lat) / DEL_T
        hist_tok_buf[:, next_th, 0] = error / S_LAT
        hist_tok_buf[:, next_th, 1] = h_act32[:, hist_head] / S_STEER
        hist_tok_buf[:, next_th, 2] = jerk / S_LAT
        hist_tok_buf[:, next_th, 3] = tgt32 / S_LAT
        hist_tok_buf[:, next_th, 4] = cur32 / S_LAT
        hist_tok_head = next_th
        prev_lat = cur32

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        fill_obs(
            obs_buf,
            tgt32,
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

        # Unroll history token ring buffer to sequential order
        split = hist_tok_head + 1
        if split >= HIST_LEN:
            hist_ordered = hist_tok_buf
        else:
            hist_ordered = torch.cat(
                [hist_tok_buf[:, split:], hist_tok_buf[:, :split]], dim=1
            )

        # Build future tokens (N, FUTURE_K, 4)
        fut_tokens = torch.stack(
            [
                future["target_lataccel"][:, step_idx] / S_LAT,
                future["roll_lataccel"][:, step_idx] / S_ROLL,
                future["v_ego"][:, step_idx] / S_VEGO,
                future["a_ego"][:, step_idx] / S_AEGO,
            ],
            dim=-1,
        ).clamp(-5.0, 5.0)

        with torch.no_grad():
            aug = ac._augment(obs_buf, hist_ordered, fut_tokens)
            logits = ac.actor(aug)
            val = ac.critic(aug).squeeze(-1)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0

        if deterministic:
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
            logp = None
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            x = dist.sample()
            raw = 2.0 * x - 1.0
            logp = dist.log_prob(x)

        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_hist[si] = hist_ordered if split >= HIST_LEN else hist_ordered.clone()
            all_fut[si] = fut_tokens
            all_raw[si] = raw
            all_logp[si] = logp
            all_val[si] = val
            si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    if deterministic:
        return costs.tolist()

    S = si
    start = CONTROL_START_IDX
    end = start + S
    # Use expected lataccel for reward if enabled, sampled otherwise
    if EXPECTED_REWARD:
        # Collect the last expected value (from the final sim_step)
        exp = getattr(sim, "expected_lataccel", None)
        if exp is not None:
            expected_history[:, S - 1] = exp.float()
        pred = expected_history[:, :S]
    else:
        pred = sim.current_lataccel_history[:, start:end].float()
    target = dg["target_lataccel"][:, start:end].float()
    act = sim.action_history[:, start:end].float()
    lat_r = (target - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    rew = (
        -(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(REWARD_SCALE, 1e-8)
    ).float()
    dones = torch.zeros((N, S), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    return dict(
        obs=all_obs[:S].permute(1, 0, 2).reshape(-1, OBS_DIM),
        hist=all_hist[:S].permute(1, 0, 2, 3).reshape(-1, HIST_LEN, HIST_TOKEN_DIM),
        fut=all_fut[:S].permute(1, 0, 2, 3).reshape(-1, FUTURE_K, 4),
        raw=all_raw[:S].T.reshape(-1),
        old_logp=all_logp[:S].T.reshape(-1),
        val_2d=all_val[:S].T,
        rew=rew,
        done=dones,
        costs=costs,
    )


# ══════════════════════════════════════════════════════════════
#  BC Pretrain
# ══════════════════════════════════════════════════════════════


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        return np.pad(np.array(vals, np.float32), (0, k - len(vals)), "edge")
    return np.full(k, fallback, dtype=np.float32)


def _build_obs_bc(target, current, state, fplan, hist_act, hist_lat):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    fp0 = (getattr(fplan, "lataccel", None) or [target])[0]
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
        np.float32,
    )
    return np.clip(
        np.concatenate(
            [
                core,
                np.array(hist_act, np.float32) / S_STEER,
                np.array(hist_lat, np.float32) / S_LAT,
                _future_raw(fplan, "lataccel", target) / S_LAT,
                _future_raw(fplan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
                _future_raw(fplan, "v_ego", state.v_ego) / S_VEGO,
                _future_raw(fplan, "a_ego", state.a_ego) / S_AEGO,
            ]
        ),
        -5.0,
        5.0,
    )


def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values
    steer = -df["steerCommand"].values
    obs_list, raw_list, h_act, h_lat = [], [], [0.0] * HIST_LEN, [0.0] * HIST_LEN
    for si in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state = State(roll_lataccel=roll_la[si], v_ego=v_ego[si], a_ego=a_ego[si])
        fplan = FuturePlan(
            lataccel=tgt[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
        )
        obs_list.append(_build_obs_bc(tgt[si], tgt[si], state, fplan, h_act, h_lat))
        raw_list.append(np.clip((steer[si] - h_act[-1]) / DELTA_SCALE_MAX, -1.0, 1.0))
        h_act = h_act[1:] + [steer[si]]
        h_lat = h_lat[1:] + [tgt[si]]
    return (np.array(obs_list, np.float32), np.array(raw_list, np.float32))


def pretrain_bc(ac, all_csvs):
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(
        _bc_worker, [str(f) for f in all_csvs], max_workers=10, chunksize=50
    )
    obs_t = torch.FloatTensor(np.concatenate([r[0] for r in results])).to(DEV)
    raw_t = torch.FloatTensor(np.concatenate([r[1] for r in results])).to(DEV)
    N = len(obs_t)
    print(f"BC pretrain: {N} samples, {BC_EPOCHS} epochs")
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)
    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            a_p, b_p = ac.beta_params(obs_t[idx])
            x = ((raw_t[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
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
        self.pi_opt = optim.Adam(
            list(ac.state_filter.parameters())
            + list(ac.plant_model.parameters())
            + list(ac.planner.parameters())
            + list(ac.actor.parameters()),
            lr=PI_LR,
            eps=1e-5,
        )
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._rms = RunningMeanStd()

    @staticmethod
    def _beta_sigma_raw(a, b):
        return 2.0 * torch.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

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

    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX):
        obs = gd["obs"]
        hist = gd["hist"]
        fut = gd["fut"]
        raw = gd["raw"].unsqueeze(-1)
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])
        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_t = (
                adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
                - adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1).mean(
                    dim=1, keepdim=True
                )
            ).reshape(-1)

        # CVaR: upweight the worst routes
        CVAR_ALPHA = float(
            os.getenv("CVAR_ALPHA", "0")
        )  # 0=disabled, 0.2=focus on worst 20%
        if CVAR_ALPHA > 0:
            costs_t = torch.from_numpy(gd["costs"]).to("cuda").float()
            n_total = len(costs_t)
            if SAMPLES_PER_ROUTE > 1:
                n_routes = n_total // SAMPLES_PER_ROUTE
                route_costs = costs_t.view(n_routes, SAMPLES_PER_ROUTE).mean(dim=1)
            else:
                route_costs = costs_t
            # Routes in the worst alpha fraction get weight 1/alpha, others get 0
            threshold = torch.quantile(route_costs, 1.0 - CVAR_ALPHA)
            cvar_mask = (route_costs >= threshold).float()  # 1 for worst routes
            # Expand to per-sample weights
            if SAMPLES_PER_ROUTE > 1:
                sample_weights = cvar_mask.unsqueeze(1).expand(-1, SAMPLES_PER_ROUTE)
                sample_weights = (
                    sample_weights.unsqueeze(2).expand(-1, -1, S).reshape(-1)
                )
            else:
                sample_weights = cvar_mask.unsqueeze(1).expand(-1, S).reshape(-1)
            # Normalize so mean weight = 1
            sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-8)
            adv_t = adv_t * sample_weights

        # Hard-clip advantages to suppress outliers
        ADV_CLIP = float(os.getenv("ADV_CLIP", "0"))
        if ADV_CLIP > 0:
            adv_t = adv_t.clamp(-ADV_CLIP, ADV_CLIP)

        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)

        # Diagnostics: advantage signal quality
        self._diag_adv_std = adv_t.std().item()
        self._diag_adv_absmax = adv_t.abs().max().item()
        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            costs_2d = (
                torch.from_numpy(gd["costs"])
                .to("cuda")
                .float()
                .view(n_routes, SAMPLES_PER_ROUTE)
            )
            self._diag_cost_spread = (
                (costs_2d.max(dim=1).values - costs_2d.min(dim=1).values).mean().item()
            )
        else:
            self._diag_cost_spread = 0.0

        n_vf = 0
        n_actor = 0
        vf_sum = 0.0
        pi_sum = 0.0
        ent_sum = 0.0
        sigma_pen_sum = 0.0

        with torch.no_grad():
            old_lp = gd.get("old_logp")
            if old_lp is None:
                a_old, b_old = self.ac.beta_params(obs, hist, fut)
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(
                    x_t.squeeze(-1)
                )

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device="cuda").split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Compute augmented obs once
                aug = self.ac._augment(obs[idx], hist[idx], fut[idx])
                bs = idx.numel()

                if critic_only:
                    val = self.ac.critic(aug).squeeze(-1)
                    vf_loss = F.mse_loss(val, ret_t[idx])
                    vf_sum += vf_loss.item() * bs
                    n_vf += bs
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                else:
                    logits = self.ac.actor(aug)
                    val = self.ac.critic(aug).squeeze(-1)
                    vf_loss = F.mse_loss(val, ret_t[idx])
                    vf_sum += vf_loss.item() * bs
                    n_vf += bs

                    a_c = F.softplus(logits[..., 0]) + 1.0
                    b_c = F.softplus(logits[..., 1]) + 1.0
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv, ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv
                    ).mean()
                    ent = dist.entropy().mean()
                    sigma_pen = F.relu(
                        SIGMA_FLOOR - self._beta_sigma_raw(a_c, b_c).mean() * ds
                    )
                    loss = (
                        pi_loss
                        + VF_COEF * vf_loss
                        - ENT_COEF * ent
                        + SIGMA_FLOOR_COEF * sigma_pen
                    )
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.pi_opt.step()
                    self.vf_opt.step()
                    pi_sum += pi_loss.item() * bs
                    ent_sum += ent.item() * bs
                    sigma_pen_sum += sigma_pen.item() * bs
                    n_actor += bs

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs[:1000], hist[:1000], fut[:1000])
            σraw = self._beta_sigma_raw(a_d, b_d).mean().item()
        return dict(
            pi=pi_sum / max(1, n_actor) if not critic_only else 0.0,
            vf=vf_sum / max(1, n_vf),
            ent=ent_sum / max(1, n_actor) if not critic_only else 0.0,
            σ=σraw * ds,
            σraw=σraw,
            σpen=sigma_pen_sum / max(1, n_actor) if not critic_only else 0.0,
            lr=self.pi_opt.param_groups[0]["lr"],
            adv_std=self._diag_adv_std,
            adv_max=self._diag_adv_absmax,
            cost_spread=self._diag_cost_spread,
        )


# ══════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════


EVAL_TEMP = float(os.getenv("EVAL_TEMP", "0.8"))


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(
        files,
        ac,
        mdl_path,
        ort_session,
        csv_cache,
        deterministic=True,
        ds=ds,
        sim_temp=EVAL_TEMP,
    )
    return float(np.mean(costs)), float(np.std(costs))


# ══════════════════════════════════════════════════════════════
#  Training context + train_one_epoch
# ══════════════════════════════════════════════════════════════


class TrainingContext:
    def __init__(self):
        self.ac = ActorCritic().to(DEV)
        self.ppo = PPO(self.ac)
        self.mdl_path = ROOT / "models" / "tinyphysics.onnx"
        self.ort_sess = make_ort_session(self.mdl_path)
        self.all_csv = sorted((ROOT / "data").glob("*.csv"))
        self.tr_f = self.all_csv
        self.va_f = self.all_csv[:EVAL_N]
        self.csv_cache = CSVCache([str(f) for f in self.all_csv])
        self.best = float("inf")
        self.best_ep = "init"
        self.warmup_off = 0
        self.ds_max = DELTA_SCALE_MAX
        self.ds_min = DELTA_SCALE_MIN
        self.cur_ds = DELTA_SCALE_MAX

    def warm_start_from_exp055(self):
        """Load exp055 MLP weights into the wider MLP. Zero-init extra columns.
        Reset actor output + critic for fresh exploration."""
        exp055_pt = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
        if not exp055_pt.exists():
            print(f"  exp055 not found at {exp055_pt}")
            return False
        ckpt = torch.load(exp055_pt, weights_only=False, map_location=DEV)
        old_sd = ckpt["ac"]
        new_sd = self.ac.state_dict()
        loaded = 0
        for name, param in new_sd.items():
            # Skip perception modules
            if any(
                name.startswith(p)
                for p in ("state_filter.", "plant_model.", "planner.")
            ):
                continue
            if name in old_sd and old_sd[name].shape == param.shape:
                param.copy_(old_sd[name])
                loaded += 1
            elif name in old_sd:
                old_p = old_sd[name]
                if old_p.dim() == 2 and param.shape[1] > old_p.shape[1]:
                    # First layer wider: zero extra columns
                    param.zero_()
                    param[:, : old_p.shape[1]].copy_(old_p)
                    loaded += 1
                elif old_p.dim() == 1 and param.shape[0] == old_p.shape[0]:
                    param.copy_(old_p)
                    loaded += 1
        self.ac.load_state_dict(new_sd)
        # Reset actor output for exploration
        _ortho(self.ac.actor[-1], gain=0.01)
        # Reset critic fully
        for layer in self.ac.critic:
            _ortho(layer)
        _ortho(self.ac.critic[-1], gain=1.0)
        ds = float(ckpt.get("delta_scale", DELTA_SCALE_MAX))
        self.ds_max = ds
        self.ds_min = min(self.ds_min, ds)
        print(
            f"  Warm-started {loaded} tensors from exp055 (Δs={ds}), reset output+critic"
        )
        return True

    def save_best(self):
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "pi_opt": self.ppo.pi_opt.state_dict(),
                "vf_opt": self.ppo.vf_opt.state_dict(),
                "ret_rms": {
                    "mean": self.ppo._rms.mean,
                    "var": self.ppo._rms.var,
                    "count": self.ppo._rms.count,
                },
                "delta_scale": self.cur_ds,
            },
            BEST_PT,
        )

    def resume(self):
        if RESUME and BEST_PT.exists():
            ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
            self.ac.load_state_dict(ckpt["ac"])
            if RESUME_OPT and "pi_opt" in ckpt:
                self.ppo.pi_opt.load_state_dict(ckpt["pi_opt"])
                self.ppo.vf_opt.load_state_dict(ckpt["vf_opt"])
                if "ret_rms" in ckpt:
                    r = ckpt["ret_rms"]
                    self.ppo._rms.mean, self.ppo._rms.var, self.ppo._rms.count = (
                        r["mean"],
                        r["var"],
                        r["count"],
                    )
                print("RESUME_OPT=1: optimizer state, LR/eps, and RMS restored")
            else:
                print("RESUME_OPT=0: resumed weights only; fresh optimizer/RMS")
            self.warmup_off = 0 if RESUME_WARMUP else CRITIC_WARMUP
            if RESUME_DS:
                ds_ckpt = ckpt.get("delta_scale")
                if ds_ckpt is not None:
                    self.ds_max = float(ds_ckpt)
                    self.ds_min = min(self.ds_min, self.ds_max)
            if RESET_CRITIC:
                for layer in self.ac.critic[:-1]:
                    if isinstance(layer, nn.Linear):
                        _ortho(layer)
                _ortho(self.ac.critic[-1], gain=1.0)
                self.ppo.vf_opt = optim.Adam(
                    self.ac.critic.parameters(), lr=VF_LR, eps=1e-5
                )
                self.ppo._rms = RunningMeanStd()
                self.warmup_off = 0
                print("RESET_CRITIC=1: critic reset, warmup re-enabled")
            print(f"Resumed from {BEST_PT.name}")
            return True
        return False

    def baseline(self):
        ds = (
            self.ds_min + 0.5 * (self.ds_max - self.ds_min) * (1 + np.cos(0.0))
            if DELTA_SCALE_DECAY
            else self.ds_max
        )
        vm, vs = evaluate(
            self.ac, self.va_f, self.mdl_path, self.ort_sess, self.csv_cache, ds=ds
        )
        self.best = vm
        self.best_ep = "init"
        print(f"Baseline: {vm:.1f} ± {vs:.1f}  (Δs={ds:.4f})")


TEMP_START = float(os.getenv("TEMP_START", "0.8"))  # starting sim temperature
TEMP_END = float(os.getenv("TEMP_END", "0.8"))  # ending sim temperature
TEMP_RAMP = int(os.getenv("TEMP_RAMP", "0"))  # epochs to ramp over (0=no ramp)


def train_one_epoch(epoch, ctx):
    """One epoch: rollout → PPO update → log → eval. Returns log line."""
    # Sim temperature curriculum
    if TEMP_RAMP > 0:
        t = min(epoch / TEMP_RAMP, 1.0)
        sim_temp = TEMP_START + t * (TEMP_END - TEMP_START)
    else:
        sim_temp = float(os.getenv("SIM_TEMP", "0.8"))
    ctx.sim_temp = sim_temp

    # Delta scale
    if DELTA_SCALE_DECAY:
        ds = ctx.ds_min + 0.5 * (ctx.ds_max - ctx.ds_min) * (
            1 + np.cos(np.pi * epoch / MAX_EP)
        )
    else:
        ds = ctx.ds_max
    ctx.cur_ds = ds

    # Learning rate
    if RESUME and RESUME_OPT and epoch == 0:
        pi_lr = ctx.ppo.pi_opt.param_groups[0]["lr"]
        vf_lr = ctx.ppo.vf_opt.param_groups[0]["lr"]
    elif LR_DECAY:
        pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
        vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
    else:
        pi_lr, vf_lr = PI_LR, VF_LR
    for pg in ctx.ppo.pi_opt.param_groups:
        pg["lr"] = pi_lr
    for pg in ctx.ppo.vf_opt.param_groups:
        pg["lr"] = vf_lr

    # Rollout
    t0 = time.time()
    n_routes = min(CSVS_EPOCH, len(ctx.tr_f)) // SAMPLES_PER_ROUTE
    batch = random.sample(ctx.tr_f, max(n_routes, 1))
    batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]
    res = rollout(
        batch,
        ctx.ac,
        ctx.mdl_path,
        ctx.ort_sess,
        ctx.csv_cache,
        deterministic=False,
        ds=ds,
        sim_temp=sim_temp,
    )
    t1 = time.time()

    # PPO update
    co = epoch < (CRITIC_WARMUP - ctx.warmup_off)
    info = ctx.ppo.update(res, critic_only=co, ds=ds)
    tu = time.time() - t1

    # Log
    phase = "  [critic warmup]" if co else ""
    line = (
        f"E{epoch:3d}  T={sim_temp:.2f}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}"
        f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}"
        f"  adv={info['adv_std']:.4f}  spread={info['cost_spread']:.1f}"
        f"  lr={info['lr']:.1e}  ⏱{t1 - t0:.0f}+{tu:.0f}s{phase}"
    )

    # Eval
    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(
            ctx.ac, ctx.va_f, ctx.mdl_path, ctx.ort_sess, ctx.csv_cache, ds=ds
        )
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            ctx.save_best()
            mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"

    print(line)
    return info


def train():
    ctx = TrainingContext()

    if not ctx.resume():
        ctx.warm_start_from_exp055()

    ctx.baseline()

    n_r = min(CSVS_EPOCH, len(ctx.tr_f)) // SAMPLES_PER_ROUTE
    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(
        f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {n_r} routes × {SAMPLES_PER_ROUTE} = {n_r * SAMPLES_PER_ROUTE} rollouts/epoch"
    )
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  γ={GAMMA}  λ={LAMDA}  K={K_EPOCHS}  dim={STATE_DIM}\n"
    )

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best: {ctx.best:.1f} (epoch {ctx.best_ep})")
    torch.save({"ac": ctx.ac.state_dict()}, EXP_DIR / "final_model.pt")


if __name__ == "__main__":
    train()
