# exp102 — Transformer policy for noise-robust control
#
# Replace the 4-layer MLP actor with a small causal transformer that processes
# a sequence of per-step observation tokens. The hypothesis: a transformer can
# learn to read the noise pattern in the recent lataccel history and compensate.
#
# Per-step token (TOKEN_DIM=8):
#   (target, current, error, action, roll_la, v_ego, a_ego, jerk_est)
#   All pre-scaled.
#
# The transformer processes the last SEQ_LEN=20 tokens with causal attention.
# The output at the last position is concatenated with the future plan (200 dims)
# and fed through an MLP head → 2 values (alpha, beta for Beta distribution).
#
# Everything else (PPO, rollout, BC, delta-action, BatchedSimulator) is identical
# to exp095_clean.

import numpy as np, pandas as pd, os, sys, time, random, math
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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # allow tf32 for faster matmuls

# ── architecture ──────────────────────────────────────────────
SEQ_LEN = 20  # matches physics model context length
TOKEN_DIM = 13  # per-step features (raw + derived + integral error)
FUTURE_K = 50
FUTURE_DIM = (
    FUTURE_K * 4
)  # 200: for flat storage, but model uses (FUTURE_K, 4) sequence

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MLP_RATIO = 2
DROPOUT = 0.0

HIDDEN = 256  # MLP head hidden dim
DELTA_SCALE_MAX = float(os.getenv("DELTA_SCALE_MAX", "0.25"))
DELTA_SCALE_MIN = float(os.getenv("DELTA_SCALE_MIN", "0.25"))

# For compatibility with existing code
HIST_LEN = SEQ_LEN
OBS_DIM = TOKEN_DIM  # not used directly, but needed for some interfaces

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL = 2.0

# ── PPO ───────────────────────────────────────────────────────
PI_LR = float(os.getenv("PI_LR", "3e-4"))
VF_LR = float(os.getenv("VF_LR", "3e-4"))
LR_MIN = 5e-5
GAMMA = float(os.getenv("GAMMA", "0.95"))
LAMDA = float(os.getenv("LAMDA", "0.9"))
K_EPOCHS = int(os.getenv("K_EPOCHS", "3"))
EPS_CLIP = 0.2
VF_COEF = 1.0
ENT_COEF = float(os.getenv("ENT_COEF", "0.003"))
SIGMA_FLOOR = float(os.getenv("SIGMA_FLOOR", "0.01"))
SIGMA_FLOOR_COEF = float(os.getenv("SIGMA_FLOOR_COEF", "0.5"))
ACT_SMOOTH = float(os.getenv("ACT_SMOOTH", "0.0"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1.0"))
MINI_BS = int(os.getenv("MINI_BS", "8192"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "0"))
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
LR_DECAY = os.getenv("LR_DECAY", "1") == "1"
REWARD_RMS_NORM = os.getenv("REWARD_RMS_NORM", "1") == "1"
ADV_NORM = os.getenv("ADV_NORM", "1") == "1"


def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"


# ══════════════════════════════════════════════════════════════
#  Transformer Model
# ══════════════════════════════════════════════════════════════


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Use PyTorch's scaled_dot_product_attention with causal mask
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


FUTURE_TOKEN_DIM = 4  # per-step future features: target, roll, v_ego, a_ego


class CrossAttention(nn.Module):
    """Single query attends to a set of key-value pairs."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, context):
        """query: (B, 1, D), context: (B, T, D) → (B, 1, D)"""
        B = query.shape[0]
        q = (
            self.q_proj(query)
            .reshape(B, 1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_proj(context).reshape(B, -1, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, 1, self.n_heads * self.head_dim)
        return self.out_proj(out)


class BidirectionalSelfAttention(nn.Module):
    """Full (non-causal) self-attention for fully-observed sequences."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # no causal mask
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class BidirectionalBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ActorCritic(nn.Module):
    """
    Physics-inverse controller architecture:

    The physics model predicts lataccel[t+1] from context[t-19:t].
    The controller inverts this: given context + future targets, predict the action.

    History encoder (feedback path):
      - Causal self-attention over 20 history tokens
      - Mirrors the physics model's context processing
      - Output at last position = "plant state summary"

    Future encoder (feedforward path):
      - Bidirectional self-attention over 50 future tokens
      - Learns curve shapes, speed changes, reference trajectory patterns
      - Full attention because the future is fully observed

    Fusion:
      - History output (plant state) as QUERY into encoded future
      - "Given where I am and what the noise looks like, what part of the
         future plan is relevant for my next action?"
      - Cross-attention output = state-conditioned plan representation

    Action head:
      - MLP on fused representation → (alpha, beta) for steer delta
    """

    def __init__(self):
        super().__init__()

        # ── History encoder (causal, mirrors physics model) ──
        self.hist_proj = nn.Linear(TOKEN_DIM, D_MODEL)
        self.hist_pos = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.hist_blocks = nn.ModuleList(
            [
                TransformerBlock(D_MODEL, N_HEADS, MLP_RATIO, DROPOUT)
                for _ in range(N_LAYERS)
            ]
        )
        self.hist_ln = nn.LayerNorm(D_MODEL)

        # ── Future encoder (bidirectional, learns plan patterns) ──
        self.fut_proj = nn.Linear(FUTURE_TOKEN_DIM, D_MODEL)
        self.fut_pos = nn.Parameter(torch.randn(1, FUTURE_K, D_MODEL) * 0.02)
        self.fut_block = BidirectionalBlock(D_MODEL, N_HEADS, MLP_RATIO)
        self.fut_ln = nn.LayerNorm(D_MODEL)

        # ── Fusion: history state queries the future ──
        self.cross_attn_ln_q = nn.LayerNorm(D_MODEL)
        self.cross_attn_ln_kv = nn.LayerNorm(D_MODEL)
        self.cross_attn = CrossAttention(D_MODEL, N_HEADS)
        self.cross_mlp_ln = nn.LayerNorm(D_MODEL)
        self.cross_mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * MLP_RATIO),
            nn.GELU(),
            nn.Linear(D_MODEL * MLP_RATIO, D_MODEL),
        )
        self.out_ln = nn.LayerNorm(D_MODEL)

        # ── Actor head ──
        self.actor = nn.Sequential(
            nn.Linear(D_MODEL, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )

        # ── Critic head ──
        self.critic = nn.Sequential(
            nn.Linear(D_MODEL, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _encode(self, seq_tokens, future_seq):
        """
        seq_tokens: (B, SEQ_LEN, TOKEN_DIM) — history context
        future_seq: (B, FUTURE_K, FUTURE_TOKEN_DIM) — future plan
        Returns: (B, D_MODEL) — fused representation for action prediction
        """
        # Encode history with causal self-attention
        h = self.hist_proj(seq_tokens) + self.hist_pos
        for block in self.hist_blocks:
            h = block(h)
        h = self.hist_ln(h)
        # Plant state summary = last position output
        state = h[:, -1:, :]  # (B, 1, D_MODEL)

        # Encode future with bidirectional self-attention
        f = self.fut_proj(future_seq) + self.fut_pos
        f = self.fut_block(f)
        f = self.fut_ln(f)  # (B, FUTURE_K, D_MODEL)

        # State-conditioned cross-attention into future
        q = self.cross_attn_ln_q(state)  # (B, 1, D_MODEL)
        kv = self.cross_attn_ln_kv(f)  # (B, FUTURE_K, D_MODEL)
        z = state + self.cross_attn(q, kv)  # residual
        z = z + self.cross_mlp(self.cross_mlp_ln(z))
        z = self.out_ln(z).squeeze(1)  # (B, D_MODEL)

        return z

    def forward_actor(self, seq_tokens, future_seq):
        return self.actor(self._encode(seq_tokens, future_seq))

    def forward_critic(self, seq_tokens, future_seq):
        return self.critic(self._encode(seq_tokens, future_seq)).squeeze(-1)

    def forward_both(self, seq_tokens, future_seq):
        z = self._encode(seq_tokens, future_seq)
        return self.actor(z), self.critic(z).squeeze(-1)

    def beta_params(self, seq_tokens, future_seq):
        out = self.forward_actor(seq_tokens, future_seq)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0


# ══════════════════════════════════════════════════════════════
#  Observation: per-step tokens + future plan
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _w(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        k: _w(dg[k]) for k in ("target_lataccel", "roll_lataccel", "v_ego", "a_ego")
    }


def build_future(future_windows, step_idx):
    """Build future plan as sequence (N, FUTURE_K, 4) for current step.
    Each future step has 4 features: target, roll, v_ego, a_ego."""
    return torch.stack(
        [
            future_windows["target_lataccel"][:, step_idx] / S_LAT,
            future_windows["roll_lataccel"][:, step_idx] / S_ROLL,
            future_windows["v_ego"][:, step_idx] / S_VEGO,
            future_windows["a_ego"][:, step_idx] / S_AEGO,
        ],
        dim=-1,
    ).clamp(-5.0, 5.0)


S_CURV = 0.02


def build_token(
    target, current, action, roll_la, v_ego, a_ego, prev_lataccel, int_error
):
    """Build a single per-step token (N, TOKEN_DIM=13)."""
    error = target - current
    jerk_est = (current - prev_lataccel) / DEL_T
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2  # target curvature
    k_cur = (current - roll_la) / v2  # current curvature
    k_err = k_tgt - k_cur  # curvature error
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0  # friction estimate
    return torch.stack(
        [
            target / S_LAT,
            current / S_LAT,
            error / S_LAT,
            action / S_STEER,
            roll_la / S_ROLL,
            v_ego / S_VEGO,
            a_ego / S_AEGO,
            jerk_est / S_LAT,
            k_tgt / S_CURV,
            k_cur / S_CURV,
            k_err / S_CURV,
            fric,
            int_error / S_LAT,
        ],
        dim=-1,
    ).clamp(-5.0, 5.0)


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
    future_windows = _precompute_future_windows(dg)

    # Token sequence buffer: ring buffer of per-step tokens
    seq_buf = torch.zeros((N, SEQ_LEN, TOKEN_DIM), dtype=torch.float32, device="cuda")
    seq_head = SEQ_LEN - 1  # points to most recent token
    prev_action = torch.zeros(N, dtype=torch.float64, device="cuda")
    prev_lataccel = torch.zeros(N, dtype=torch.float32, device="cuda")
    # Integral error tracking (running sum of errors, windowed)
    err_hist = torch.zeros((N, SEQ_LEN), dtype=torch.float32, device="cuda")
    err_head = SEQ_LEN - 1
    int_error = torch.zeros(N, dtype=torch.float32, device="cuda")

    if not deterministic:
        # Store flattened (seq_tokens, future) pairs for PPO
        all_seq = torch.empty(
            (max_steps, N, SEQ_LEN, TOKEN_DIM), dtype=torch.float32, device="cuda"
        )
        all_fut = torch.empty(
            (max_steps, N, FUTURE_K, FUTURE_TOKEN_DIM),
            dtype=torch.float32,
            device="cuda",
        )
        all_raw = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_val = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")

    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si, seq_head, prev_action, prev_lataccel, err_head, int_error
        target = dg["target_lataccel"][:, step_idx].float()
        current = sim_ref.current_lataccel.float()
        roll_la = dg["roll_lataccel"][:, step_idx].float()
        v_ego = dg["v_ego"][:, step_idx].float()
        a_ego = dg["a_ego"][:, step_idx].float()

        # Update integral error (windowed running sum)
        error = (target - current).float()
        next_eh = (err_head + 1) % SEQ_LEN
        old_err = err_hist[:, next_eh]
        err_hist[:, next_eh] = error
        int_error = int_error + error - old_err
        err_head = next_eh
        ei = int_error * (DEL_T / SEQ_LEN)

        # Build token for this step and write into ring buffer
        next_head = (seq_head + 1) % SEQ_LEN
        seq_buf[:, next_head] = build_token(
            target,
            current,
            prev_action.float(),
            roll_la,
            v_ego,
            a_ego,
            prev_lataccel,
            ei,
        )
        seq_head = next_head

        if step_idx < CONTROL_START_IDX:
            prev_lataccel = current
            prev_action = torch.zeros(N, dtype=torch.float64, device="cuda")
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        # Unroll ring buffer into sequential order: oldest first
        split = seq_head + 1
        if split >= SEQ_LEN:
            seq_ordered = seq_buf.clone()
        else:
            seq_ordered = torch.cat([seq_buf[:, split:], seq_buf[:, :split]], dim=1)

        future = build_future(future_windows, step_idx)

        with torch.no_grad():
            logits, val = ac.forward_both(seq_ordered, future)
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

        delta = raw.double() * ds
        action = (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        if not deterministic and step_idx < COST_END_IDX:
            all_seq[si] = seq_ordered
            all_fut[si] = future
            all_raw[si] = raw
            all_logp[si] = logp
            all_val[si] = val
            si += 1

        prev_lataccel = current
        prev_action = action
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    if deterministic:
        return costs.tolist()

    S = si
    start = CONTROL_START_IDX
    end = start + S
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
        seq=all_seq[:S].permute(1, 0, 2, 3).reshape(-1, SEQ_LEN, TOKEN_DIM),
        fut=all_fut[:S].permute(1, 0, 2, 3).reshape(-1, FUTURE_K, FUTURE_TOKEN_DIM),
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


def _bc_worker(csv_path):
    """Extract (seq_tokens, future, raw_delta) tuples for BC."""
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values
    steer = -df["steerCommand"].values

    seq_list, fut_list, raw_list = [], [], []
    # Ring buffer for token sequence
    token_ring = np.zeros((SEQ_LEN, TOKEN_DIM), dtype=np.float32)
    ring_head = SEQ_LEN - 1
    prev_act = 0.0
    prev_lat = 0.0
    # Integral error tracking
    err_ring = np.zeros(SEQ_LEN, dtype=np.float32)
    err_head = SEQ_LEN - 1
    int_err = 0.0

    for si in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        # Build token
        target = tgt[si]
        current = tgt[si]  # before control, current = target
        error = target - current
        jerk_est = (current - prev_lat) / DEL_T
        v2 = max(v_ego[si] ** 2, 1.0)
        k_tgt = (target - roll_la[si]) / v2
        k_cur = (current - roll_la[si]) / v2
        k_err = k_tgt - k_cur
        fric = np.sqrt(current**2 + a_ego[si] ** 2) / 7.0
        # Update integral error
        next_eh = (err_head + 1) % SEQ_LEN
        int_err = int_err + error - err_ring[next_eh]
        err_ring[next_eh] = error
        err_head = next_eh
        ei = int_err * (DEL_T / SEQ_LEN)
        token = np.clip(
            np.array(
                [
                    target / S_LAT,
                    current / S_LAT,
                    error / S_LAT,
                    prev_act / S_STEER,
                    roll_la[si] / S_ROLL,
                    v_ego[si] / S_VEGO,
                    a_ego[si] / S_AEGO,
                    jerk_est / S_LAT,
                    k_tgt / S_CURV,
                    k_cur / S_CURV,
                    k_err / S_CURV,
                    fric,
                    ei / S_LAT,
                ],
                np.float32,
            ),
            -5.0,
            5.0,
        )

        ring_head = (ring_head + 1) % SEQ_LEN
        token_ring[ring_head] = token

        # Unroll ring buffer
        split = ring_head + 1
        if split >= SEQ_LEN:
            seq = token_ring.copy()
        else:
            seq = np.concatenate([token_ring[split:], token_ring[:split]])

        # Future plan as (FUTURE_K, 4) sequence
        state = State(roll_lataccel=roll_la[si], v_ego=v_ego[si], a_ego=a_ego[si])
        fplan = FuturePlan(
            lataccel=tgt[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[si + 1 : si + FUTURE_PLAN_STEPS].tolist(),
        )
        fut = np.clip(
            np.stack(
                [
                    _future_raw(fplan, "lataccel", target) / S_LAT,
                    _future_raw(fplan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
                    _future_raw(fplan, "v_ego", state.v_ego) / S_VEGO,
                    _future_raw(fplan, "a_ego", state.a_ego) / S_AEGO,
                ],
                axis=-1,  # (FUTURE_K, 4)
            ),
            -5.0,
            5.0,
        )

        seq_list.append(seq)
        fut_list.append(fut)
        raw_list.append(np.clip((steer[si] - prev_act) / DELTA_SCALE_MAX, -1.0, 1.0))

        prev_act = steer[si]
        prev_lat = current

    return (
        np.array(seq_list, np.float32),
        np.array(fut_list, np.float32),
        np.array(raw_list, np.float32),
    )


def pretrain_bc(ac, all_csvs):
    if BC_EPOCHS <= 0:
        print("BC pretrain: skipped (BC_EPOCHS=0)")
        return
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(
        _bc_worker, [str(f) for f in all_csvs], max_workers=10, chunksize=50
    )
    seq_t = torch.FloatTensor(np.concatenate([r[0] for r in results])).to(DEV)
    fut_t = torch.FloatTensor(np.concatenate([r[1] for r in results])).to(DEV)
    raw_t = torch.FloatTensor(np.concatenate([r[2] for r in results])).to(DEV)
    N = len(seq_t)
    print(f"BC pretrain: {N} samples, {BC_EPOCHS} epochs")
    opt = optim.AdamW(
        [p for p in ac.parameters() if p.requires_grad],  # all params for BC
        lr=BC_LR,
        weight_decay=1e-4,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)
    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            a_p, b_p = ac.beta_params(seq_t[idx], fut_t[idx])
            x = ((raw_t[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), BC_GRAD_CLIP)
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
        # Single optimizer for everything (encoder is shared between actor & critic)
        self.opt = optim.Adam(ac.parameters(), lr=PI_LR, eps=1e-5)
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
        seq = gd["seq"]
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

        ADV_CLIP = float(os.getenv("ADV_CLIP", "0"))
        if ADV_CLIP > 0:
            adv_t = adv_t.clamp(-ADV_CLIP, ADV_CLIP)

        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)

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

        with torch.no_grad():
            old_lp = gd.get("old_logp")
            if old_lp is None:
                a_old, b_old = self.ac.beta_params(seq, fut)
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(
                    x_t.squeeze(-1)
                )

        n_vf, n_actor = 0, 0
        vf_sum, pi_sum, ent_sum, sigma_pen_sum = 0.0, 0.0, 0.0, 0.0

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(seq), device="cuda").split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                bs = idx.numel()

                if critic_only:
                    val = self.ac.forward_critic(seq[idx], fut[idx])
                    vf_loss = F.mse_loss(val, ret_t[idx])
                    vf_sum += vf_loss.item() * bs
                    n_vf += bs
                    self.opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                    self.opt.step()
                else:
                    # Single _encode call for both actor and critic
                    actor_out, val = self.ac.forward_both(seq[idx], fut[idx])
                    vf_loss = F.mse_loss(val, ret_t[idx])
                    vf_sum += vf_loss.item() * bs
                    n_vf += bs

                    a_c = F.softplus(actor_out[..., 0]) + 1.0
                    b_c = F.softplus(actor_out[..., 1]) + 1.0
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv,
                        ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv,
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
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                    self.opt.step()
                    pi_sum += pi_loss.item() * bs
                    ent_sum += ent.item() * bs
                    sigma_pen_sum += sigma_pen.item() * bs
                    n_actor += bs

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(seq[:1000], fut[:1000])
            σraw = self._beta_sigma_raw(a_d, b_d).mean().item()
        return dict(
            pi=pi_sum / max(1, n_actor) if not critic_only else 0.0,
            vf=vf_sum / max(1, n_vf),
            ent=ent_sum / max(1, n_actor) if not critic_only else 0.0,
            σ=σraw * ds,
            σraw=σraw,
            σpen=sigma_pen_sum / max(1, n_actor) if not critic_only else 0.0,
            lr=self.opt.param_groups[0]["lr"],
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
        n_params = sum(p.numel() for p in self.ac.parameters())
        n_hist = sum(p.numel() for p in self.ac.hist_blocks.parameters())
        print(f"Model: {n_params:,} params  (hist_encoder={n_hist:,})")

    def save_best(self):
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "opt": self.ppo.opt.state_dict(),
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
            if RESUME_OPT and "opt" in ckpt:
                self.ppo.opt.load_state_dict(ckpt["opt"])
                if "ret_rms" in ckpt:
                    r = ckpt["ret_rms"]
                    self.ppo._rms.mean, self.ppo._rms.var, self.ppo._rms.count = (
                        r["mean"],
                        r["var"],
                        r["count"],
                    )
            self.warmup_off = CRITIC_WARMUP
            print(f"Resumed from {BEST_PT.name}")
            return True
        return False

    def baseline(self):
        vm, vs = evaluate(
            self.ac,
            self.va_f,
            self.mdl_path,
            self.ort_sess,
            self.csv_cache,
            ds=self.ds_max,
        )
        self.best = vm
        self.best_ep = "init"
        print(f"Baseline: {vm:.1f} ± {vs:.1f}")


SIM_TEMP = float(os.getenv("SIM_TEMP", "0.8"))


def train_one_epoch(epoch, ctx):
    sim_temp = SIM_TEMP
    ds = ctx.ds_max

    if LR_DECAY:
        pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
        vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
    else:
        pi_lr, vf_lr = PI_LR, VF_LR
    for pg in ctx.ppo.opt.param_groups:
        pg["lr"] = pi_lr

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

    co = epoch < (CRITIC_WARMUP - ctx.warmup_off)
    info = ctx.ppo.update(res, critic_only=co, ds=ds)
    tu = time.time() - t1

    phase = "  [critic warmup]" if co else ""
    line = (
        f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}"
        f"  π={info['pi']:+.4f}  vf={info['vf']:.4f}"
        f"  adv={info['adv_std']:.4f}  spread={info['cost_spread']:.1f}"
        f"  lr={info['lr']:.1e}  ⏱{t1 - t0:.0f}+{tu:.0f}s{phase}"
    )

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(
            ctx.ac,
            ctx.va_f,
            ctx.mdl_path,
            ctx.ort_sess,
            ctx.csv_cache,
            ds=ds,
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
        pretrain_bc(ctx.ac, ctx.all_csv)

    # Compile encoder for faster PPO updates
    ctx.ac._encode = torch.compile(ctx.ac._encode, mode="max-autotune")
    print("torch.compile applied")

    ctx.baseline()

    n_r = min(CSVS_EPOCH, len(ctx.tr_f)) // SAMPLES_PER_ROUTE
    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(
        f"  transformer: d={D_MODEL} heads={N_HEADS} layers={N_LAYERS} seq={SEQ_LEN} tok={TOKEN_DIM}"
    )
    print(
        f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {n_r} routes × {SAMPLES_PER_ROUTE} = {n_r * SAMPLES_PER_ROUTE} rollouts/epoch"
    )
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  γ={GAMMA}  λ={LAMDA}  K={K_EPOCHS}\n"
    )

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best: {ctx.best:.1f} (epoch {ctx.best_ep})")
    torch.save({"ac": ctx.ac.state_dict()}, EXP_DIR / "final_model.pt")


if __name__ == "__main__":
    train()
