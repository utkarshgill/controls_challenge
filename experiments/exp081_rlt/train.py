# exp081 — RLT-style residual RL on frozen exp055 policy
#
# Architecture (from PI's RLT paper):
#   - Frozen exp055 actor (263K params) provides base_delta and features
#   - Tiny residual actor (2×64, ~8K params) predicts correction to base_delta
#   - Twin critics (2×64 each) for TD3 off-policy learning
#   - final_delta = base_delta + correction
#
# Key differences from exp065 (which diverged):
#   - Reward scaled to O(1) per step from the start
#   - Long critic warmup (500K steps) before actor trains
#   - Target Q hard-clamped to [-10, 10]
#   - BC regularization with β=2.0
#   - Reference-action dropout (50%) per RLT paper
#   - Small exploration noise (0.02)
#   - Conservative UTD ratio (2)
#   - MAX_CORRECTION=0.1 (not 0.3) — small refinements only

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

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))

# ── obs scaling (must match exp055) ───────────────────────────
S_LAT, S_STEER, S_VEGO, S_AEGO, S_ROLL, S_CURV = 5.0, 2.0, 40.0, 4.0, 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256

# ── residual actor / critic ───────────────────────────────────
RES_HIDDEN = int(os.getenv("RES_HIDDEN", "64"))
MAX_CORRECTION = float(os.getenv("MAX_CORRECTION", "0.1"))

# ── TD3 hyperparameters ──────────────────────────────────────
ACTOR_LR = float(os.getenv("ACTOR_LR", "1e-4"))
CRITIC_LR = float(os.getenv("CRITIC_LR", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.95"))
TAU = float(os.getenv("TAU", "0.005"))
EXPLORE_NOISE = float(os.getenv("EXPLORE_NOISE", "0.02"))
TARGET_NOISE = float(os.getenv("TARGET_NOISE", "0.03"))
NOISE_CLIP = float(os.getenv("NOISE_CLIP", "0.05"))
ACTOR_DELAY = int(os.getenv("ACTOR_DELAY", "2"))
UTD_RATIO = int(os.getenv("UTD_RATIO", "2"))
BC_BETA = float(os.getenv("BC_BETA", "2.0"))
REF_DROPOUT = float(os.getenv("REF_DROPOUT", "0.5"))
Q_CLAMP = float(os.getenv("Q_CLAMP", "10.0"))
BATCH_RL = int(os.getenv("BATCH_RL", "256"))

# ── replay buffer ────────────────────────────────────────────
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "2000000"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "50000"))

# ── runtime ──────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "500"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = 100

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)

# Reward normalization: raw per-step cost is O(50-200), we want rewards O(1)
# total_cost ~ 42 over 400 steps, so per-step ~ 0.1
# But lat_r per step = err^2 * 5000, and jerk per step = jerk^2 * 100
# Typical per-step reward magnitude ~ 5-50, so scale by 0.01
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "0.01"))


# ══════════════════════════════════════════════════════════════
#  Frozen base actor (from exp055)
# ══════════════════════════════════════════════════════════════


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*layers)
        self._features = None
        # Hook on penultimate ReLU to capture features
        self.actor[-2].register_forward_hook(self._hook)
        for p in self.parameters():
            p.requires_grad = False

    def _hook(self, mod, inp, out):
        self._features = out

    def forward(self, obs):
        """Returns (base_delta, features)."""
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        base_raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return base_raw, self._features.detach()


# ══════════════════════════════════════════════════════════════
#  Residual actor (tiny, trainable)
# ══════════════════════════════════════════════════════════════


class ResidualActor(nn.Module):
    """Input: [features(256), base_delta(1)] -> correction(1)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN + 1, RES_HIDDEN),
            nn.ReLU(),
            nn.Linear(RES_HIDDEN, RES_HIDDEN),
            nn.ReLU(),
            nn.Linear(RES_HIDDEN, 1),
        )
        # Init last layer to zero so correction starts at 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base_delta):
        """features: (B, 256), base_delta: (B, 1) -> correction: (B, 1)."""
        x = torch.cat([features, base_delta], dim=-1)
        return torch.tanh(self.net(x)) * MAX_CORRECTION


# ══════════════════════════════════════════════════════════════
#  Twin critic
# ══════════════════════════════════════════════════════════════


class TwinCritic(nn.Module):
    """Two Q-networks. Input: [features(256), action(1)] -> Q(1)."""

    def __init__(self):
        super().__init__()

        def _make():
            return nn.Sequential(
                nn.Linear(HIDDEN + 1, RES_HIDDEN),
                nn.ReLU(),
                nn.Linear(RES_HIDDEN, RES_HIDDEN),
                nn.ReLU(),
                nn.Linear(RES_HIDDEN, 1),
            )

        self.q1 = _make()
        self.q2 = _make()

    def forward(self, features, action):
        x = torch.cat([features, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, features, action):
        x = torch.cat([features, action], dim=-1)
        return self.q1(x)


# ══════════════════════════════════════════════════════════════
#  Replay buffer (GPU-resident)
# ══════════════════════════════════════════════════════════════


class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.feat = torch.zeros((max_size, HIDDEN), dtype=torch.float32, device=DEV)
        self.base_d = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.action = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.next_feat = torch.zeros(
            (max_size, HIDDEN), dtype=torch.float32, device=DEV
        )
        self.next_base = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)

    def add(self, feat, base_d, action, reward, next_feat, next_base, done):
        n = feat.shape[0]
        if self.ptr + n <= self.max_size:
            s = slice(self.ptr, self.ptr + n)
            self.feat[s] = feat
            self.base_d[s] = base_d
            self.action[s] = action
            self.reward[s] = reward
            self.next_feat[s] = next_feat
            self.next_base[s] = next_base
            self.done[s] = done
        else:
            rem = self.max_size - self.ptr
            self.add(
                feat[:rem],
                base_d[:rem],
                action[:rem],
                reward[:rem],
                next_feat[:rem],
                next_base[:rem],
                done[:rem],
            )
            self.ptr = 0
            self.add(
                feat[rem:],
                base_d[rem:],
                action[rem:],
                reward[rem:],
                next_feat[rem:],
                next_base[rem:],
                done[rem:],
            )
            return
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=DEV)
        return (
            self.feat[idx],
            self.base_d[idx],
            self.action[idx],
            self.reward[idx],
            self.next_feat[idx],
            self.next_base[idx],
            self.done[idx],
        )


# ══════════════════════════════════════════════════════════════
#  Obs builder (identical to exp055)
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
#  Rollout: collect transitions into replay buffer
# ══════════════════════════════════════════════════════════════


def collect_rollout(
    csv_files, base_actor, res_actor, mdl_path, ort_session, csv_cache, replay_buf
):
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

    # Storage for transitions (to add to replay buffer after rollout)
    S = COST_END_IDX - CONTROL_START_IDX
    t_feat = torch.empty((S, N, HIDDEN), dtype=torch.float32, device="cuda")
    t_base = torch.empty((S, N, 1), dtype=torch.float32, device="cuda")
    t_act = torch.empty((S, N, 1), dtype=torch.float32, device="cuda")
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
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
            base_raw, features = base_actor(obs_buf)
            correction = res_actor(features, base_raw.unsqueeze(-1))
            raw = (base_raw + correction.squeeze(-1)).clamp(-1.0, 1.0)

            # Exploration noise
            raw = raw + torch.randn_like(raw) * EXPLORE_NOISE
            raw = raw.clamp(-1.0, 1.0)

        delta = raw * DELTA_SCALE
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )

        # Store transition data
        if step_idx < COST_END_IDX:
            t_feat[si] = features
            t_base[si] = base_raw.unsqueeze(-1)
            t_act[si] = raw.unsqueeze(-1)
            si += 1

        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]

    # Compute per-step rewards from sim histories
    start, end = CONTROL_START_IDX, CONTROL_START_IDX + si
    pred = sim.current_lataccel_history[:, start:end].float()
    tgt = dg["target_lataccel"][:, start:end].float()
    lat_r = (tgt - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    rewards = -(lat_r + jerk**2 * 100) * REWARD_SCALE  # (N, S)

    # Add transitions to replay buffer
    for t in range(si - 1):
        done_t = torch.zeros((N, 1), device="cuda")
        if t == si - 2:
            done_t[:] = 1.0
        replay_buf.add(
            t_feat[t],
            t_base[t],
            t_act[t],
            rewards[:, t].unsqueeze(-1),
            t_feat[t + 1],
            t_base[t + 1],
            done_t,
        )

    return float(np.mean(costs))


# ══════════════════════════════════════════════════════════════
#  Eval (deterministic, no exploration noise)
# ══════════════════════════════════════════════════════════════


def evaluate(base_actor, res_actor, files, mdl_path, ort_session, csv_cache):
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
            base_raw, features = base_actor(obs_buf)
            correction = res_actor(features, base_raw.unsqueeze(-1))
            raw = (base_raw + correction.squeeze(-1)).clamp(-1.0, 1.0)
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
#  TD3 update
# ══════════════════════════════════════════════════════════════


def td3_update(
    res_actor,
    critic,
    critic_target,
    res_actor_target,
    actor_opt,
    critic_opt,
    replay_buf,
    total_steps,
):
    """One TD3 update step. Returns dict of metrics."""
    feat, base_d, action, reward, next_feat, next_base, done = replay_buf.sample(
        BATCH_RL
    )

    # ── Critic update ──
    with torch.no_grad():
        # Target actor with noise
        next_corr = res_actor_target(next_feat, next_base)
        next_act = (next_base + next_corr).clamp(-1.0, 1.0)
        noise = (torch.randn_like(next_act) * TARGET_NOISE).clamp(
            -NOISE_CLIP, NOISE_CLIP
        )
        next_act = (next_act + noise).clamp(-1.0, 1.0)

        tq1, tq2 = critic_target(next_feat, next_act)
        target_q = reward + GAMMA * (1 - done) * torch.min(tq1, tq2)
        target_q = target_q.clamp(-Q_CLAMP, Q_CLAMP)

    q1, q2 = critic(feat, action)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_opt.step()

    # ── Actor update (delayed) ──
    actor_loss_val = 0.0
    bc_loss_val = 0.0
    if total_steps % ACTOR_DELAY == 0:
        # Reference-action dropout
        if REF_DROPOUT > 0:
            mask = (torch.rand(feat.shape[0], 1, device=DEV) > REF_DROPOUT).float()
            base_input = base_d * mask
        else:
            base_input = base_d

        corr = res_actor(feat, base_input)
        act_for_q = (base_d + corr).clamp(-1.0, 1.0)

        q_val = critic.q1_only(feat, act_for_q)
        actor_loss = -q_val.mean()

        # BC regularization: penalize correction magnitude
        bc_loss = (corr**2).mean()

        total_loss = actor_loss + BC_BETA * bc_loss

        actor_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(res_actor.parameters(), 0.5)
        actor_opt.step()

        # Soft update targets
        for p, tp in zip(res_actor.parameters(), res_actor_target.parameters()):
            tp.data.mul_(1 - TAU).add_(p.data * TAU)
        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.data.mul_(1 - TAU).add_(p.data * TAU)

        actor_loss_val = actor_loss.item()
        bc_loss_val = bc_loss.item()

    return {
        "c_loss": critic_loss.item(),
        "a_loss": actor_loss_val,
        "bc_loss": bc_loss_val,
    }


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════


def train():
    # ── Load frozen base ──
    base_actor = BaseActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_keys = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    base_actor.load_state_dict(actor_keys, strict=False)
    base_actor.eval()
    ds = ckpt.get("delta_scale", None)
    if ds is not None:
        global DELTA_SCALE
        DELTA_SCALE = float(ds)
    print(f"Loaded frozen base from {BASE_PT} (Δs={DELTA_SCALE:.4f})")

    # ── Trainable components ──
    res_actor = ResidualActor().to(DEV)
    res_actor_target = ResidualActor().to(DEV)
    res_actor_target.load_state_dict(res_actor.state_dict())

    critic = TwinCritic().to(DEV)
    critic_target = TwinCritic().to(DEV)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(res_actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    n_res = sum(p.numel() for p in res_actor.parameters())
    n_crit = sum(p.numel() for p in critic.parameters())
    print(f"Residual actor: {n_res:,} params  Critic: {n_crit:,} params")

    # ── Replay buffer ──
    replay_buf = ReplayBuffer()

    # ── Sim ──
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # ── Baseline ──
    vm, vs = evaluate(base_actor, res_actor, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, "init"
    print(f"\nBaseline (base + zero correction): {vm:.1f} ± {vs:.1f}")

    print(f"\nTD3 RLT Training")
    print(f"  actor_lr={ACTOR_LR}  critic_lr={CRITIC_LR}  gamma={GAMMA}  tau={TAU}")
    print(f"  explore={EXPLORE_NOISE}  target_noise={TARGET_NOISE}  bc_beta={BC_BETA}")
    print(f"  ref_dropout={REF_DROPOUT}  max_corr={MAX_CORRECTION}  q_clamp={Q_CLAMP}")
    print(
        f"  utd={UTD_RATIO}  critic_warmup={CRITIC_WARMUP}  reward_scale={REWARD_SCALE}"
    )
    print(f"  buffer={BUFFER_SIZE}  csvs/ep={CSVS_EPOCH}")
    print()

    def save_best():
        torch.save(
            {
                "base_actor": base_actor.state_dict(),
                "res_actor": res_actor.state_dict(),
                "delta_scale": DELTA_SCALE,
            },
            BEST_PT,
        )

    total_steps = 0
    for epoch in range(MAX_EP):
        t0 = time.time()
        batch = random.sample(all_csv, min(CSVS_EPOCH, len(all_csv)))
        train_cost = collect_rollout(
            batch, base_actor, res_actor, mdl_path, ort_sess, csv_cache, replay_buf
        )
        t_collect = time.time() - t0

        # TD3 updates
        n_updates = min(replay_buf.size, CSVS_EPOCH * 400) * UTD_RATIO // BATCH_RL
        if replay_buf.size < BATCH_RL:
            n_updates = 0

        warmup = total_steps < CRITIC_WARMUP
        c_sum, a_sum, bc_sum, n_up = 0.0, 0.0, 0.0, 0
        t1 = time.time()
        for _ in range(n_updates):
            if warmup:
                # Critic-only warmup: still update critic, skip actor
                feat, base_d, action, reward, next_feat, next_base, done = (
                    replay_buf.sample(BATCH_RL)
                )
                with torch.no_grad():
                    next_corr = res_actor_target(next_feat, next_base)
                    next_act = (next_base + next_corr).clamp(-1.0, 1.0)
                    noise = (torch.randn_like(next_act) * TARGET_NOISE).clamp(
                        -NOISE_CLIP, NOISE_CLIP
                    )
                    next_act = (next_act + noise).clamp(-1.0, 1.0)
                    tq1, tq2 = critic_target(next_feat, next_act)
                    target_q = reward + GAMMA * (1 - done) * torch.min(tq1, tq2)
                    target_q = target_q.clamp(-Q_CLAMP, Q_CLAMP)
                q1, q2 = critic(feat, action)
                cl = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_opt.zero_grad()
                cl.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.mul_(1 - TAU).add_(p.data * TAU)
                c_sum += cl.item()
                n_up += 1
            else:
                info = td3_update(
                    res_actor,
                    critic,
                    critic_target,
                    res_actor_target,
                    actor_opt,
                    critic_opt,
                    replay_buf,
                    total_steps,
                )
                c_sum += info["c_loss"]
                a_sum += info["a_loss"]
                bc_sum += info["bc_loss"]
                n_up += 1
            total_steps += 1

        t_update = time.time() - t1

        # Diagnostics
        with torch.no_grad():
            if replay_buf.size >= 1000:
                sf, sb, sa, sr, _, _, _ = replay_buf.sample(1000)
                q1v = critic.q1_only(sf, sa)
                q_mean = q1v.mean().item()
                r_mean = sr.mean().item()
            else:
                q_mean = r_mean = 0.0

        phase = "  [warmup]" if warmup else ""
        c_avg = c_sum / max(1, n_up)
        a_avg = a_sum / max(1, n_up)
        bc_avg = bc_sum / max(1, n_up)
        line = (
            f"E{epoch:3d}  train={train_cost:6.1f}  C={c_avg:.2f}  A={a_avg:.3f}"
            f"  BC={bc_avg:.4f}  Q={q_mean:.2f}  R={r_mean:.3f}"
            f"  buf={replay_buf.size // 1000}K  ⏱{t_collect:.0f}+{t_update:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            vm, vs = evaluate(
                base_actor, res_actor, va_f, mdl_path, ort_sess, csv_cache
            )
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
