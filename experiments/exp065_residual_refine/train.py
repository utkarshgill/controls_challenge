# exp065 — Residual Refinement via Off-Policy TD3
#
# Inspired by PI's RLT paper: freeze the base PPO actor from exp055,
# extract penultimate-layer activations as a compact state representation,
# and train a small residual actor-critic with off-policy TD3 to refine
# the base policy's actions.
#
# Key design decisions:
#   - Frozen base actor (exp055 best checkpoint) provides base_delta + features
#   - Small residual actor (2-layer 128-hidden) outputs correction to base_delta
#   - Twin critics (TD3-style) for stable Q-learning
#   - Off-policy replay buffer — high update-to-data ratio
#   - BC regularization: actor loss = -Q(s,a) + beta * ||a - base_a||^2
#   - Reference-action dropout (50%) to prevent copying the base
#   - Expected-value physics for clean training signal
#   - Dense per-step reward (same as exp055)

import numpy as np, os, sys, time, random, collections
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
    State,
    FuturePlan,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture (base — must match exp055) ────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))

# ── residual architecture ─────────────────────────────────────
RES_HIDDEN = int(os.getenv("RES_HIDDEN", "128"))
RES_LAYERS = int(os.getenv("RES_LAYERS", "2"))
MAX_CORRECTION = float(
    os.getenv("MAX_CORRECTION", "0.3")
)  # max residual in raw [-1,1] space

# ── scaling (must match exp055) ───────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── TD3 / off-policy ─────────────────────────────────────────
ACTOR_LR = float(os.getenv("ACTOR_LR", "1e-4"))
CRITIC_LR = float(os.getenv("CRITIC_LR", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.95"))
TAU = float(os.getenv("TAU", "0.005"))  # target network soft update
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "2_000_000"))
MINI_BS = int(os.getenv("MINI_BS", "4096"))
UTD_RATIO = int(os.getenv("UTD_RATIO", "4"))  # gradient steps per env step
ACTOR_DELAY = int(os.getenv("ACTOR_DELAY", "2"))  # update actor every N critic updates
TARGET_NOISE = float(os.getenv("TARGET_NOISE", "0.05"))  # TD3 target policy smoothing
NOISE_CLIP = float(os.getenv("NOISE_CLIP", "0.1"))
EXPLORE_NOISE = float(
    os.getenv("EXPLORE_NOISE", "0.03")
)  # exploration noise during rollout
BC_BETA = float(os.getenv("BC_BETA", "2.0"))  # BC regularization weight
REF_DROPOUT = float(os.getenv("REF_DROPOUT", "0.5"))  # reference action dropout rate
CRITIC_WARMUP_STEPS = int(
    os.getenv("CRITIC_WARMUP_STEPS", "500000")
)  # train critic only first (~5 epochs)
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "0.001"))  # scale raw rewards down

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "500"))
SAMPLES_PER_ROUTE = int(
    os.getenv("SAMPLES_PER_ROUTE", "1")
)  # off-policy doesn't need batch-of-batch
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = 5
EVAL_N = 100
USE_EXPECTED = os.getenv("USE_EXPECTED", "1") == "1"

EXP_DIR = Path(__file__).parent
BASE_PT = Path(
    os.getenv(
        "BASE_MODEL",
        str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"),
    )
)
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout (must match exp055) ────────────────────────────
C = 16
H1 = C + HIST_LEN  # 36
H2 = H1 + HIST_LEN  # 56
F_LAT = H2  # 56
F_ROLL = F_LAT + FUTURE_K  # 106
F_V = F_ROLL + FUTURE_K  # 156
F_A = F_V + FUTURE_K  # 206
OBS_DIM = F_A + FUTURE_K  # 256


# ══════════════════════════════════════════════════════════════
#  Base Actor (frozen, from exp055)
# ══════════════════════════════════════════════════════════════


class BaseActor(nn.Module):
    """exp055 actor — loaded frozen. We hook the penultimate layer."""

    def __init__(self):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*layers)
        self._features = None
        # Hook on the last ReLU (before final linear) to capture features
        # That's layer index -2 (the ReLU) in the sequential
        self.actor[-2].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, obs):
        """Returns (base_delta, features_256)"""
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        # Deterministic: Beta mean mapped to [-1, 1]
        base_raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return base_raw, self._features.detach()


# ══════════════════════════════════════════════════════════════
#  Residual Actor and Twin Critics
# ══════════════════════════════════════════════════════════════


class ResidualActor(nn.Module):
    """Small MLP: (features_256, base_delta) -> correction"""

    def __init__(self):
        super().__init__()
        layers = []
        in_dim = HIDDEN + 1  # 256 features + 1 base delta
        for i in range(RES_LAYERS):
            out_dim = RES_HIDDEN
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        # Initialize last layer near zero so correction starts small
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base_delta):
        """
        features: (B, 256) — penultimate activation from frozen base
        base_delta: (B, 1) — base actor's raw delta in [-1, 1]
        Returns: correction in [-MAX_CORRECTION, MAX_CORRECTION]
        """
        x = torch.cat([features, base_delta], dim=-1)
        return torch.tanh(self.net(x)) * MAX_CORRECTION


class TwinCritic(nn.Module):
    """Two Q-networks for TD3."""

    def __init__(self):
        super().__init__()
        in_dim = HIDDEN + 1  # features + action delta
        self.q1 = self._build(in_dim)
        self.q2 = self._build(in_dim)

    def _build(self, in_dim):
        layers = []
        d = in_dim
        for _ in range(RES_LAYERS):
            layers += [nn.Linear(d, RES_HIDDEN), nn.ReLU()]
            d = RES_HIDDEN
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def forward(self, features, action_delta):
        """
        features: (B, 256)
        action_delta: (B, 1) — the actual delta applied (base + correction)
        Returns: (q1, q2) each (B, 1)
        """
        x = torch.cat([features, action_delta], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, features, action_delta):
        x = torch.cat([features, action_delta], dim=-1)
        return self.q1(x)


# ══════════════════════════════════════════════════════════════
#  Replay Buffer
# ══════════════════════════════════════════════════════════════


class ReplayBuffer:
    def __init__(self, max_size, feat_dim=HIDDEN):
        self.max_size = max_size
        self.feat_dim = feat_dim
        self.ptr = 0
        self.size = 0
        # Pre-allocate on GPU
        self.features = torch.zeros(
            (max_size, feat_dim), dtype=torch.float32, device=DEV
        )
        self.base_delta = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.action = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.next_feat = torch.zeros(
            (max_size, feat_dim), dtype=torch.float32, device=DEV
        )
        self.next_base = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)

    def add(self, features, base_delta, action, reward, next_feat, next_base, done):
        """Add a batch of transitions. All inputs are (B, ...) GPU tensors."""
        B = features.shape[0]
        if B == 0:
            return
        end = self.ptr + B
        if end <= self.max_size:
            self.features[self.ptr : end] = features
            self.base_delta[self.ptr : end] = base_delta
            self.action[self.ptr : end] = action
            self.reward[self.ptr : end] = reward
            self.next_feat[self.ptr : end] = next_feat
            self.next_base[self.ptr : end] = next_base
            self.done[self.ptr : end] = done
        else:
            # Wrap around
            first = self.max_size - self.ptr
            self.features[self.ptr :] = features[:first]
            self.base_delta[self.ptr :] = base_delta[:first]
            self.action[self.ptr :] = action[:first]
            self.reward[self.ptr :] = reward[:first]
            self.next_feat[self.ptr :] = next_feat[:first]
            self.next_base[self.ptr :] = next_base[:first]
            self.done[self.ptr :] = done[:first]
            rest = B - first
            self.features[:rest] = features[first:]
            self.base_delta[:rest] = base_delta[first:]
            self.action[:rest] = action[first:]
            self.reward[:rest] = reward[first:]
            self.next_feat[:rest] = next_feat[first:]
            self.next_base[:rest] = next_base[first:]
            self.done[:rest] = done[first:]
        self.ptr = end % self.max_size
        self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=DEV)
        return (
            self.features[idx],
            self.base_delta[idx],
            self.action[idx],
            self.reward[idx],
            self.next_feat[idx],
            self.next_base[idx],
            self.done[idx],
        )


# ══════════════════════════════════════════════════════════════
#  Observation builder (reused from exp055, GPU batched)
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


# ══════════════════════════════════════════════════════════════
#  GPU Rollout (collects transitions into replay buffer)
# ══════════════════════════════════════════════════════════════


def rollout(
    csv_files,
    base_actor,
    res_actor,
    mdl_path,
    ort_session,
    csv_cache,
    replay_buf=None,
    deterministic=False,
):
    """Run batched rollout. If replay_buf is provided, store transitions."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    if USE_EXPECTED and not deterministic:
        sim.use_expected = True
    N, T = sim.N, sim.T
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    # Storage for transitions (features, base_delta, action_delta, reward)
    if replay_buf is not None:
        step_features = []
        step_base_deltas = []
        step_action_deltas = []

    si = 0
    hist_head = HIST_LEN - 1
    prev_features = None
    prev_base_delta = None
    prev_action_delta = None

    def ctrl(step_idx, sim_ref):
        nonlocal \
            si, \
            hist_head, \
            err_sum, \
            prev_features, \
            prev_base_delta, \
            prev_action_delta
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

        with torch.no_grad():
            base_raw, features = base_actor(obs_buf)
            base_delta_1 = base_raw.unsqueeze(-1)  # (N, 1)

            if deterministic:
                correction = res_actor(features, base_delta_1)
                raw_policy = (base_raw + correction.squeeze(-1)).clamp(-1.0, 1.0)
            else:
                correction = res_actor(features, base_delta_1)
                raw_combined = base_raw + correction.squeeze(-1)
                # Add exploration noise
                noise = torch.randn_like(raw_combined) * EXPLORE_NOISE
                raw_policy = (raw_combined + noise).clamp(-1.0, 1.0)

        delta = raw_policy.to(h_act.dtype) * DELTA_SCALE
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # Store transition data for replay buffer
        if replay_buf is not None and not deterministic and step_idx < COST_END_IDX:
            actual_delta = raw_policy.detach().unsqueeze(
                -1
            )  # (N, 1) — what was actually used
            step_features.append(features.clone())
            step_base_deltas.append(base_delta_1.clone())
            step_action_deltas.append(actual_delta)
            si += 1

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)

    if replay_buf is not None and not deterministic and si > 0:
        # Compute per-step rewards from sim histories
        S = si
        start = CONTROL_START_IDX
        end = start + S
        pred = sim.current_lataccel_history[:, start:end].float()
        target_la = dg["target_lataccel"][:, start:end].float()
        act_hist = sim.action_history[:, start:end].float()

        lat_r = (target_la - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
        jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
        rewards = -(lat_r + jerk**2 * 100) * REWARD_SCALE  # (N, S) scaled down

        # Stack features/deltas: each is a list of S tensors of shape (N, dim)
        feats_all = torch.stack(step_features, dim=1)  # (N, S, 256)
        base_d_all = torch.stack(step_base_deltas, dim=1)  # (N, S, 1)
        act_d_all = torch.stack(step_action_deltas, dim=1)  # (N, S, 1)

        # Compute dones
        dones = torch.zeros((N, S), dtype=torch.float32, device="cuda")
        dones[:, -1] = 1.0

        # Add transitions to replay buffer
        # For each step t, transition is (feat_t, base_t, act_t, rew_t, feat_{t+1}, base_{t+1}, done_t)
        for t in range(S - 1):
            replay_buf.add(
                feats_all[:, t],
                base_d_all[:, t],
                act_d_all[:, t],
                rewards[:, t : t + 1],
                feats_all[:, t + 1],
                base_d_all[:, t + 1],
                dones[:, t : t + 1],
            )
        # Last step: next_feat/next_base can be same as current (terminal)
        replay_buf.add(
            feats_all[:, S - 1],
            base_d_all[:, S - 1],
            act_d_all[:, S - 1],
            rewards[:, S - 1 : S],
            feats_all[:, S - 1],  # doesn't matter, done=1
            base_d_all[:, S - 1],
            dones[:, S - 1 : S],
        )

    return costs["total_cost"]


# ══════════════════════════════════════════════════════════════
#  TD3 Trainer
# ══════════════════════════════════════════════════════════════


class TD3:
    def __init__(self, res_actor, critic):
        self.actor = res_actor
        self.critic = critic
        # Target networks (deep copy)
        self.actor_target = ResidualActor().to(DEV)
        self.actor_target.load_state_dict(res_actor.state_dict())
        self.critic_target = TwinCritic().to(DEV)
        self.critic_target.load_state_dict(critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.total_updates = 0

    def _soft_update(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def update(self, replay_buf, batch_size=MINI_BS, critic_only=False):
        """One TD3 update step. Returns dict of diagnostics."""
        feat, base_d, act_d, rew, next_feat, next_base_d, done = replay_buf.sample(
            batch_size
        )

        # ── Critic update ──
        with torch.no_grad():
            # Target policy smoothing
            next_correction = self.actor_target(next_feat, next_base_d)
            noise = (torch.randn_like(next_correction) * TARGET_NOISE).clamp(
                -NOISE_CLIP, NOISE_CLIP
            )
            next_action = (next_base_d + next_correction + noise).clamp(-1.0, 1.0)

            tq1, tq2 = self.critic_target(next_feat, next_action)
            target_q = rew + GAMMA * (1.0 - done) * torch.min(tq1, tq2)
            target_q = target_q.clamp(-100.0, 100.0)  # prevent runaway Q

        q1, q2 = self.critic(feat, act_d)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        bc_loss_val = 0.0
        self.total_updates += 1

        # ── Actor update (delayed) ──
        if not critic_only and self.total_updates % ACTOR_DELAY == 0:
            # Reference action dropout
            if REF_DROPOUT > 0:
                mask = (torch.rand(feat.shape[0], 1, device=DEV) > REF_DROPOUT).float()
                base_d_input = base_d * mask
            else:
                base_d_input = base_d

            correction = self.actor(feat, base_d_input)
            # For the Q evaluation, use the actual combined action (with real base_d, not dropped out)
            action_for_q = (base_d + correction).clamp(-1.0, 1.0)

            q_val = self.critic.q1_only(feat, action_for_q)
            actor_loss = -q_val.mean()

            # BC regularization: stay close to base action
            bc_loss = ((action_for_q - base_d) ** 2).mean()

            total_actor_loss = actor_loss + BC_BETA * bc_loss

            self.actor_opt.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            # Soft update targets
            self._soft_update(self.actor_target, self.actor, TAU)
            self._soft_update(self.critic_target, self.critic, TAU)

            actor_loss_val = actor_loss.item()
            bc_loss_val = bc_loss.item()

        return dict(
            critic=critic_loss.item(),
            actor=actor_loss_val,
            bc=bc_loss_val,
        )


# ══════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════


def evaluate(base_actor, res_actor, files, mdl_path, ort_session, csv_cache):
    base_actor.eval()
    res_actor.eval()
    costs = rollout(
        files,
        base_actor,
        res_actor,
        mdl_path,
        ort_session,
        csv_cache,
        replay_buf=None,
        deterministic=True,
    )
    return float(np.mean(costs)), float(np.std(costs))


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════


def train():
    # ── Load frozen base actor ──
    base_actor = BaseActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    # exp055 checkpoint has 'ac' with 'actor.*' and 'critic.*' keys
    ac_state = ckpt["ac"]
    # Extract only actor keys
    actor_state = {k: v for k, v in ac_state.items() if k.startswith("actor.")}
    base_actor.load_state_dict(actor_state)
    base_actor.eval()
    for p in base_actor.parameters():
        p.requires_grad = False

    ds_ckpt = ckpt.get("delta_scale", None)
    if ds_ckpt is not None:
        global DELTA_SCALE
        DELTA_SCALE = float(ds_ckpt)
        print(f"Using delta_scale={DELTA_SCALE:.4f} from checkpoint")

    print(f"Loaded frozen base actor from {BASE_PT}")
    print(
        f"  Base actor params: {sum(p.numel() for p in base_actor.parameters()):,} (all frozen)"
    )

    # ── Create residual actor + critic ──
    res_actor = ResidualActor().to(DEV)
    critic = TwinCritic().to(DEV)
    td3 = TD3(res_actor, critic)

    print(
        f"  Residual actor params: {sum(p.numel() for p in res_actor.parameters()):,}"
    )
    print(f"  Twin critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # ── Setup ──
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    tr_f = all_csv
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])
    replay_buf = ReplayBuffer(BUFFER_SIZE)

    # Baseline eval (frozen base only, residual = 0)
    vm, vs = evaluate(base_actor, res_actor, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, "init"
    print(f"Baseline (frozen base + zero residual): {vm:.1f} +/- {vs:.1f}")

    def save_best():
        torch.save(
            {
                "base_actor": base_actor.state_dict(),
                "res_actor": res_actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": td3.critic_target.state_dict(),
                "actor_target": td3.actor_target.state_dict(),
                "actor_opt": td3.actor_opt.state_dict(),
                "critic_opt": td3.critic_opt.state_dict(),
                "delta_scale": DELTA_SCALE,
            },
            BEST_PT,
        )

    total_env_steps = 0

    print(f"\nTD3 Residual Refinement")
    print(f"  csvs/epoch={CSVS_EPOCH}  samples_per_route={SAMPLES_PER_ROUTE}")
    print(f"  actor_lr={ACTOR_LR}  critic_lr={CRITIC_LR}  gamma={GAMMA}  tau={TAU}")
    print(f"  utd_ratio={UTD_RATIO}  mini_bs={MINI_BS}  actor_delay={ACTOR_DELAY}")
    print(
        f"  bc_beta={BC_BETA}  ref_dropout={REF_DROPOUT}  explore_noise={EXPLORE_NOISE}"
    )
    print(f"  max_correction={MAX_CORRECTION}  use_expected={USE_EXPECTED}")
    print(f"  target_noise={TARGET_NOISE}  noise_clip={NOISE_CLIP}")
    print(f"  critic_warmup_steps={CRITIC_WARMUP_STEPS}  buffer_size={BUFFER_SIZE}")
    print(f"  reward_scale={REWARD_SCALE}")
    print(f"  delta_scale={DELTA_SCALE}\n")

    for epoch in range(MAX_EP):
        t0 = time.time()
        res_actor.train()
        critic.train()

        # ── Collect rollouts ──
        n_routes = min(CSVS_EPOCH, len(tr_f))
        batch = random.sample(tr_f, n_routes)
        if SAMPLES_PER_ROUTE > 1:
            batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]

        costs = rollout(
            batch,
            base_actor,
            res_actor,
            mdl_path,
            ort_sess,
            csv_cache,
            replay_buf=replay_buf,
            deterministic=False,
        )
        t_roll = time.time() - t0

        # Count new env steps this epoch
        max_steps = COST_END_IDX - CONTROL_START_IDX
        new_steps = len(batch) * max_steps
        total_env_steps += new_steps

        # ── TD3 updates ──
        t1 = time.time()
        n_updates = max(1, int(new_steps * UTD_RATIO / MINI_BS))
        critic_only = total_env_steps < CRITIC_WARMUP_STEPS

        c_sum, a_sum, bc_sum, n_up = 0.0, 0.0, 0.0, 0
        if replay_buf.size >= MINI_BS:
            for _ in range(n_updates):
                info = td3.update(replay_buf, critic_only=critic_only)
                c_sum += info["critic"]
                a_sum += info["actor"]
                bc_sum += info["bc"]
                n_up += 1
        t_update = time.time() - t1

        # Sample some Q-values for diagnostics
        q_diag = ""
        if replay_buf.size >= 1000:
            with torch.no_grad():
                d_feat, d_base, d_act, d_rew, _, _, _ = replay_buf.sample(
                    min(1000, replay_buf.size)
                )
                q1_d, _ = critic(d_feat, d_act)
                q_diag = f"  Q={q1_d.mean().item():.2f}  R={d_rew.mean().item():.4f}"

        phase = " [critic warmup]" if critic_only else ""
        line = (
            f"E{epoch:3d}  train={np.mean(costs):6.1f}  "
            f"C={c_sum / max(1, n_up):.4f}  A={a_sum / max(1, n_up):.4f}  BC={bc_sum / max(1, n_up):.6f}  "
            f"buf={replay_buf.size:,}  updates={n_up}{q_diag}  "
            f"roll={t_roll:.0f}s  upd={t_update:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            res_actor.eval()
            vm, vs = evaluate(
                base_actor, res_actor, va_f, mdl_path, ort_sess, csv_cache
            )
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " *"
            line += f"  val={vm:6.1f}+/-{vs:4.1f}{mk}"

        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    torch.save(
        {
            "base_actor": base_actor.state_dict(),
            "res_actor": res_actor.state_dict(),
            "delta_scale": DELTA_SCALE,
        },
        EXP_DIR / "final_model.pt",
    )


if __name__ == "__main__":
    train()
