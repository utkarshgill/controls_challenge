import collections
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T, LAT_ACCEL_COST_MULTIPLIER, MAX_ACC_DELTA, STEER_RANGE
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── base architecture (must match exp055) ─────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
MAX_DELTA = 0.5

# ── residual architecture ─────────────────────────────────────
RES_HIDDEN = int(os.getenv("RES_HIDDEN", "128"))
RES_LAYERS = int(os.getenv("RES_LAYERS", "2"))
MAX_CORRECTION = float(os.getenv("MAX_CORRECTION", "0.15"))

# ── scaling (must match exp055) ───────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── critical-phase mask ───────────────────────────────────────
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", "10"))
CRIT_THRESHOLD = float(os.getenv("CRIT_THRESHOLD", "0.10"))
APPLY_EPS = float(os.getenv("APPLY_EPS", "0.02"))

# ── TD3 / off-policy ──────────────────────────────────────────
ACTOR_LR = float(os.getenv("ACTOR_LR", "1e-4"))
CRITIC_LR = float(os.getenv("CRITIC_LR", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.95"))
TAU = float(os.getenv("TAU", "0.005"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "2_000_000"))
MINI_BS = int(os.getenv("MINI_BS", "4096"))
UTD_RATIO = int(os.getenv("UTD_RATIO", "4"))
ACTOR_DELAY = int(os.getenv("ACTOR_DELAY", "2"))
TARGET_NOISE = float(os.getenv("TARGET_NOISE", "0.03"))
NOISE_CLIP = float(os.getenv("NOISE_CLIP", "0.08"))
EXPLORE_NOISE = float(os.getenv("EXPLORE_NOISE", "0.02"))
RESIDUAL_L2 = float(os.getenv("RESIDUAL_L2", "2.0"))
CRITIC_WARMUP_STEPS = int(os.getenv("CRITIC_WARMUP_STEPS", "500000"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "0.001"))
Q_CLAMP = float(os.getenv("Q_CLAMP", "100.0"))

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "500"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "1"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = int(os.getenv("EVAL_N", "100"))
USE_EXPECTED = os.getenv("USE_EXPECTED", "0") == "1"

EXP_DIR = Path(__file__).parent
BASE_PT = Path(
    os.getenv(
        "BASE_MODEL",
        str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"),
    )
)
BEST_PT = EXP_DIR / "best_model.pt"
FINAL_PT = EXP_DIR / "final_model.pt"

# ── obs layout (must match exp055) ────────────────────────────
C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
OBS_DIM = F_A + FUTURE_K


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


def _criticality_batch(future_target, current):
    horizon = min(CRIT_LOOKAHEAD, future_target.shape[1])
    fp = future_target[:, :horizon]
    if horizon <= 1:
        return (fp[:, 0] - current).abs() / S_LAT
    slope = torch.diff(fp, dim=1, prepend=fp[:, :1]) / DEL_T
    mismatch = (fp[:, min(4, horizon - 1)] - current).abs() / S_LAT
    span = (fp[:, -1] - fp[:, 0]).abs() / S_LAT
    peak_slope = slope.abs().amax(dim=1) / max(S_LAT / DEL_T, 1e-6)
    flip = ((fp[:, :-1] * fp[:, 1:]) < 0).any(dim=1).float()
    return mismatch + 0.5 * span + 0.5 * peak_slope + 0.5 * flip


def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale
        return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale


def fill_obs(buf, target, current, roll_la, v_ego, a_ego, h_act, h_lat, hist_head, error_integral, future, step_idx):
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


def apply_crit_mask(correction, criticality):
    return correction * (criticality >= CRIT_THRESHOLD).float()


def _predict_next_lataccel_expected(sim_ref, step_idx, action):
    h = sim_ref._hist_len
    cl = CONTEXT_LENGTH

    if sim_ref._gpu:
        dg = sim_ref.data_gpu
        sim_states = sim_ref.state_history[:, h - cl + 1 : h + 1, :].clone()
        sim_states[:, -1, 0] = dg["roll_lataccel"][:, step_idx]
        sim_states[:, -1, 1] = dg["v_ego"][:, step_idx]
        sim_states[:, -1, 2] = dg["a_ego"][:, step_idx]

        actions = sim_ref.action_history[:, h - cl + 1 : h + 1].clone()
        if step_idx < CONTROL_START_IDX:
            actions[:, -1] = dg["steer_command"][:, step_idx]
        else:
            actions[:, -1] = torch.clamp(action.to(actions.dtype), STEER_RANGE[0], STEER_RANGE[1])

        past_preds = sim_ref.current_lataccel_history[:, h - cl : h]
        _, expected = sim_ref.sim_model.get_current_lataccel(
            sim_states=sim_states,
            actions=actions,
            past_preds=past_preds,
            rng_u=sim_ref._rng_all_gpu[step_idx - cl],
            return_expected=True,
        )
        expected = torch.clamp(
            expected,
            sim_ref.current_lataccel - MAX_ACC_DELTA,
            sim_ref.current_lataccel + MAX_ACC_DELTA,
        )
        if step_idx < CONTROL_START_IDX:
            expected = dg["target_lataccel"][:, step_idx].clone()
        return expected.float()

    d = sim_ref.data
    sim_states = sim_ref.state_history[:, h - cl + 1 : h + 1, :].copy()
    sim_states[:, -1, 0] = d["roll_lataccel"][:, step_idx]
    sim_states[:, -1, 1] = d["v_ego"][:, step_idx]
    sim_states[:, -1, 2] = d["a_ego"][:, step_idx]

    actions = sim_ref.action_history[:, h - cl + 1 : h + 1].copy()
    if step_idx < CONTROL_START_IDX:
        actions[:, -1] = d["steer_command"][:, step_idx]
    else:
        actions[:, -1] = np.clip(np.asarray(action), STEER_RANGE[0], STEER_RANGE[1])

    past_preds = sim_ref.current_lataccel_history[:, h - cl : h]
    _, expected = sim_ref.sim_model.get_current_lataccel(
        sim_states=sim_states,
        actions=actions,
        past_preds=past_preds,
        rng_u=sim_ref._rng_all[step_idx - cl],
        return_expected=True,
    )
    expected = np.clip(
        expected,
        sim_ref.current_lataccel - MAX_ACC_DELTA,
        sim_ref.current_lataccel + MAX_ACC_DELTA,
    )
    if step_idx < CONTROL_START_IDX:
        expected = d["target_lataccel"][:, step_idx].copy()
    return torch.as_tensor(expected, dtype=torch.float32, device=DEV)


def _counterfactual_delta_reward(sim_ref, step_idx, actual_action, base_action):
    current = sim_ref.current_lataccel.float()
    if sim_ref._gpu:
        target = sim_ref.data_gpu["target_lataccel"][:, step_idx].float()
    else:
        target = torch.as_tensor(sim_ref.data["target_lataccel"][:, step_idx], dtype=torch.float32, device=DEV)

    actual_next = _predict_next_lataccel_expected(sim_ref, step_idx, actual_action)
    base_next = _predict_next_lataccel_expected(sim_ref, step_idx, base_action)

    actual_jerk = (actual_next - current) / DEL_T
    base_jerk = (base_next - current) / DEL_T
    actual_cost = (target - actual_next).square() * (100 * LAT_ACCEL_COST_MULTIPLIER) + actual_jerk.square() * 100
    base_cost = (target - base_next).square() * (100 * LAT_ACCEL_COST_MULTIPLIER) + base_jerk.square() * 100
    return (base_cost - actual_cost).unsqueeze(-1) * REWARD_SCALE


class BaseTokenActor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*layers)
        self._features = None
        self.actor[-2].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out

    def forward(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        base_raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return base_raw, self._features.detach()


class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = HIDDEN + 2
        layers = []
        for _ in range(RES_LAYERS):
            layers += [nn.Linear(in_dim, RES_HIDDEN), nn.ReLU()]
            in_dim = RES_HIDDEN
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base_raw, criticality):
        x = torch.cat([features, base_raw, criticality], dim=-1)
        correction = torch.tanh(self.net(x)) * MAX_CORRECTION
        return apply_crit_mask(correction, criticality)


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = HIDDEN + 3
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

    def forward(self, features, base_raw, criticality, correction):
        x = torch.cat([features, base_raw, criticality, correction], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, features, base_raw, criticality, correction):
        x = torch.cat([features, base_raw, criticality, correction], dim=-1)
        return self.q1(x)


class ReplayBuffer:
    def __init__(self, max_size, feat_dim=HIDDEN):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.features = torch.zeros((max_size, feat_dim), dtype=torch.float32, device=DEV)
        self.base_raw = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.criticality = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.correction = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.next_feat = torch.zeros((max_size, feat_dim), dtype=torch.float32, device=DEV)
        self.next_base = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.next_crit = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)

    def add(self, features, base_raw, criticality, correction, reward, next_feat, next_base, next_crit, done):
        n = features.shape[0]
        if n == 0:
            return
        end = self.ptr + n
        if end <= self.max_size:
            sl = slice(self.ptr, end)
            self.features[sl] = features
            self.base_raw[sl] = base_raw
            self.criticality[sl] = criticality
            self.correction[sl] = correction
            self.reward[sl] = reward
            self.next_feat[sl] = next_feat
            self.next_base[sl] = next_base
            self.next_crit[sl] = next_crit
            self.done[sl] = done
        else:
            first = self.max_size - self.ptr
            self.add(
                features[:first],
                base_raw[:first],
                criticality[:first],
                correction[:first],
                reward[:first],
                next_feat[:first],
                next_base[:first],
                next_crit[:first],
                done[:first],
            )
            rest = n - first
            if rest > 0:
                self.add(
                    features[first:],
                    base_raw[first:],
                    criticality[first:],
                    correction[first:],
                    reward[first:],
                    next_feat[first:],
                    next_base[first:],
                    next_crit[first:],
                    done[first:],
                )
            return
        self.ptr = end % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=DEV)
        return (
            self.features[idx],
            self.base_raw[idx],
            self.criticality[idx],
            self.correction[idx],
            self.reward[idx],
            self.next_feat[idx],
            self.next_base[idx],
            self.next_crit[idx],
            self.done[idx],
        )


def rollout(csv_files, base_actor, residual_actor, mdl_path, ort_session, csv_cache, replay_buf=None, deterministic=False):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng)
    if USE_EXPECTED and not deterministic:
        sim.use_expected = True
    n, _ = sim.N, sim.T
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((n, HIST_LEN), dtype=torch.float64, device=DEV)
    h_act32 = torch.zeros((n, HIST_LEN), dtype=torch.float32, device=DEV)
    h_lat = torch.zeros((n, HIST_LEN), dtype=torch.float32, device=DEV)
    h_error = torch.zeros((n, HIST_LEN), dtype=torch.float32, device=DEV)
    err_sum = torch.zeros(n, dtype=torch.float32, device=DEV)
    obs_buf = torch.empty((n, OBS_DIM), dtype=torch.float32, device=DEV)

    if replay_buf is not None and not deterministic:
        step_features = []
        step_base = []
        step_crit = []
        step_corr = []
        step_delta_reward = []

    hist_head = HIST_LEN - 1
    rollout_stats = collections.Counter()

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum
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
            return torch.zeros(n, dtype=h_act.dtype, device=DEV)

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
            base_raw_1 = base_raw.unsqueeze(-1)
            criticality = _criticality_batch(future["target_lataccel"][:, step_idx], cur32).unsqueeze(-1)
            correction = residual_actor(features, base_raw_1, criticality)
            if not deterministic:
                noise = torch.randn_like(correction) * EXPLORE_NOISE
                correction = (correction + noise).clamp(-MAX_CORRECTION, MAX_CORRECTION)
                correction = apply_crit_mask(correction, criticality)
            raw_policy = (base_raw_1 + correction).clamp(-1.0, 1.0)

        delta = raw_policy.squeeze(-1).to(h_act.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        if replay_buf is not None and not deterministic and step_idx < COST_END_IDX:
            step_features.append(features.clone())
            step_base.append(base_raw_1.clone())
            step_crit.append(criticality.clone())
            step_corr.append(correction.clone())
            base_action = (h_act[:, hist_head] + base_raw.to(h_act.dtype) * DELTA_SCALE).clamp(STEER_RANGE[0], STEER_RANGE[1])
            delta_reward = _counterfactual_delta_reward(sim_ref, step_idx, action, base_action)
            step_delta_reward.append(delta_reward)

        crit_mask = (criticality.squeeze(-1) >= CRIT_THRESHOLD).float()
        apply_mask = (correction.squeeze(-1).abs() > APPLY_EPS).float() * crit_mask
        rollout_stats["steps"] += n
        rollout_stats["critical_steps"] += int(crit_mask.sum().item())
        rollout_stats["applied_steps"] += int(apply_mask.sum().item())
        rollout_stats["corr_abs_sum"] += float(correction.abs().sum().item())

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)

    if replay_buf is not None and not deterministic and step_features:
        steps = len(step_features)
        start = CONTROL_START_IDX
        end = start + steps
        feats_all = torch.stack(step_features, dim=1)
        base_all = torch.stack(step_base, dim=1)
        crit_all = torch.stack(step_crit, dim=1)
        corr_all = torch.stack(step_corr, dim=1)
        rew_all = torch.stack(step_delta_reward, dim=1)

        done = torch.zeros((n, steps), dtype=torch.float32, device=DEV)
        done[:, -1] = 1.0

        for t in range(steps - 1):
            replay_buf.add(
                feats_all[:, t],
                base_all[:, t],
                crit_all[:, t],
                corr_all[:, t],
                rew_all[:, t],
                feats_all[:, t + 1],
                base_all[:, t + 1],
                crit_all[:, t + 1],
                done[:, t : t + 1],
            )
        replay_buf.add(
            feats_all[:, -1],
            base_all[:, -1],
            crit_all[:, -1],
            corr_all[:, -1],
            rew_all[:, -1],
            feats_all[:, -1],
            base_all[:, -1],
            crit_all[:, -1],
            done[:, -1:],
        )

    diag = {
        "apply_frac": rollout_stats["applied_steps"] / max(1, rollout_stats["steps"]),
        "crit_frac": rollout_stats["critical_steps"] / max(1, rollout_stats["steps"]),
        "corr_mag": rollout_stats["corr_abs_sum"] / max(1, rollout_stats["steps"]),
    }
    return np.asarray(costs["total_cost"], dtype=np.float64), diag


class DeltaQTD3:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.actor_target = ResidualActor().to(DEV)
        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target = TwinCritic().to(DEV)
        self.critic_target.load_state_dict(critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.total_updates = 0

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(TAU * sp.data + (1.0 - TAU) * tp.data)

    def update(self, replay_buf, batch_size=MINI_BS, critic_only=False):
        feat, base_raw, criticality, correction, reward, next_feat, next_base, next_crit, done = replay_buf.sample(batch_size)

        with torch.no_grad():
            next_corr = self.actor_target(next_feat, next_base, next_crit)
            noise = (torch.randn_like(next_corr) * TARGET_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_corr = (next_corr + noise).clamp(-MAX_CORRECTION, MAX_CORRECTION)
            next_corr = apply_crit_mask(next_corr, next_crit)
            tq1, tq2 = self.critic_target(next_feat, next_base, next_crit, next_corr)
            target_q = reward + GAMMA * (1.0 - done) * torch.min(tq1, tq2)
            target_q = target_q.clamp(-Q_CLAMP, Q_CLAMP)

        q1, q2 = self.critic(feat, base_raw, criticality, correction)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        dq_val = 0.0
        l2_val = 0.0
        q_corr_val = 0.0
        q_zero_val = 0.0

        self.total_updates += 1
        if not critic_only and self.total_updates % ACTOR_DELAY == 0:
            corr = self.actor(feat, base_raw, criticality)
            q_corr = self.critic.q1_only(feat, base_raw, criticality, corr)
            with torch.no_grad():
                zero_corr = torch.zeros_like(corr)
                q_zero = self.critic.q1_only(feat, base_raw, criticality, zero_corr)
            delta_q = q_corr - q_zero
            residual_l2 = corr.pow(2).mean()
            actor_loss = -delta_q.mean() + RESIDUAL_L2 * residual_l2

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

            actor_loss_val = actor_loss.item()
            dq_val = delta_q.mean().item()
            l2_val = residual_l2.item()
            q_corr_val = q_corr.mean().item()
            q_zero_val = q_zero.mean().item()

        return {
            "critic": critic_loss.item(),
            "actor": actor_loss_val,
            "dq": dq_val,
            "l2": l2_val,
            "q_corr": q_corr_val,
            "q_zero": q_zero_val,
        }


def evaluate(base_actor, residual_actor, files, mdl_path, ort_session, csv_cache):
    base_actor.eval()
    residual_actor.eval()
    costs, diag = rollout(
        files,
        base_actor,
        residual_actor,
        mdl_path,
        ort_session,
        csv_cache,
        replay_buf=None,
        deterministic=True,
    )
    return float(costs.mean()), float(costs.std()), diag


def load_base_actor():
    base_actor = BaseTokenActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    base_actor.load_state_dict(actor_state)
    base_actor.eval()
    for p in base_actor.parameters():
        p.requires_grad_(False)
    ds_ckpt = ckpt.get("delta_scale", None)
    return base_actor, ds_ckpt


def save_ckpt(path, base_actor, residual_actor, critic, td3):
    torch.save(
        {
            "base_actor": base_actor.state_dict(),
            "res_actor": residual_actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_target": td3.actor_target.state_dict(),
            "critic_target": td3.critic_target.state_dict(),
            "actor_opt": td3.actor_opt.state_dict(),
            "critic_opt": td3.critic_opt.state_dict(),
            "delta_scale": DELTA_SCALE,
            "max_correction": MAX_CORRECTION,
            "crit_threshold": CRIT_THRESHOLD,
            "apply_eps": APPLY_EPS,
        },
        path,
    )


def train():
    global DELTA_SCALE
    base_actor, ds_ckpt = load_base_actor()
    if ds_ckpt is not None:
        DELTA_SCALE = float(ds_ckpt)

    residual_actor = ResidualActor().to(DEV)
    critic = TwinCritic().to(DEV)
    td3 = DeltaQTD3(residual_actor, critic)

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    tr_f = all_csv
    va_f = all_csv[: min(EVAL_N, len(all_csv))]
    csv_cache = CSVCache([str(f) for f in all_csv])
    replay_buf = ReplayBuffer(BUFFER_SIZE)

    vm, vs, vdiag = evaluate(base_actor, residual_actor, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, "init"
    print(f"Baseline (frozen exp055 + zero residual): {vm:.1f} ± {vs:.1f}  apply={vdiag['apply_frac']:.3f} crit={vdiag['crit_frac']:.3f}")

    total_env_steps = 0
    print("\nDelta-Q TD3")
    print(f"  csvs/epoch={CSVS_EPOCH}  samples_per_route={SAMPLES_PER_ROUTE}  use_expected={USE_EXPECTED}")
    print(f"  actor_lr={ACTOR_LR}  critic_lr={CRITIC_LR}  gamma={GAMMA}  tau={TAU}")
    print(f"  utd_ratio={UTD_RATIO}  mini_bs={MINI_BS}  actor_delay={ACTOR_DELAY}")
    print(f"  residual_l2={RESIDUAL_L2}  explore_noise={EXPLORE_NOISE}  target_noise={TARGET_NOISE}")
    print(f"  max_correction={MAX_CORRECTION}  crit_threshold={CRIT_THRESHOLD}  delta_scale={DELTA_SCALE:.4f}")
    print(f"  critic_warmup_steps={CRITIC_WARMUP_STEPS}  reward_scale={REWARD_SCALE}\n")

    for epoch in range(MAX_EP):
        t0 = time.time()
        residual_actor.train()
        critic.train()

        n_routes = min(CSVS_EPOCH, len(tr_f))
        batch = random.sample(tr_f, n_routes)
        if SAMPLES_PER_ROUTE > 1:
            batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]

        costs, roll_diag = rollout(
            batch,
            base_actor,
            residual_actor,
            mdl_path,
            ort_sess,
            csv_cache,
            replay_buf=replay_buf,
            deterministic=False,
        )
        t_roll = time.time() - t0

        max_steps = COST_END_IDX - CONTROL_START_IDX
        new_steps = len(batch) * max_steps
        total_env_steps += new_steps
        critic_only = total_env_steps < CRITIC_WARMUP_STEPS
        n_updates = max(1, int(new_steps * UTD_RATIO / MINI_BS))

        t1 = time.time()
        c_sum = a_sum = dq_sum = l2_sum = q_corr_sum = q_zero_sum = 0.0
        n_up = 0
        if replay_buf.size >= MINI_BS:
            for _ in range(n_updates):
                info = td3.update(replay_buf, critic_only=critic_only)
                c_sum += info["critic"]
                a_sum += info["actor"]
                dq_sum += info["dq"]
                l2_sum += info["l2"]
                q_corr_sum += info["q_corr"]
                q_zero_sum += info["q_zero"]
                n_up += 1
        t_update = time.time() - t1

        phase = " [critic warmup]" if critic_only else ""
        line = (
            f"E{epoch:3d}  train={np.mean(costs):6.1f}  "
            f"C={c_sum / max(1, n_up):.4f}  A={a_sum / max(1, n_up):+.4f}  "
            f"dQ={dq_sum / max(1, n_up):+.4f}  L2={l2_sum / max(1, n_up):.6f}  "
            f"Qc={q_corr_sum / max(1, n_up):+.3f}  Q0={q_zero_sum / max(1, n_up):+.3f}  "
            f"apply={roll_diag['apply_frac']:.3f}  crit={roll_diag['crit_frac']:.3f}  "
            f"|corr|={roll_diag['corr_mag']:.4f}  buf={replay_buf.size:,}  "
            f"roll={t_roll:.0f}s  upd={t_update:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            residual_actor.eval()
            vm, vs, vdiag = evaluate(base_actor, residual_actor, va_f, mdl_path, ort_sess, csv_cache)
            marker = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_ckpt(BEST_PT, base_actor, residual_actor, critic, td3)
                marker = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}  v_apply={vdiag['apply_frac']:.3f}  v_crit={vdiag['crit_frac']:.3f}{marker}"

        print(line)

    save_ckpt(FINAL_PT, base_actor, residual_actor, critic, td3)
    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
