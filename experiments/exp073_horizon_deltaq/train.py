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

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T, LAT_ACCEL_COST_MULTIPLIER, STEER_RANGE
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

# ── horizon label collection ──────────────────────────────────
STEP_STRIDE = int(os.getenv("STEP_STRIDE", "5"))
TOPK_PER_ROUTE = int(os.getenv("TOPK_PER_ROUTE", "4"))
RAND_PER_ROUTE = int(os.getenv("RAND_PER_ROUTE", "4"))
HORIZON_H = int(os.getenv("HORIZON_H", "10"))
BRANCH_USE_EXPECTED = os.getenv("BRANCH_USE_EXPECTED", "1") == "1"
HORIZON_GAMMA = float(os.getenv("HORIZON_GAMMA", "0.95"))
ACTION_PROBES = int(os.getenv("ACTION_PROBES", "2"))
PROBE_CORR_STD = float(os.getenv("PROBE_CORR_STD", "0.03"))

# ── off-policy critic / actor ─────────────────────────────────
ACTOR_LR = float(os.getenv("ACTOR_LR", "1e-4"))
CRITIC_LR = float(os.getenv("CRITIC_LR", "3e-4"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "500000"))
MINI_BS = int(os.getenv("MINI_BS", "4096"))
UTD_RATIO = float(os.getenv("UTD_RATIO", "4.0"))
ACTOR_DELAY = int(os.getenv("ACTOR_DELAY", "2"))
EXPLORE_NOISE = float(os.getenv("EXPLORE_NOISE", "0.03"))
RESIDUAL_L2 = float(os.getenv("RESIDUAL_L2", "1.0"))
ZERO_ANCHOR_COEF = float(os.getenv("ZERO_ANCHOR_COEF", "1.0"))
ACTOR_START_SAMPLES = int(os.getenv("ACTOR_START_SAMPLES", "20000"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "0.001"))

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "128"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "1"))
MAX_EP = int(os.getenv("EPOCHS", "2000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = int(os.getenv("EVAL_N", "100"))
ROLLOUT_USE_EXPECTED = os.getenv("ROLLOUT_USE_EXPECTED", "0") == "1"

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
        self.target_deltaq = torch.zeros((max_size, 1), dtype=torch.float32, device=DEV)

    def add(self, features, base_raw, criticality, correction, target_deltaq):
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
            self.target_deltaq[sl] = target_deltaq
        else:
            first = self.max_size - self.ptr
            self.add(features[:first], base_raw[:first], criticality[:first], correction[:first], target_deltaq[:first])
            rest = n - first
            if rest > 0:
                self.add(features[first:], base_raw[first:], criticality[first:], correction[first:], target_deltaq[first:])
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
            self.target_deltaq[idx],
        )


def build_obs_from_sim(sim_ref, future, step_idx):
    dg = sim_ref.data_gpu
    target = dg["target_lataccel"][:, step_idx].float()
    current = sim_ref.current_lataccel.float()
    roll_la = dg["roll_lataccel"][:, step_idx].float()
    v_ego = dg["v_ego"][:, step_idx].float()
    a_ego = dg["a_ego"][:, step_idx].float()
    h_act = sim_ref.action_history[:, step_idx - HIST_LEN : step_idx].float()
    h_lat = sim_ref.current_lataccel_history[:, step_idx - HIST_LEN : step_idx].float()
    h_tgt = dg["target_lataccel"][:, step_idx - HIST_LEN : step_idx].float()

    error_integral = (h_tgt - h_lat).mean(dim=1) * DEL_T
    prev_act = h_act[:, -1]
    prev_act2 = h_act[:, -2]
    prev_lat = h_lat[:, -1]
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0

    obs = torch.empty((sim_ref.N, OBS_DIM), dtype=torch.float32, device=DEV)
    obs[:, 0] = target / S_LAT
    obs[:, 1] = current / S_LAT
    obs[:, 2] = (target - current) / S_LAT
    obs[:, 3] = k_tgt / S_CURV
    obs[:, 4] = k_cur / S_CURV
    obs[:, 5] = (k_tgt - k_cur) / S_CURV
    obs[:, 6] = v_ego / S_VEGO
    obs[:, 7] = a_ego / S_AEGO
    obs[:, 8] = roll_la / S_ROLL
    obs[:, 9] = prev_act / S_STEER
    obs[:, 10] = error_integral / S_LAT
    obs[:, 11] = (fp0 - target) / DEL_T / S_LAT
    obs[:, 12] = (current - prev_lat) / DEL_T / S_LAT
    obs[:, 13] = (prev_act - prev_act2) / DEL_T / S_STEER
    obs[:, 14] = fric
    obs[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    obs[:, C:H1] = h_act / S_STEER
    obs[:, H1:H2] = h_lat / S_LAT
    obs[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    obs[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    obs[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    obs[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    obs.clamp_(-5.0, 5.0)
    return obs


def rollout_policy(csv_files, base_actor, residual_actor, mdl_path, ort_session, csv_cache, deterministic=False):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng)
    if ROLLOUT_USE_EXPECTED and not deterministic:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    n, t_total = sim.N, sim.T
    raw_hist = torch.zeros((n, t_total), dtype=torch.float32, device=DEV)
    base_hist = torch.zeros((n, t_total), dtype=torch.float32, device=DEV)
    corr_hist = torch.zeros((n, t_total), dtype=torch.float32, device=DEV)
    crit_hist = torch.zeros((n, t_total), dtype=torch.float32, device=DEV)
    feat_hist = torch.zeros((n, t_total, HIDDEN), dtype=torch.float32, device=DEV)

    stats = collections.Counter()

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(sim_ref.N, dtype=torch.float64, device=DEV)

        with torch.no_grad():
            obs = build_obs_from_sim(sim_ref, future, step_idx)
            base_raw, features = base_actor(obs)
            crit = _criticality_batch(future["target_lataccel"][:, step_idx], sim_ref.current_lataccel.float()).unsqueeze(-1)
            corr = residual_actor(features, base_raw.unsqueeze(-1), crit)
            if not deterministic:
                noise = torch.randn_like(corr) * EXPLORE_NOISE
                corr = (corr + noise).clamp(-MAX_CORRECTION, MAX_CORRECTION)
                corr = apply_crit_mask(corr, crit)
            raw = (base_raw.unsqueeze(-1) + corr).clamp(-1.0, 1.0).squeeze(-1)

        raw_hist[:, step_idx] = raw
        base_hist[:, step_idx] = base_raw
        corr_hist[:, step_idx] = corr.squeeze(-1)
        crit_hist[:, step_idx] = crit.squeeze(-1)
        feat_hist[:, step_idx] = features

        crit_mask = (crit.squeeze(-1) >= CRIT_THRESHOLD).float()
        apply_mask = (corr.squeeze(-1).abs() > APPLY_EPS).float() * crit_mask
        stats["steps"] += sim_ref.N
        stats["critical_steps"] += int(crit_mask.sum().item())
        stats["applied_steps"] += int(apply_mask.sum().item())
        stats["corr_abs_sum"] += float(corr.abs().sum().item())

        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)
    diag = {
        "apply_frac": stats["applied_steps"] / max(1, stats["steps"]),
        "crit_frac": stats["critical_steps"] / max(1, stats["steps"]),
        "corr_mag": stats["corr_abs_sum"] / max(1, stats["steps"]),
    }
    return sim, future, np.asarray(costs["total_cost"], dtype=np.float64), raw_hist, base_hist, corr_hist, crit_hist, feat_hist, diag


def select_windows(policy_sim, crit_hist, csv_files):
    steps = list(range(CONTROL_START_IDX, COST_END_IDX, STEP_STRIDE))
    rows = []
    for route_idx, csv_file in enumerate(csv_files):
        crit = crit_hist[route_idx, steps]
        topk = min(TOPK_PER_ROUTE, len(steps))
        _, top_idx = torch.topk(crit, k=topk, largest=True, sorted=True)
        chosen = set(int(i) for i in top_idx.tolist())
        if RAND_PER_ROUTE > 0:
            remaining = [i for i in range(len(steps)) if i not in chosen]
            if remaining:
                chosen.update(random.sample(remaining, min(RAND_PER_ROUTE, len(remaining))))
        for local_idx in sorted(chosen):
            step_idx = steps[local_idx]
            rows.append(
                {
                    "csv_file": str(csv_file),
                    "route_idx": route_idx,
                    "step_idx": int(step_idx),
                    "criticality": float(crit_hist[route_idx, step_idx].item()),
                }
            )
    rows.sort(key=lambda r: (r["route_idx"], r["step_idx"]))
    return rows


def branch_horizon_delta_return(csv_file, csv_cache, base_actor, residual_actor, mdl_path, ort_sess, policy_sim, route_idx, step_idx, first_action_actual, first_action_base):
    tiled = [csv_file, csv_file]
    data, rng = csv_cache.slice(tiled)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    if BRANCH_USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    h = step_idx
    sim._hist_len = h
    sim.action_history[:, :h] = policy_sim.action_history[route_idx : route_idx + 1, :h].repeat(2, 1)
    sim.current_lataccel_history[:, :h] = policy_sim.current_lataccel_history[route_idx : route_idx + 1, :h].repeat(2, 1)
    sim.current_lataccel = sim.current_lataccel_history[:, h - 1].clone()
    sim.state_history[:, :h, 0] = sim.data_gpu["roll_lataccel"][:, :h]
    sim.state_history[:, :h, 1] = sim.data_gpu["v_ego"][:, :h]
    sim.state_history[:, :h, 2] = sim.data_gpu["a_ego"][:, :h]

    total_cost = torch.zeros(2, dtype=torch.float32, device=DEV)
    end_step = min(step_idx + HORIZON_H, COST_END_IDX)

    for offset, t in enumerate(range(step_idx, end_step)):
        if offset == 0:
            actions = torch.tensor([first_action_actual, first_action_base], dtype=torch.float64, device=DEV)
        else:
            with torch.no_grad():
                obs = build_obs_from_sim(sim, future, t)
                base_raw, features = base_actor(obs)
                crit = _criticality_batch(future["target_lataccel"][:, t], sim.current_lataccel.float()).unsqueeze(-1)
                corr = residual_actor(features, base_raw.unsqueeze(-1), crit)
                raw_resid = (base_raw.unsqueeze(-1) + corr).clamp(-1.0, 1.0).squeeze(-1)
                raw_base = base_raw
                prev_action = sim.action_history[:, sim._hist_len - 1]
                action_resid = (prev_action + raw_resid.to(prev_action.dtype) * DELTA_SCALE).clamp(STEER_RANGE[0], STEER_RANGE[1])
                action_base = (prev_action + raw_base.to(prev_action.dtype) * DELTA_SCALE).clamp(STEER_RANGE[0], STEER_RANGE[1])
                actions = torch.stack([action_resid[0], action_base[1]])

        sim.step(t, actions)
        cur = sim.current_lataccel.float()
        tgt = sim.data_gpu["target_lataccel"][:, t].float()
        prev_pred = sim.current_lataccel_history[:, t - 1].float()
        jerk = (cur - prev_pred) / DEL_T
        step_cost = (tgt - cur).square() * (100 * LAT_ACCEL_COST_MULTIPLIER) + jerk.square() * 100
        total_cost += (HORIZON_GAMMA ** offset) * step_cost

    return float((total_cost[1] - total_cost[0]).item() * REWARD_SCALE)


def collect_horizon_targets(routes, split_name, base_actor, residual_actor, mdl_path, ort_sess, csv_cache):
    policy_sim, future, rollout_costs, raw_hist, base_hist, corr_hist, crit_hist, feat_hist, rollout_diag = rollout_policy(
        routes, base_actor, residual_actor, mdl_path, ort_sess, csv_cache, deterministic=False
    )
    windows = select_windows(policy_sim, crit_hist, routes)
    rows = []
    for i, win in enumerate(windows, start=1):
        route_idx = win["route_idx"]
        step_idx = win["step_idx"]
        prev_action = float(policy_sim.action_history[route_idx, step_idx - 1].item())
        base_raw = float(base_hist[route_idx, step_idx].item())
        exec_corr = float(corr_hist[route_idx, step_idx].item())
        base_action = float(np.clip(prev_action + base_raw * DELTA_SCALE, STEER_RANGE[0], STEER_RANGE[1]))
        features = feat_hist[route_idx, step_idx].detach().clone()
        crit = float(win["criticality"])

        candidates = [0.0, exec_corr]
        if crit >= CRIT_THRESHOLD:
            for k in range(1, ACTION_PROBES + 1):
                mag = PROBE_CORR_STD * float(k)
                candidates.extend([mag, -mag])
            if abs(exec_corr) > 1e-6:
                candidates.append(-exec_corr)

        # Deduplicate near-identical probes after clipping to the valid correction range.
        dedup = []
        seen = set()
        for c in candidates:
            cc = float(np.clip(c, -MAX_CORRECTION, MAX_CORRECTION))
            key = round(cc, 5)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(cc)

        best_delta_q = -1e9
        best_corr = 0.0
        exec_delta_q = 0.0
        for cand_corr in dedup:
            if abs(cand_corr) < 1e-8:
                delta_q = 0.0
            else:
                cand_raw = float(np.clip(base_raw + cand_corr, -1.0, 1.0))
                cand_action = float(np.clip(prev_action + cand_raw * DELTA_SCALE, STEER_RANGE[0], STEER_RANGE[1]))
                delta_q = branch_horizon_delta_return(
                    csv_file=win["csv_file"],
                    csv_cache=csv_cache,
                    base_actor=base_actor,
                    residual_actor=residual_actor,
                    mdl_path=mdl_path,
                    ort_sess=ort_sess,
                    policy_sim=policy_sim,
                    route_idx=route_idx,
                    step_idx=step_idx,
                    first_action_actual=cand_action,
                    first_action_base=base_action,
                )

            rows.append(
                {
                    "split": split_name,
                    "csv_file": win["csv_file"],
                    "route_idx": route_idx,
                    "step_idx": step_idx,
                    "criticality": crit,
                    "features": features,
                    "base_raw": torch.tensor([base_raw], dtype=torch.float32, device=DEV),
                    "correction": torch.tensor([cand_corr], dtype=torch.float32, device=DEV),
                    "delta_q": delta_q,
                }
            )
            if abs(cand_corr - exec_corr) < 1e-6:
                exec_delta_q = delta_q
            if delta_q > best_delta_q:
                best_delta_q = delta_q
                best_corr = cand_corr

        print(
            f"[{split_name} {i:4d}/{len(windows):4d}] route={route_idx:3d} step={step_idx:3d}"
            f" crit={crit:.3f} exec={exec_corr:+7.4f} ΔQ_exec={exec_delta_q:+9.3f}"
            f" best={best_corr:+7.4f} ΔQ_best={best_delta_q:+9.3f}"
        )
    return rows, rollout_costs, rollout_diag


class HorizonDeltaQ:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.total_updates = 0

    def update(self, replay_buf, batch_size=MINI_BS, critic_only=False):
        feat, base_raw, criticality, correction, target_deltaq = replay_buf.sample(batch_size)
        q1, q2 = self.critic(feat, base_raw, criticality, correction)
        zero = torch.zeros_like(correction)
        q1_zero, q2_zero = self.critic(feat, base_raw, criticality, zero)
        critic_loss = (
            F.mse_loss(q1, target_deltaq)
            + F.mse_loss(q2, target_deltaq)
            + ZERO_ANCHOR_COEF * (q1_zero.pow(2).mean() + q2_zero.pow(2).mean())
        )

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

        if (not critic_only) and self.total_updates % ACTOR_DELAY == 0:
            corr = self.actor(feat, base_raw, criticality)
            q_corr = self.critic.q1_only(feat, base_raw, criticality, corr)
            with torch.no_grad():
                q_zero = self.critic.q1_only(feat, base_raw, criticality, torch.zeros_like(corr))
            residual_l2 = corr.pow(2).mean()
            actor_loss = -q_corr.mean() + RESIDUAL_L2 * residual_l2

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            actor_loss_val = actor_loss.item()
            dq_val = (q_corr - q_zero).mean().item()
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
    _, _, costs, _, _, _, crit_hist, _, diag = rollout_policy(
        files, base_actor, residual_actor, mdl_path, ort_session, csv_cache, deterministic=True
    )
    return float(costs.mean()), float(costs.std()), {
        "apply_frac": diag["apply_frac"],
        "crit_frac": diag["crit_frac"],
        "corr_mag": diag["corr_mag"],
        "crit_mean": float(crit_hist[:, CONTROL_START_IDX:COST_END_IDX].mean().item()),
    }


def load_base_actor():
    base_actor = BaseTokenActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    base_actor.load_state_dict(actor_state)
    base_actor.eval()
    for p in base_actor.parameters():
        p.requires_grad_(False)
    return base_actor, ckpt.get("delta_scale", None)


def save_ckpt(path, base_actor, residual_actor, critic):
    torch.save(
        {
            "base_actor": base_actor.state_dict(),
            "res_actor": residual_actor.state_dict(),
            "critic": critic.state_dict(),
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
    learner = HorizonDeltaQ(residual_actor, critic)

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    tr_f = all_csv
    va_f = all_csv[: min(EVAL_N, len(all_csv))]
    csv_cache = CSVCache([str(f) for f in all_csv])
    replay_buf = ReplayBuffer(BUFFER_SIZE)

    vm, vs, vdiag = evaluate(base_actor, residual_actor, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, "init"
    print(
        f"Baseline (frozen exp055 + zero residual): {vm:.1f} ± {vs:.1f}"
        f"  apply={vdiag['apply_frac']:.3f}  crit={vdiag['crit_frac']:.3f}"
    )

    print("\nHorizon Delta-Q")
    print(f"  csvs/epoch={CSVS_EPOCH}  samples_per_route={SAMPLES_PER_ROUTE}")
    print(f"  step_stride={STEP_STRIDE}  topk={TOPK_PER_ROUTE}  rand={RAND_PER_ROUTE}  H={HORIZON_H}")
    print(f"  actor_lr={ACTOR_LR}  critic_lr={CRITIC_LR}  mini_bs={MINI_BS}  utd_ratio={UTD_RATIO}")
    print(f"  residual_l2={RESIDUAL_L2}  zero_anchor={ZERO_ANCHOR_COEF}  explore_noise={EXPLORE_NOISE}")
    print(f"  max_correction={MAX_CORRECTION}  crit_threshold={CRIT_THRESHOLD}  delta_scale={DELTA_SCALE:.4f}")
    print(f"  branch_use_expected={BRANCH_USE_EXPECTED}  horizon_gamma={HORIZON_GAMMA}\n")

    total_samples = 0
    for epoch in range(MAX_EP):
        t0 = time.time()
        residual_actor.train()
        critic.train()

        n_routes = min(CSVS_EPOCH, len(tr_f))
        batch = random.sample(tr_f, n_routes)
        if SAMPLES_PER_ROUTE > 1:
            batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]

        rows, rollout_costs, rollout_diag = collect_horizon_targets(
            batch, "train", base_actor, residual_actor, mdl_path, ort_sess, csv_cache
        )
        total_samples += len(rows)
        if rows:
            feats = torch.stack([r["features"] for r in rows], dim=0)
            base_raw = torch.stack([r["base_raw"] for r in rows], dim=0)
            criticality = torch.tensor([[r["criticality"]] for r in rows], dtype=torch.float32, device=DEV)
            correction = torch.stack([r["correction"] for r in rows], dim=0)
            target_deltaq = torch.tensor([[r["delta_q"]] for r in rows], dtype=torch.float32, device=DEV)
            replay_buf.add(feats, base_raw, criticality, correction, target_deltaq)
        t_collect = time.time() - t0

        n_updates = max(1, int(len(rows) * UTD_RATIO / MINI_BS)) if rows else 0
        critic_only = replay_buf.size < ACTOR_START_SAMPLES
        t1 = time.time()
        c_sum = a_sum = dq_sum = l2_sum = q_corr_sum = q_zero_sum = 0.0
        n_up = 0
        if replay_buf.size >= MINI_BS:
            for _ in range(n_updates):
                info = learner.update(replay_buf, critic_only=critic_only)
                c_sum += info["critic"]
                a_sum += info["actor"]
                dq_sum += info["dq"]
                l2_sum += info["l2"]
                q_corr_sum += info["q_corr"]
                q_zero_sum += info["q_zero"]
                n_up += 1
        t_update = time.time() - t1

        target_mean = float(np.mean([r["delta_q"] for r in rows])) if rows else 0.0
        target_pos = float(np.mean([r["delta_q"] > 0 for r in rows])) if rows else 0.0
        phase = " [critic only]" if critic_only else ""
        line = (
            f"E{epoch:3d}  train={np.mean(rollout_costs):6.1f}  "
            f"C={c_sum / max(1, n_up):.4f}  A={a_sum / max(1, n_up):+.4f}  "
            f"dQ={dq_sum / max(1, n_up):+.4f}  L2={l2_sum / max(1, n_up):.6f}  "
            f"Qc={q_corr_sum / max(1, n_up):+.3f}  Q0={q_zero_sum / max(1, n_up):+.3f}  "
            f"apply={rollout_diag['apply_frac']:.3f}  crit={rollout_diag['crit_frac']:.3f}  |corr|={rollout_diag['corr_mag']:.4f}  "
            f"ΔQ_H={target_mean:+.4f}  pos={target_pos:.3f}  buf={replay_buf.size:,}  "
            f"collect={t_collect:.0f}s  upd={t_update:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            residual_actor.eval()
            vm, vs, vdiag = evaluate(base_actor, residual_actor, va_f, mdl_path, ort_sess, csv_cache)
            marker = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_ckpt(BEST_PT, base_actor, residual_actor, critic)
                marker = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}  v_apply={vdiag['apply_frac']:.3f}  v_crit={vdiag['crit_frac']:.3f}{marker}"

        print(line)

    save_ckpt(FINAL_PT, base_actor, residual_actor, critic)
    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
