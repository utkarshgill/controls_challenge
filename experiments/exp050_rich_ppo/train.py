"""
exp050 — Physics-Aligned PPO
==============================
381-dim obs = 11 core + 6×20 history + 4×50 future + 50 future_κ.
Core: target, current, error, κ_tgt, κ_cur, Δκ, v, a, roll, prev_act, innovation.
Beta policy. Kalman innovation. Delta actions. BC pretrain. ReLU. Huber VF loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, os, random, json, subprocess, tempfile, multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, BaseController,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, FUTURE_PLAN_STEPS,
    STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, ACC_G,
    VOCAB_SIZE, LATACCEL_RANGE, MAX_ACC_DELTA,
    State, FuturePlan,
)
from tinyphysics_batched import (
    BatchedSimulator, pool_init, get_pool_cache,
    chunk_list, run_parallel_chunked,
)
import pandas as pd
from tqdm.contrib.concurrent import process_map

# ── Config ────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

HIST_LEN    = 20
STATE_DIM   = 256        # 16 core + 40 history (2×20) + 200 future (4×50)
HIDDEN      = 256
A_LAYERS    = 4
C_LAYERS    = 4
FUTURE_K    = 50

DELTA_SCALE = 0.25
MAX_DELTA   = 0.5

PI_LR       = 3e-4
VF_LR       = 3e-4
GAMMA       = 0.95
LAMDA       = 0.95
K_EPOCHS    = 3
EPS_CLIP    = 0.1
VF_COEF     = 1.0
ENT_COEF    = 0.01
MINI_BS     = 5000

CSVS_EPOCH  = int(os.getenv('CSVS',    '500'))
MAX_EP      = int(os.getenv('EPOCHS',   '200'))
EVAL_EVERY  = 5
EVAL_N      = 100
WORKERS     = int(os.getenv('WORKERS',  '10'))
BC_WORKERS  = int(os.getenv('BC_WORKERS', '10'))
RESUME      = os.getenv('RESUME', '0') == '1'
BATCHED     = os.getenv('BATCHED', '1') == '1'

SCORED_N    = COST_END_IDX - CONTROL_START_IDX     # 400

INIT_SIGMA  = 0.20   # reference for diagnostics only
BC_EPOCHS     = int(os.getenv('BC_EPOCHS', '40'))
BC_LR         = float(os.getenv('BC_LR', '3e-4'))
BC_BS         = int(os.getenv('BC_BS', '8192'))
BC_GRAD_CLIP  = float(os.getenv('BC_GRAD_CLIP', '1.0'))
CRITIC_WARMUP = 3    # epochs: critic-only before actor unfreezes

# ── Scaling ───────────────────────────────────────────────────
S_LAT   = 5.0    # lataccel [-5, 5]
S_STEER = 2.0    # steer    [-2, 2]
S_VEGO  = 40.0   # v_ego    [0, ~40] m/s
S_AEGO  = 4.0    # a_ego    [-4, 4] m/s²
S_ROLL  = 2.0    # roll_lataccel [-1.7, 1.7]
S_CURV  = 0.02   # curvature: ~5 / 20² = 0.0125 typical max


EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
BEST_PT = EXP_DIR / 'best_model.pt'

# ── Distributed config ───────────────────────────────────────
# Set REMOTE=1 to enable distributed rollouts. Without it, pure local.
REMOTE_HOST = os.getenv('REMOTE_HOST', 'hawking@169.254.35.179')
REMOTE_DIR  = os.getenv('REMOTE_DIR',
    '~/Desktop/stuff/controls_challenge')
REMOTE_PY   = os.getenv('REMOTE_PY',
    '~/Desktop/stuff/controls_challenge/.venv/bin/python')
REMOTE_WORKERS = int(os.getenv('REMOTE_WORKERS', '10'))
SSH_KEY     = os.path.expanduser('~/.ssh/id_ed25519')
USE_REMOTE  = os.getenv('REMOTE', '0') == '1'  # master switch
REMOTE_FRAC = float(os.getenv('REMOTE_FRAC', '0.4'))  # fraction of CSVs sent to remote


# ── Innovation helpers (Kalman-style expected prediction) ─────

def predict_expected_la(model, h_act, h_states, h_lat):
    """Expected (mean) lataccel from the ONNX model — deterministic, no sampling.

    Args:
        model: TinyPhysicsModel instance (has .ort_session, .tokenizer, .softmax)
        h_act:    list of 20 floats — recent steer actions
        h_states: list of 20 (roll_la, v_ego, a_ego) tuples
        h_lat:    list of 20 floats — recent current_lataccel values
    Returns:
        float — expected lataccel (weighted mean over 1024 bins)
    """
    tokenized = model.tokenizer.encode(h_lat)                     # (20,)
    raw_states = [list(s) for s in h_states]
    states = np.column_stack([h_act, raw_states])                 # (20, 4)
    input_data = {
        'states': np.expand_dims(states, 0).astype(np.float32),  # (1, 20, 4)
        'tokens': np.expand_dims(tokenized, 0).astype(np.int64), # (1, 20)
    }
    logits = model.ort_session.run(None, input_data)[0]           # (1, 20, 1024)
    probs = model.softmax(logits / 0.8, axis=-1)[0, -1, :]       # (1024,)
    return float(np.sum(probs * model.tokenizer.bins))


def predict_expected_la_batch(sim_model, h_act, h_roll, h_v, h_a, h_lat):
    """Batched expected (mean) lataccel — no sampling, deterministic.

    Args:
        sim_model: BatchedPhysicsModel instance
        h_act:  (N, 20) float64 — steer actions
        h_roll: (N, 20) float32 — roll_lataccel
        h_v:    (N, 20) float32 — v_ego
        h_a:    (N, 20) float32 — a_ego
        h_lat:  (N, 20) float32 — current_lataccel values
    Returns:
        (N,) float64 — expected lataccel per episode
    """
    sim_states = np.stack([h_roll, h_v, h_a], axis=-1)            # (N, 20, 3)
    tokenized = sim_model.tokenizer.encode(h_lat)                 # (N, 20)
    states = np.concatenate([h_act[:, :, None], sim_states], axis=-1)  # (N, 20, 4)
    input_data = {
        'states': states.astype(np.float32),
        'tokens': tokenized.astype(np.int64),
    }
    logits = sim_model.ort_session.run(None, input_data)[0]       # (N, 20, 1024)
    probs = sim_model.softmax(logits / 0.8, axis=-1)[:, -1, :]   # (N, 1024)
    return np.sum(probs * sim_model.tokenizer.bins[None, :], axis=-1)  # (N,)


# ── Observation (381 dims) ────────────────────────────────────



def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    """Extract raw (unscaled) future profile."""
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)


def _curv(lat, roll, v):
    """Curvature: κ = (lat - roll) / max(v², 1.0)"""
    return (lat - roll) / max(v * v, 1.0)


def _hist(buf, scale):
    """Convert history buffer to scaled numpy array."""
    return np.array(buf, np.float32) / scale


def build_obs(target, current, state, fplan,
              hist_act, hist_lat, hist_v, hist_a, hist_roll,
              innov=0.0, hist_innov=None):
    """256-dim observation:
      16 core + 2×20 history + 4×50 future.

    Core (16): target, current, error, κ_target, κ_current, κ_error,
               v_ego, a_ego, roll_lataccel, prev_act, innovation,
               target_rate, current_rate, steer_rate, friction_circle, v².
    History (40): prev_act, current_lataccel (20 each).
    Future (200): lataccel, roll_lataccel, v_ego, a_ego (50 each).
    """
    if hist_innov is None:
        hist_innov = [0.0] * HIST_LEN
    error = target - current
    k_target  = _curv(target, state.roll_lataccel, state.v_ego)
    k_current = _curv(current, state.roll_lataccel, state.v_ego)

    obs = np.empty(STATE_DIM, np.float32)
    i = 0
    # Core (11 dims)
    obs[i] = target / S_LAT;               i += 1
    obs[i] = current / S_LAT;              i += 1
    obs[i] = error / S_LAT;                i += 1
    obs[i] = k_target / S_CURV;            i += 1
    obs[i] = k_current / S_CURV;           i += 1
    obs[i] = (k_target - k_current) / S_CURV; i += 1
    obs[i] = state.v_ego / S_VEGO;         i += 1
    obs[i] = state.a_ego / S_AEGO;         i += 1
    obs[i] = state.roll_lataccel / S_ROLL; i += 1
    obs[i] = hist_act[-1] / S_STEER;       i += 1   # prev_act
    obs[i] = innov / S_LAT;                i += 1   # innovation
    # Physics features (5 dims)
    _flat = getattr(fplan, 'lataccel', None)
    fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target
    obs[i] = (fplan_lat0 - target) / DEL_T / S_LAT;              i += 1  # target_rate
    obs[i] = (current - hist_lat[-1]) / DEL_T / S_LAT;           i += 1  # current_rate
    obs[i] = (hist_act[-1] - hist_act[-2]) / DEL_T / S_STEER;   i += 1  # steer_rate
    obs[i] = np.sqrt(current**2 + state.a_ego**2) / 7.0;        i += 1  # friction_circle
    obs[i] = max(0.0, 1.0 - np.sqrt(current**2 + state.a_ego**2) / 7.0); i += 1  # grip_headroom
    # History (40 dims = 2 × 20)
    obs[i:i+HIST_LEN] = _hist(hist_act,   S_STEER); i += HIST_LEN
    obs[i:i+HIST_LEN] = _hist(hist_lat,   S_LAT);   i += HIST_LEN
    # Future plan (200 dims = 4 × 50)
    f_lat  = _future_raw(fplan, 'lataccel',      target)
    f_roll = _future_raw(fplan, 'roll_lataccel',  state.roll_lataccel)
    f_v    = _future_raw(fplan, 'v_ego',          state.v_ego)
    f_a    = _future_raw(fplan, 'a_ego',          state.a_ego)
    obs[i:i+FUTURE_K] = f_lat / S_LAT;   i += FUTURE_K
    obs[i:i+FUTURE_K] = f_roll / S_ROLL;  i += FUTURE_K
    obs[i:i+FUTURE_K] = f_v / S_VEGO;     i += FUTURE_K
    obs[i:i+FUTURE_K] = f_a / S_AEGO;     i += FUTURE_K
    return np.clip(obs, -5.0, 5.0)


# ── Network (orthogonal init, σ floor) ───────────────────────

def _ortho_init(module, gain=np.sqrt(2)):
    """Orthogonal init for Linear layers (standard PPO)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))       # Beta: (α_raw, β_raw)
        self.actor = nn.Sequential(*a)

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        # Orthogonal init: √2 for hidden, 0.01 for actor output, 1.0 for critic output
        for layer in self.actor[:-1]:
            _ortho_init(layer)
        _ortho_init(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]:
            _ortho_init(layer)
        _ortho_init(self.critic[-1], gain=1.0)

    def beta_params(self, obs_t):
        """Return (alpha, beta) each shape (...,) from actor output."""
        out = self.actor(obs_t)                    # (..., 2)
        alpha = F.softplus(out[..., 0]) + 1.0
        beta  = F.softplus(out[..., 1]) + 1.0
        return alpha, beta

    @torch.inference_mode()
    def act(self, obs_np, deterministic=False):
        s   = torch.from_numpy(obs_np).unsqueeze(0)
        a, b = self.beta_params(s)
        val  = self.critic(s).item()
        if deterministic:
            raw = (2.0 * a / (a + b) - 1.0).item()       # Beta mean → [-1,1]
        else:
            x   = torch.distributions.Beta(a, b).sample()
            raw = (2.0 * x - 1.0).item()
        return raw, val


# ── Controller ───────────────────────────────────────────────

class DeltaController(BaseController):
    def __init__(self, ac, deterministic=False):
        self.ac, self.det = ac, deterministic
        self.n = 0
        self._h_act   = [0.0] * HIST_LEN
        self._h_lat   = [0.0] * HIST_LEN
        self._h_v     = [0.0] * HIST_LEN
        self._h_a     = [0.0] * HIST_LEN
        self._h_roll  = [0.0] * HIST_LEN
        self._h_innov = [0.0] * HIST_LEN
        self._pred_la = None        # expected lataccel from previous step
        self._model   = None        # ONNX model (set via set_model)
        self.traj = []

    def set_model(self, model):
        """Receive ONNX model reference (called by TinyPhysicsSimulator)."""
        self._model = model

    def _push(self, action, current, state, innov):
        self._h_act.append(action);                  self._h_act.pop(0)
        self._h_lat.append(current);                  self._h_lat.pop(0)
        self._h_v.append(state.v_ego);                self._h_v.pop(0)
        self._h_a.append(state.a_ego);                self._h_a.pop(0)
        self._h_roll.append(state.roll_lataccel);     self._h_roll.pop(0)
        self._h_innov.append(innov);                  self._h_innov.pop(0)

    def _predict_next(self):
        """Run ONNX model to get expected next lataccel (deterministic mean)."""
        if self._model is None:
            return None
        h_states = list(zip(self._h_roll, self._h_v, self._h_a))
        return predict_expected_la(self._model, self._h_act, h_states, self._h_lat)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        step_idx = CONTEXT_LENGTH + self.n - 1

        # Compute innovation: actual − expected
        if self._pred_la is not None:
            innov = current_lataccel - self._pred_la
        else:
            innov = 0.0

        if step_idx < CONTROL_START_IDX:
            self._push(0.0, current_lataccel, state, innov)
            self._pred_la = self._predict_next()
            return 0.0   # simulator overrides anyway

        obs = build_obs(target_lataccel, current_lataccel, state, future_plan,
                        self._h_act, self._h_lat, self._h_v, self._h_a, self._h_roll,
                        innov=innov, hist_innov=self._h_innov)
        raw, val = self.ac.act(obs, self.det)

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))
        self._push(action, current_lataccel, state, innov)
        self._pred_la = self._predict_next()

        if step_idx < COST_END_IDX:
            self.traj.append(dict(obs=obs, raw=raw, val=val,
                                  tgt=target_lataccel, cur=current_lataccel))
        return action


# ── Rewards ──────────────────────────────────────────────────

def compute_rewards(traj):
    tgt = np.array([t['tgt'] for t in traj], np.float32)
    cur = np.array([t['cur'] for t in traj], np.float32)
    lat  = (tgt - cur)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
    jerk = np.diff(cur, prepend=cur[0]) / DEL_T
    return (-(lat + jerk**2 * 100) / 500.0).astype(np.float32)


# ── PPO ──────────────────────────────────────────────────────

class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(ac.actor.parameters(), lr=PI_LR)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR)

    @staticmethod
    def gae(all_r, all_v, all_d):
        advs, rets = [], []
        for r, v, d in zip(all_r, all_v, all_d):
            T = len(r); adv = np.zeros(T, np.float32); g = 0.0
            for t in range(T-1, -1, -1):
                nv = 0.0 if t == T-1 else v[t+1]
                g = (r[t] + GAMMA*nv*(1-d[t]) - v[t]) + GAMMA*LAMDA*(1-d[t])*g
                adv[t] = g
            advs.append(adv); rets.append(adv + v)
        a = np.concatenate(advs); r = np.concatenate(rets)
        return (a - a.mean()) / (a.std() + 1e-8), r

    @staticmethod
    def _beta_sigma(a, b):
        """Std of 2*Beta(a,b)-1 in [-1,1]."""
        return 2.0 * torch.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))

    def update(self, all_obs, all_raw, all_rew, all_val, all_done,
               critic_only=False):
        obs_t = torch.FloatTensor(np.concatenate(all_obs))
        raw_t = torch.FloatTensor(np.concatenate(all_raw)).unsqueeze(-1)
        adv, ret = self.gae(all_rew, all_val, all_done)
        adv_t = torch.FloatTensor(adv)

        ret_t = torch.FloatTensor(ret)

        # Map raw actions to Beta support (0,1) with clamping for log-prob safety
        x_t = ((raw_t + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)

        with torch.no_grad():
            a_old, b_old = self.ac.beta_params(obs_t)
            old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t.squeeze(-1))
            old_val = self.ac.critic(obs_t).squeeze(-1)

        _LOG2 = np.log(2.0)
        N = len(obs_t)
        for _ in range(K_EPOCHS):
            for idx in torch.randperm(N).split(MINI_BS):
                val = self.ac.critic(obs_t[idx]).squeeze(-1)
                v_clipped = old_val[idx] + (val - old_val[idx]).clamp(-10.0, 10.0)
                vf_loss = torch.max(
                    F.huber_loss(val, ret_t[idx], delta=10.0, reduction='none'),
                    F.huber_loss(v_clipped, ret_t[idx], delta=10.0, reduction='none'),
                ).mean()

                if critic_only:
                    self.vf_opt.zero_grad()
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                    self.vf_opt.step()
                else:
                    a_cur, b_cur = self.ac.beta_params(obs_t[idx])
                    dist = torch.distributions.Beta(a_cur, b_cur)
                    lp   = dist.log_prob(x_t[idx].squeeze(-1))  # on (0,1)

                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * adv_t[idx],
                        ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_t[idx]).mean()
                    ent = dist.entropy().mean()   # +log2 is constant, drops out

                    loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent
                    self.pi_opt.zero_grad(); self.vf_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                    self.pi_opt.step(); self.vf_opt.step()

        pi_val = pi_loss.item() if not critic_only else 0.0
        ent_val = ent.item() if not critic_only else 0.0
        # Report mean σ of the Beta policy for diagnostics
        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs_t[:1000])
            sigma = self._beta_sigma(a_d, b_d).mean().item()
        return dict(pi=pi_val, vf=vf_loss.item(),
                    ent=ent_val, σ=sigma,
                    lr=PI_LR)


# ── Batched rollout via BatchedSimulator ──────────────────────

def build_obs_batch(target, current, roll_la, v_ego, a_ego,
                    h_act, h_lat, fplan_data, step_idx,
                    innov=None, h_innov=None):
    """Vectorized build_obs for N episodes.  Returns (N, 256) float32."""
    N = target.shape[0]
    T = fplan_data['target_lataccel'].shape[1]
    obs = np.empty((N, STATE_DIM), np.float32)

    if innov is None:
        innov = np.zeros(N, np.float32)
    if h_innov is None:
        h_innov = np.zeros((N, HIST_LEN), np.float32)

    error = target - current
    v2 = np.maximum(v_ego * v_ego, 1.0)
    k_target  = (target - roll_la) / v2
    k_current = (current - roll_la) / v2

    i = 0
    obs[:, i] = target / S_LAT;                i += 1
    obs[:, i] = current / S_LAT;               i += 1
    obs[:, i] = error / S_LAT;                 i += 1
    obs[:, i] = k_target / S_CURV;             i += 1
    obs[:, i] = k_current / S_CURV;            i += 1
    obs[:, i] = (k_target - k_current) / S_CURV; i += 1
    obs[:, i] = v_ego / S_VEGO;                i += 1
    obs[:, i] = a_ego / S_AEGO;                i += 1
    obs[:, i] = roll_la / S_ROLL;              i += 1
    # History: cast to float32 before dividing, matching _hist() in build_obs
    # which does np.array(lst, float32) / scale
    h_act32 = h_act.astype(np.float32) if h_act.dtype != np.float32 else h_act
    obs[:, i] = h_act32[:, -1] / S_STEER;       i += 1
    obs[:, i] = innov / S_LAT;                   i += 1   # innovation
    # Physics features (5 dims)
    T_ = fplan_data['target_lataccel'].shape[1]
    fplan_lat0 = fplan_data['target_lataccel'][:, min(step_idx + 1, T_ - 1)]
    obs[:, i] = (fplan_lat0 - target) / DEL_T / S_LAT;           i += 1  # target_rate
    obs[:, i] = (current - h_lat[:, -1]) / DEL_T / S_LAT;        i += 1  # current_rate
    obs[:, i] = (h_act32[:, -1] - h_act32[:, -2]) / DEL_T / S_STEER; i += 1  # steer_rate
    obs[:, i] = np.sqrt(current**2 + a_ego**2) / 7.0;            i += 1  # friction_circle
    obs[:, i] = np.maximum(0.0, 1.0 - np.sqrt(current**2 + a_ego**2) / 7.0); i += 1  # grip_headroom
    obs[:, i:i+HIST_LEN] = h_act32 / S_STEER;   i += HIST_LEN
    obs[:, i:i+HIST_LEN] = h_lat / S_LAT;        i += HIST_LEN

    # Future plan: cast to float32 before dividing, matching _future_raw()
    # which does np.asarray(vals[:k], float32)
    end = min(step_idx + FUTURE_PLAN_STEPS, T)
    def _pad_future(arr_slice, k=FUTURE_K):
        if arr_slice.shape[1] == 0: return None
        if arr_slice.shape[1] < k:
            return np.concatenate([arr_slice,
                np.repeat(arr_slice[:, -1:], k - arr_slice.shape[1], axis=1)], 1)
        return arr_slice
    future_raw = {}
    fallback = {'target_lataccel': target, 'roll_lataccel': roll_la,
                'v_ego': v_ego, 'a_ego': a_ego}
    for attr, scale in [('target_lataccel', S_LAT), ('roll_lataccel', S_ROLL),
                        ('v_ego', S_VEGO), ('a_ego', S_AEGO)]:
        slc = fplan_data[attr][:, step_idx+1:end]
        padded = _pad_future(slc)
        if padded is None:
            padded = np.repeat(fallback[attr][:, None], FUTURE_K, axis=1)
        padded32 = padded.astype(np.float32)
        future_raw[attr] = padded32
        obs[:, i:i+FUTURE_K] = padded32 / scale;  i += FUTURE_K
    return np.clip(obs, -5.0, 5.0)


def batched_rollout(csv_files, ac, mdl_path, deterministic=False, ort_session=None):
    """Run N rollouts via BatchedSimulator.  Returns list of per-episode tuples
    (obs, raw, rewards, values, dones, cost) for train, or list of costs for eval.
    """
    sim = BatchedSimulator(str(mdl_path), csv_files, ort_session=ort_session)
    sim.compute_expected = True   # piggyback E[lataccel] from sim_step's ONNX pass
    N = sim.N

    # Controller state (lives outside the simulator)
    # h_act must be float64 — actions accumulate via prev + delta, matching
    # DeltaController which stores Python floats (float64) in a list.
    h_act   = np.zeros((N, HIST_LEN), np.float64)
    h_lat   = np.zeros((N, HIST_LEN), np.float32)
    h_innov = np.zeros((N, HIST_LEN), np.float32)

    all_obs, all_raw, all_val = [], [], []
    tgt_hist, cur_hist = [], []

    def controller_fn(step_idx, target, current_la, state_dict, future_plan):
        nonlocal h_act, h_lat, h_innov

        roll_la = state_dict['roll_lataccel']
        v_ego   = state_dict['v_ego']
        a_ego   = state_dict['a_ego']

        cla32  = np.float32(current_la)

        # Innovation: actual − expected  (piggybacked from previous sim_step)
        if sim.expected_lataccel is not None:
            innov = np.float32(current_la - sim.expected_lataccel)
        else:
            innov = np.zeros(N, np.float32)

        if step_idx < CONTROL_START_IDX:
            # Warmup: controller pushes 0.0 action (matches DeltaController)
            h_act   = np.concatenate([h_act[:, 1:],   np.zeros((N, 1), np.float64)], axis=1)
            h_lat   = np.concatenate([h_lat[:, 1:],   cla32[:, None]], axis=1)
            h_innov = np.concatenate([h_innov[:, 1:], innov[:, None]], axis=1)
            return np.zeros(N)  # placeholder; sim overrides with CSV steer

        # Build obs → policy forward
        obs = build_obs_batch(target, current_la, roll_la, v_ego, a_ego,
                              h_act, h_lat,
                              sim.data, step_idx,
                              innov=innov, h_innov=h_innov)

        obs_t = torch.from_numpy(obs)
        with torch.inference_mode():
            a_p, b_p = ac.beta_params(obs_t)
            val = ac.critic(obs_t).squeeze(-1).numpy()

        if deterministic:
            raw = (2.0 * a_p / (a_p + b_p) - 1.0).numpy()      # Beta mean
        else:
            x   = torch.distributions.Beta(a_p, b_p).sample()
            raw = (2.0 * x - 1.0).numpy()

        delta  = np.clip(np.float64(raw) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        action = np.clip(h_act[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])

        h_act   = np.concatenate([h_act[:, 1:],   action[:, None]], axis=1)
        h_lat   = np.concatenate([h_lat[:, 1:],   cla32[:, None]], axis=1)
        h_innov = np.concatenate([h_innov[:, 1:], innov[:, None]], axis=1)

        if step_idx < COST_END_IDX:
            all_obs.append(obs.copy())
            all_raw.append(raw.copy())
            all_val.append(val.copy())

        if step_idx < COST_END_IDX:
            tgt_hist.append(target.copy())
            cur_hist.append(current_la.copy())

        return action

    cost_dict = sim.rollout(controller_fn)
    total_costs = cost_dict['total_cost']  # (N,)

    if deterministic:
        return total_costs.tolist()

    # Assemble per-episode training data
    obs_arr = np.stack(all_obs, axis=1)  # (N, S, 360)
    raw_arr = np.stack(all_raw, axis=1)
    val_arr = np.stack(all_val, axis=1)
    tgt_arr = np.stack(tgt_hist, axis=1)
    cur_arr = np.stack(cur_hist, axis=1)
    S = obs_arr.shape[1]

    results = []
    for i in range(N):
        dones = np.concatenate([np.zeros(S-1, np.float32), [1.0]])
        tgt_ep, cur_ep = tgt_arr[i], cur_arr[i]
        lat_r  = (tgt_ep - cur_ep)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
        jerk_r = np.diff(cur_ep, prepend=cur_ep[0]) / DEL_T
        rew    = (-(lat_r + jerk_r**2 * 100) / 500.0).astype(np.float32)
        results.append((obs_arr[i], raw_arr[i], rew, val_arr[i], dones, float(total_costs[i])))
    return results


# ── Remote orchestration ─────────────────────────────────────

SSH_SOCK   = '/tmp/_ppo_ssh_mux_%r@%h:%p'
SSH_BASE   = ['-i', SSH_KEY, '-o', 'ConnectTimeout=5',
              '-o', f'ControlPath={SSH_SOCK}',
              '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600']


def _sync_remote():
    """Rsync train.py + remote_worker.py to spare Mac so code stays in sync."""
    src = str(EXP_DIR) + '/'
    dst = f'{REMOTE_HOST}:{REMOTE_DIR}/experiments/exp050_rich_ppo/'
    r = subprocess.run(
        ['rsync', '-az', '--include=*.py', '--exclude=*',
         '-e', 'ssh ' + ' '.join(SSH_BASE), src, dst],
        capture_output=True, timeout=15)
    if r.returncode == 0:
        print("  [remote] synced .py files ✓")
    else:
        print(f"  [remote] sync failed: {r.stderr.decode()[:200]}")


def _scp_to(local, remote):
    """scp a file to REMOTE_HOST. Returns True on success."""
    r = subprocess.run(['scp'] + SSH_BASE + ['-q', str(local),
                        f'{REMOTE_HOST}:{remote}'],
                       capture_output=True, timeout=30)
    return r.returncode == 0


def _scp_from(remote, local):
    """scp a file from REMOTE_HOST. Returns True on success."""
    r = subprocess.run(['scp'] + SSH_BASE + ['-q',
                        f'{REMOTE_HOST}:{remote}', str(local)],
                       capture_output=True, timeout=60)
    return r.returncode == 0


def _launch_remote(batch_csvs, ckpt_path, mode='train'):
    """Send checkpoint + batch list to spare Mac, kick off remote_worker.py.
    Returns a subprocess.Popen (non-blocking) or None on failure."""
    if not USE_REMOTE:
        return None
    try:
        remote_ckpt  = f'{REMOTE_DIR}/experiments/exp050_rich_ppo/.remote_ckpt.pt'
        remote_batch = '/tmp/_remote_batch.json'
        remote_out   = '/tmp/_remote_results.npz'

        # Write batch JSON locally
        batch_json = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        json.dump(batch_csvs, batch_json)
        batch_json.close()

        # Upload checkpoint + batch in one scp call
        ok = subprocess.run(
            ['scp'] + SSH_BASE + ['-q', str(ckpt_path), batch_json.name,
             f'{REMOTE_HOST}:/tmp/'],
            capture_output=True, timeout=30).returncode == 0
        os.unlink(batch_json.name)

        if not ok:
            print("  [remote] scp upload failed — falling back to local")
            return None

        # Move checkpoint to correct path on remote, rename batch file
        _pre = (f'mv /tmp/{Path(ckpt_path).name} {remote_ckpt}; '
                f'mv /tmp/{Path(batch_json.name).name} {remote_batch}; ')

        # Launch remote_worker.py via SSH (non-blocking)
        cmd = (_pre +
               f'cd {REMOTE_DIR} && {REMOTE_PY} '
               f'experiments/exp050_rich_ppo/remote_worker.py '
               f'--ckpt {remote_ckpt} --batch {remote_batch} '
               f'--out {remote_out} --workers {REMOTE_WORKERS} '
               f'--mode {mode}')
        proc = subprocess.Popen(
            ['ssh'] + SSH_BASE + [REMOTE_HOST, cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc
    except Exception as e:
        print(f"  [remote] launch failed: {e}")
        return None


def _collect_remote(proc, mode='train'):
    """Wait for remote worker to finish, fetch and parse results.
    Returns (list_of_tuples, remote_tag) for train, or (costs_list,) for eval."""
    if proc is None:
        return [] if mode == 'train' else []
    try:
        proc.wait(timeout=600)
        if proc.returncode != 0:
            err = proc.stderr.read().decode() if proc.stderr else ''
            print(f"  [remote] worker failed (rc={proc.returncode}): {err[:200]}")
            return []

        # Fetch results
        local_npz = EXP_DIR / '.remote_results.npz'
        _scp_from('/tmp/_remote_results.npz', local_npz)

        data = np.load(str(local_npz), allow_pickle=False)
        if mode == 'eval':
            costs = data['costs'].tolist()
            local_npz.unlink(missing_ok=True)
            return costs

        # Train mode: split concatenated arrays back into per-episode lists
        obs_all     = data['obs']
        raw_all     = data['raw']
        rew_all     = data['rew']
        val_all     = data['val']
        done_all    = data['done']
        costs       = data['costs']
        ep_lens     = data['ep_lens']

        results = []
        offset = 0
        for i, L in enumerate(ep_lens):
            results.append((
                obs_all[offset:offset+L],
                raw_all[offset:offset+L],
                rew_all[offset:offset+L],
                val_all[offset:offset+L],
                done_all[offset:offset+L],
                float(costs[i]),
            ))
            offset += L
        local_npz.unlink(missing_ok=True)
        return results
    except Exception as e:
        print(f"  [remote] collect failed: {e}")
        return []


# ── Behavioral cloning pretrain ──────────────────────────────

def _bc_worker(csv_path):
    """Extract (obs, raw_target) pairs from one CSV's warmup steer data."""
    df = pd.read_csv(csv_path)
    data = pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values,
    })
    steer = data['steer_command'].values
    tgt   = data['target_lataccel'].values
    obs_list, raw_list = [], []

    # Build history buffers from zero, same as DeltaController
    h_act  = [0.0] * HIST_LEN
    h_lat  = [0.0] * HIST_LEN
    h_v    = [0.0] * HIST_LEN
    h_a    = [0.0] * HIST_LEN
    h_roll = [0.0] * HIST_LEN

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        target_la = tgt[step_idx]
        # In BC replay, current_lataccel ≈ target (expert tracking)
        current_la = tgt[step_idx]

        state = State(
            roll_lataccel=data['roll_lataccel'].values[step_idx],
            v_ego=data['v_ego'].values[step_idx],
            a_ego=data['a_ego'].values[step_idx])
        fplan = FuturePlan(
            lataccel=tgt[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=data['roll_lataccel'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            v_ego=data['v_ego'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            a_ego=data['a_ego'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist())

        obs = build_obs(target_la, current_la, state, fplan,
                        h_act, h_lat, h_v, h_a, h_roll)

        prev_act = h_act[-1]
        delta = steer[step_idx] - prev_act
        raw_target = np.clip(delta / DELTA_SCALE, -1.0, 1.0)  # Beta support
        obs_list.append(obs)
        raw_list.append(raw_target)

        # Push this step's values into history
        h_act.append(steer[step_idx]);                         h_act.pop(0)
        h_lat.append(tgt[step_idx]);                           h_lat.pop(0)
        h_v.append(data['v_ego'].values[step_idx]);            h_v.pop(0)
        h_a.append(data['a_ego'].values[step_idx]);            h_a.pop(0)
        h_roll.append(data['roll_lataccel'].values[step_idx]); h_roll.pop(0)

    return (np.array(obs_list, np.float32),
            np.array(raw_list, np.float32))


def pretrain_bc(ac, csv_files, epochs=BC_EPOCHS, lr=BC_LR, batch_size=BC_BS):
    """Behavioral cloning: pretrain actor on CSV warmup steer data."""
    print(f"BC pretrain: extracting from {len(csv_files)} CSVs ...")
    results = process_map(_bc_worker, [str(f) for f in csv_files],
                          max_workers=BC_WORKERS, chunksize=50, disable=False)
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    N = len(all_obs)
    print(f"BC pretrain: {N} samples, {epochs} epochs")

    obs_t = torch.FloatTensor(all_obs)
    raw_t = torch.FloatTensor(all_raw)
    opt = optim.Adam(list(ac.actor.parameters()), lr=lr)

    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for idx in perm.split(batch_size):
            a_p, b_p = ac.beta_params(obs_t[idx])
            x_target = ((raw_t[idx] + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x_target).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  BC epoch {ep}: loss={total_loss/n_batches:.6f}")

    print("BC pretrain done.\n")


# ── Pool workers ─────────────────────────────────────────────

_W = {}   # per-process training-specific globals

def _pool_init(mdl_path):
    """Per-worker init: ONNX cache (via batched module) + training state."""
    pool_init(mdl_path)                                 # cache ONNX session
    torch.set_num_threads(1)
    _W['mdl'] = TinyPhysicsModel(str(mdl_path), debug=False)  # for BATCHED=0
    _W['ckpt_mtime'] = None
    _W['ac'] = None

def _load_ckpt(ckpt):
    """Reload checkpoint (mtime-aware)."""
    mtime = os.path.getmtime(ckpt)
    if _W.get('ckpt_mtime') != mtime:
        data = torch.load(ckpt, weights_only=False, map_location='cpu')
        ac = ActorCritic()
        ac.load_state_dict(data['ac'])
        ac.eval()
        _W['ac'] = ac
        _W['ckpt_mtime'] = mtime

def _batched_train_worker(args):
    csv_chunk, ckpt = args
    _load_ckpt(ckpt)
    c = get_pool_cache()
    return batched_rollout(csv_chunk, _W['ac'], c['model_path'],
                           deterministic=False, ort_session=c['ort_session'])

def _batched_eval_worker(args):
    csv_chunk, ckpt = args
    _load_ckpt(ckpt)
    c = get_pool_cache()
    return batched_rollout(csv_chunk, _W['ac'], c['model_path'],
                           deterministic=True, ort_session=c['ort_session'])

def _seq_train_worker(args):
    csv, ckpt = args
    _load_ckpt(ckpt)
    ctrl = DeltaController(_W['ac'], deterministic=False)
    sim = TinyPhysicsSimulator(_W['mdl'], str(csv), controller=ctrl, debug=False)
    cost = sim.rollout()
    T = len(ctrl.traj)
    return (np.array([t['obs'] for t in ctrl.traj], np.float32),
            np.array([t['raw'] for t in ctrl.traj], np.float32),
            compute_rewards(ctrl.traj),
            np.array([t['val'] for t in ctrl.traj], np.float32),
            np.concatenate([np.zeros(T-1, np.float32), [1.0]]),
            cost['total_cost'])

def _seq_eval_worker(args):
    csv, ckpt = args
    _load_ckpt(ckpt)
    ctrl = DeltaController(_W['ac'], deterministic=True)
    sim = TinyPhysicsSimulator(_W['mdl'], str(csv), controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


# ── Training ─────────────────────────────────────────────────

class Ctx:
    def __init__(self):
        self.ac  = ActorCritic()
        self.ppo = PPO(self.ac)
        self.mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
        all_f = sorted((ROOT / 'data').glob('*.csv'))
        self.va_f = all_f[:EVAL_N]
        rest = all_f[EVAL_N:]
        random.seed(42)
        random.shuffle(rest)
        self.tr_f = rest
        self.best = float('inf')
        self.best_ep = -1
        self.pool = multiprocessing.Pool(
            WORKERS, initializer=_pool_init, initargs=(self.mdl_path,))

    def save_ckpt(self, path=TMP):
        torch.save({'ac': self.ac.state_dict()}, path)

    def save_best(self):
        torch.save({'ac': self.ac.state_dict()}, BEST_PT)


def evaluate(ctx, files, n=EVAL_N):
    ctx.save_ckpt(TMP)
    all_files = files[:n]
    if BATCHED:
        costs = run_parallel_chunked(ctx.pool, all_files,
                                     _batched_eval_worker, WORKERS,
                                     extra_args=(str(TMP),))
    else:
        args = [(f, str(TMP)) for f in all_files]
        costs = list(ctx.pool.map(_seq_eval_worker, args))
    if not costs:
        return float('inf'), 0.0
    return float(np.mean(costs)), float(np.std(costs))


def train_one_epoch(epoch, ctx):
    t0 = time.time()
    ctx.save_ckpt(TMP)
    batch = random.sample(ctx.tr_f, min(CSVS_EPOCH, len(ctx.tr_f)))

    if BATCHED:
        res = run_parallel_chunked(ctx.pool, batch,
                                   _batched_train_worker, WORKERS,
                                   extra_args=(str(TMP),))
    else:
        args = [(f, str(TMP)) for f in batch]
        res = list(ctx.pool.map(_seq_train_worker, args))
    tc = time.time() - t0

    if not res:
        print(f"E{epoch:3d}  NO DATA — skipping update"); return

    t1 = time.time()
    co = epoch < CRITIC_WARMUP
    info = ctx.ppo.update(
        [r[0] for r in res], [r[1] for r in res],   # obs, raw_actions
        [r[2] for r in res], [r[3] for r in res],   # rewards, values
        [r[4] for r in res],                          # dones
        critic_only=co)
    tu = time.time() - t1

    costs = [r[5] for r in res]
    phase = "  [critic warmup]" if co else ""
    line = (f"E{epoch:3d}  train={np.mean(costs):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  lr={info['lr']:.1e}  ⏱{tc:.0f}+{tu:.0f}s{phase}")

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(ctx, ctx.va_f)
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            ctx.save_best()
            mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"

    print(line)


def train():
    ctx = Ctx()

    if RESUME and (EXP_DIR / 'best_model.pt').exists():
        ckpt = torch.load(EXP_DIR / 'best_model.pt', weights_only=False, map_location='cpu')
        ctx.ac.load_state_dict(ckpt['ac'])
        print(f"Resumed from best_model.pt")
    else:
        # Behavioral cloning warm-start: pretrain actor on CSV steer data
        all_csvs = sorted((ROOT / 'data').glob('*.csv'))
        pretrain_bc(ctx.ac, all_csvs)
        # Beta concentrations learned during BC — no manual σ reset needed

    vm, vs = evaluate(ctx, ctx.va_f)
    ctx.best, ctx.best_ep = vm, 'init'
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")
    ctx.save_best()

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  workers={WORKERS}  (batched chunks)")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}"
          f"  dist=Beta  Δscale={DELTA_SCALE}"
          f"  layers={A_LAYERS}+{C_LAYERS}  K={K_EPOCHS}  dim={STATE_DIM}\n")

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best val: {ctx.best:.1f} (epoch {ctx.best_ep})")
    ctx.save_ckpt(EXP_DIR / 'final_model.pt')
    ctx.pool.terminate(); ctx.pool.join()
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
