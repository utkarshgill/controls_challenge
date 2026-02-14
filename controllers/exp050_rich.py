"""exp050: Physics-Aligned PPO controller (256-dim, Beta, ReLU, 4+4 layers, deterministic)
With optional Newton 1-step predict-and-correct via ONNX physics model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import (CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, DEL_T,
                          MAX_ACC_DELTA, LATACCEL_RANGE, VOCAB_SIZE,
                          LAT_ACCEL_COST_MULTIPLIER, LataccelTokenizer, State)

torch.set_num_threads(1)

HIST_LEN    = 20
STATE_DIM   = 256
HIDDEN      = 256
A_LAYERS    = 4
C_LAYERS    = 4
DELTA_SCALE = 0.25
MAX_DELTA   = 0.5
WARMUP_N    = CONTROL_START_IDX - CONTEXT_LENGTH
FUTURE_K    = 50

S_LAT   = 5.0
S_STEER = 2.0
S_VEGO  = 40.0
S_AEGO  = 4.0
S_ROLL  = 2.0
S_CURV  = 0.02

LPF_ALPHA  = float(os.getenv('LPF_ALPHA', '0.1'))  # low-pass: 0=off, 0.15=subtle
NEWTON     = int(os.getenv('NEWTON', '0'))          # 1-step predict-and-correct via ONNX
NEWTON_K   = float(os.getenv('NEWTON_K', '0.2'))   # correction gain
NEWTON_MAX = float(os.getenv('NEWTON_MAX', '0.1'))  # max correction magnitude
RATE_LIMIT = float(os.getenv('RATE_LIMIT', '0'))    # max |Δsteer|/step after all corrections (0=off)
VNORM      = int(os.getenv('VNORM', '0'))           # speed-normalize steer: steer *= v/v_ref


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)


def _curv(lat, roll, v):
    return (lat - roll) / max(v * v, 1.0)


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



class Controller(BaseController):
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            exp = Path(__file__).parent.parent / 'experiments' / 'exp050_rich_ppo'
            for name in ('best_model.pt', 'final_model.pt'):
                p = exp / name
                if p.exists():
                    checkpoint_path = str(p); break
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint in {exp}")

        self.ac = ActorCritic()
        data = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        self.ac.load_state_dict(data['ac'])
        self.ac.eval()
        self.n = 0
        self._h_act   = [0.0] * HIST_LEN
        self._h_lat   = [0.0] * HIST_LEN
        self._h_v     = [0.0] * HIST_LEN
        self._h_a     = [0.0] * HIST_LEN
        self._h_roll  = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN
        self._sim_model = None    # set via set_model() hook
        self._tokenizer = LataccelTokenizer()

    # ── hook called by TinyPhysicsSimulator.__init__ ──
    def set_model(self, model):
        self._sim_model = model

    def _push(self, action, current, state):
        self._h_act.append(action);              self._h_act.pop(0)
        self._h_lat.append(current);             self._h_lat.pop(0)
        self._h_v.append(state.v_ego);           self._h_v.pop(0)
        self._h_a.append(state.a_ego);           self._h_a.pop(0)
        self._h_roll.append(state.roll_lataccel); self._h_roll.pop(0)

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan, error_integral, h_act, h_lat):
        """Build the 256-dim observation vector."""
        error = target_lataccel - current_lataccel
        obs = np.empty(STATE_DIM, np.float32)
        i = 0
        obs[i] = target_lataccel / S_LAT;                          i += 1
        obs[i] = current_lataccel / S_LAT;                         i += 1
        obs[i] = error / S_LAT;                                    i += 1
        k_tgt = _curv(target_lataccel, state.roll_lataccel, state.v_ego)
        k_cur = _curv(current_lataccel, state.roll_lataccel, state.v_ego)
        obs[i] = k_tgt / S_CURV;                                   i += 1
        obs[i] = k_cur / S_CURV;                                   i += 1
        obs[i] = (k_tgt - k_cur) / S_CURV;                         i += 1
        obs[i] = state.v_ego / S_VEGO;                             i += 1
        obs[i] = state.a_ego / S_AEGO;                             i += 1
        obs[i] = state.roll_lataccel / S_ROLL;                     i += 1
        obs[i] = h_act[-1] / S_STEER;                              i += 1
        obs[i] = error_integral / S_LAT;                            i += 1
        _flat = getattr(future_plan, 'lataccel', None)
        fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target_lataccel
        obs[i] = (fplan_lat0 - target_lataccel) / DEL_T / S_LAT;  i += 1
        obs[i] = (current_lataccel - h_lat[-1]) / DEL_T / S_LAT;  i += 1
        obs[i] = (h_act[-1] - h_act[-2]) / DEL_T / S_STEER;      i += 1
        obs[i] = np.sqrt(current_lataccel**2 + state.a_ego**2) / 7.0; i += 1
        obs[i] = max(0.0, 1.0 - np.sqrt(current_lataccel**2 + state.a_ego**2) / 7.0); i += 1
        obs[i:i+HIST_LEN] = np.array(h_act, np.float32) / S_STEER;  i += HIST_LEN
        obs[i:i+HIST_LEN] = np.array(h_lat, np.float32) / S_LAT;    i += HIST_LEN
        f_lat  = _future_raw(future_plan, 'lataccel',      target_lataccel)
        f_roll = _future_raw(future_plan, 'roll_lataccel',  state.roll_lataccel)
        f_v    = _future_raw(future_plan, 'v_ego',          state.v_ego)
        f_a    = _future_raw(future_plan, 'a_ego',          state.a_ego)
        obs[i:i+FUTURE_K] = f_lat / S_LAT;   i += FUTURE_K
        obs[i:i+FUTURE_K] = f_roll / S_ROLL;  i += FUTURE_K
        obs[i:i+FUTURE_K] = f_v / S_VEGO;     i += FUTURE_K
        obs[i:i+FUTURE_K] = f_a / S_AEGO;     i += FUTURE_K
        np.clip(obs, -5.0, 5.0, out=obs)
        return obs

    def _sample_action(self, obs):
        """Sample a stochastic delta from the Beta policy."""
        with torch.no_grad():
            out = self.ac.actor(torch.from_numpy(obs).unsqueeze(0))
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            x = torch.distributions.Beta(a_p, b_p).sample()
            raw = (2.0 * x - 1.0).item()
        return raw

    def _mean_action(self, obs):
        """Deterministic (mean) delta from the Beta policy."""
        with torch.no_grad():
            out = self.ac.actor(torch.from_numpy(obs).unsqueeze(0))
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            raw = (2.0 * a_p / (a_p + b_p) - 1.0).item()
        return raw

    # ── Newton: 1-step predict-and-correct via ONNX ──
    def _newton_correct(self, action, target_next, current_lataccel, state):
        """Predict what `action` will produce, nudge to reduce overshoot."""
        CL = CONTEXT_LENGTH
        K = NEWTON_K

        # Build ONNX input matching sim's exact context (include current step)
        h_preds = np.array(self._h_lat + [current_lataccel], np.float64)[-CL:]
        h_states = np.array(
            list(zip(self._h_roll, self._h_v, self._h_a))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)],
            np.float64)[-CL:]                                       # (CL, 3)
        h_actions = np.array(self._h_act, np.float64)[-CL:]
        a = np.concatenate([h_actions[1:], [action]])

        # One ONNX call — predict next lataccel with this action
        pred = self._onnx_expected(
            a[None], h_states[None], h_preds[None])[0]              # scalar

        # Cost-optimal 1-step target: y* = (target + 2*current) / 3
        optimal_next = (target_next + 2.0 * current_lataccel) / 3.0
        overshoot = pred - optimal_next
        correction = np.clip(K * overshoot, -NEWTON_MAX, NEWTON_MAX)
        return float(np.clip(action - correction, *STEER_RANGE))

    def _onnx_expected(self, p_actions, p_states, p_preds):
        """Single batched ONNX call → E[next_lataccel] for each of P proposals."""
        tok = self._tokenizer
        tokenized = tok.encode(p_preds)                        # (P, CL) int
        states_in = np.concatenate(
            [p_actions[:, :, None], p_states], axis=-1)        # (P, CL, 4)
        res = self._sim_model.ort_session.run(None, {
            'states': states_in.astype(np.float32),
            'tokens': tokenized.astype(np.int64),
        })[0]                                                  # (P, CL, VOCAB)
        logits = res[:, -1, :] / 0.8                           # temperature
        e_x = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e_x / e_x.sum(axis=-1, keepdims=True)         # (P, VOCAB)
        return np.sum(probs * tok.bins[None, :], axis=-1)      # (P,) E[la]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1

        error = target_lataccel - current_lataccel
        self._h_error.append(error); self._h_error.pop(0)
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if self.n <= WARMUP_N:
            self._push(0.0, current_lataccel, state)
            return 0.0

        obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan,
                              error_integral, list(self._h_act), list(self._h_lat))
        raw = self._mean_action(obs)
        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))

        # ── Newton: 1-step predict-and-correct ──
        if NEWTON and self._sim_model is not None:
            _flat = getattr(future_plan, 'lataccel', None)
            target_next = _flat[0] if (_flat and len(_flat) > 0) else target_lataccel
            action = self._newton_correct(action, target_next, current_lataccel, state)

        # ── Subtle low-pass: blend towards previous action ──
        if LPF_ALPHA > 0:
            action = (1 - LPF_ALPHA) * action + LPF_ALPHA * self._h_act[-1]

        # ── Final rate limit: cap max steer change per step ──
        if RATE_LIMIT > 0:
            prev = self._h_act[-1]
            action = float(np.clip(action, prev - RATE_LIMIT, prev + RATE_LIMIT))

        # ── Speed-normalized steer: scale output by v/v_ref ──
        # At low speed, less steer needed for same lat_accel; at high speed, more
        if VNORM:
            V_REF = 20.0  # m/s reference speed (~45 mph)
            v = max(state.v_ego, 1.0)
            action = action * (v / V_REF)

        self._push(action, current_lataccel, state)
        return action
