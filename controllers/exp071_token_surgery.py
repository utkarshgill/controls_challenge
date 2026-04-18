import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, DEL_T, STEER_RANGE

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE_DEFAULT = 0.25
MAX_DELTA = 0.5
MAX_CORRECTION_DEFAULT = 0.20
RES_HIDDEN = 128
RES_LAYERS = 2
HARD_GATE_DEFAULT = os.getenv("HARD_GATE", "1") == "1"
GATE_THRESH_DEFAULT = float(os.getenv("GATE_THRESH", "0.65"))
CRIT_GATE_THRESH = float(os.getenv("CRIT_GATE_THRESH", "0.10"))

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
CRIT_LOOKAHEAD = 10

torch.set_num_threads(1)


def _future(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    if vals is not None and len(vals) > 0:
        arr = np.asarray(vals, np.float32)
        return np.pad(arr, (0, k - len(arr)), mode="edge")
    return np.full(k, fallback, dtype=np.float32)


def _curvature(lataccel, roll_lataccel, v_ego):
    return (lataccel - roll_lataccel) / max(v_ego * v_ego, 1.0)


def _criticality(target_series, current):
    horizon = min(len(target_series), CRIT_LOOKAHEAD)
    fp = np.asarray(target_series[:horizon], dtype=np.float32)
    if len(fp) <= 1:
        return abs(float(fp[0]) - current) / S_LAT if len(fp) else 0.0
    slope = np.diff(fp, prepend=fp[:1]) / DEL_T
    mismatch = abs(float(fp[min(4, len(fp) - 1)]) - current) / S_LAT
    span = abs(float(fp[-1]) - float(fp[0])) / S_LAT
    peak_slope = float(np.max(np.abs(slope))) / max(S_LAT / DEL_T, 1e-6)
    flip = 1.0 if np.any(fp[:-1] * fp[1:] < 0.0) else 0.0
    return mismatch + 0.5 * span + 0.5 * peak_slope + 0.5 * flip


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

    def token_and_raw(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return raw, self._features


class Model(nn.Module):
    def __init__(self, max_correction):
        super().__init__()
        self.max_correction = max_correction
        self.base_actor = BaseTokenActor()
        in_dim = HIDDEN + 2

        res = [nn.Linear(in_dim, RES_HIDDEN), nn.ReLU()]
        for _ in range(RES_LAYERS - 1):
            res += [nn.Linear(RES_HIDDEN, RES_HIDDEN), nn.ReLU()]
        res.append(nn.Linear(RES_HIDDEN, 1))
        self.residual_head = nn.Sequential(*res)

        self.gate_head = nn.Sequential(
            nn.Linear(in_dim, RES_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(RES_HIDDEN // 2, 1),
        )

    def runtime_raw(self, obs, criticality, hard_gate, gate_thresh):
        base_raw, token = self.base_actor.token_and_raw(obs)
        x = torch.cat([token, base_raw.unsqueeze(-1), criticality.unsqueeze(-1)], dim=-1)
        corr = torch.tanh(self.residual_head(x)).squeeze(-1) * self.max_correction
        gate_prob = torch.sigmoid(self.gate_head(x).squeeze(-1))
        crit_mask = (criticality >= CRIT_GATE_THRESH).float()
        if hard_gate:
            corr = corr * (gate_prob >= gate_thresh).float() * crit_mask
        else:
            corr = corr * gate_prob * crit_mask
        raw = (base_raw + corr).clamp(-1.0, 1.0)
        return raw


class Controller(BaseController):
    def __init__(self):
        exp = Path(__file__).parent.parent / "experiments" / "exp071_token_surgery"
        ckpt = None

        model_env = os.getenv("MODEL", "").strip()
        if model_env:
            p = Path(model_env)
            if not p.exists():
                raise FileNotFoundError(f"MODEL path does not exist: {p}")
            ckpt = str(p)
        else:
            p = exp / "best_model.pt"
            if p.exists():
                ckpt = str(p)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {exp}")

        data = torch.load(ckpt, weights_only=False, map_location="cpu")
        max_correction = float(data.get("max_correction", MAX_CORRECTION_DEFAULT))
        self.model = Model(max_correction)
        self.model.load_state_dict(data["model"], strict=False)
        self.model.eval()
        self.delta_scale = float(data.get("delta_scale", DELTA_SCALE_DEFAULT))
        self.hard_gate = bool(data.get("hard_gate", HARD_GATE_DEFAULT))
        self.gate_thresh = float(data.get("gate_thresh", GATE_THRESH_DEFAULT))

        self.n = 0
        self._h_act = [0.0] * HIST_LEN
        self._h_lat = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN

    def _push(self, action, current_lataccel):
        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current_lataccel]

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan, error_integral):
        prev_act = self._h_act[-1]
        prev_act2 = self._h_act[-2]
        prev_lat = self._h_lat[-1]
        fplan_lat = getattr(future_plan, "lataccel", None)
        fplan_lat0 = fplan_lat[0] if (fplan_lat and len(fplan_lat) > 0) else target_lataccel
        fric = np.sqrt(current_lataccel**2 + state.a_ego**2) / 7.0
        k_tgt = _curvature(target_lataccel, state.roll_lataccel, state.v_ego)
        k_cur = _curvature(current_lataccel, state.roll_lataccel, state.v_ego)

        core = np.array(
            [
                target_lataccel / S_LAT,
                current_lataccel / S_LAT,
                (target_lataccel - current_lataccel) / S_LAT,
                k_tgt / S_CURV,
                k_cur / S_CURV,
                (k_tgt - k_cur) / S_CURV,
                state.v_ego / S_VEGO,
                state.a_ego / S_AEGO,
                state.roll_lataccel / S_ROLL,
                prev_act / S_STEER,
                error_integral / S_LAT,
                (fplan_lat0 - target_lataccel) / DEL_T / S_LAT,
                (current_lataccel - prev_lat) / DEL_T / S_LAT,
                (prev_act - prev_act2) / DEL_T / S_STEER,
                fric,
                max(0.0, 1.0 - fric),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                core,
                np.asarray(self._h_act, np.float32) / S_STEER,
                np.asarray(self._h_lat, np.float32) / S_LAT,
                _future(future_plan, "lataccel", target_lataccel) / S_LAT,
                _future(future_plan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
                _future(future_plan, "v_ego", state.v_ego) / S_VEGO,
                _future(future_plan, "a_ego", state.a_ego) / S_AEGO,
            ]
        )
        return np.clip(obs, -5.0, 5.0)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        warmup_n = CONTROL_START_IDX - CONTEXT_LENGTH

        error = target_lataccel - current_lataccel
        self._h_error = self._h_error[1:] + [error]
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if self.n <= warmup_n:
            self._push(0.0, current_lataccel)
            return 0.0

        obs = self._build_obs(
            target_lataccel=target_lataccel,
            current_lataccel=current_lataccel,
            state=state,
            future_plan=future_plan,
            error_integral=error_integral,
        )
        crit = _criticality(getattr(future_plan, "lataccel", None) or [target_lataccel], current_lataccel)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            crit_t = torch.tensor([crit], dtype=torch.float32)
            raw = float(self.model.runtime_raw(obs_t, crit_t, self.hard_gate, self.gate_thresh).item())

        delta = float(np.clip(raw * self.delta_scale, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))
        self._push(action, current_lataccel)
        return action
