import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, DEL_T

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN   = 256, 256
A_LAYERS            = 4
DELTA_SCALE_DEFAULT = 0.25
MAX_DELTA           = 0.5

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

torch.set_num_threads(1)


def _future(fplan, attr, fallback, k=FUTURE_K):
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
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)


class Controller(BaseController):
    def __init__(self):
        exp = Path(__file__).parent.parent / 'experiments' / 'exp052_clean_ppo'
        ckpt = None
        model_env = os.getenv('MODEL', '').strip()
        if model_env:
            p = Path(model_env)
            if not p.exists():
                raise FileNotFoundError(f"MODEL path does not exist: {p}")
            ckpt = str(p)
        else:
            for name in ('best_model.pt', 'final_model.pt'):
                p = exp / name
                if p.exists():
                    ckpt = str(p)
                    break
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {exp}")

        self.ac = ActorCritic()
        data = torch.load(ckpt, weights_only=False, map_location='cpu')
        self.ac.load_state_dict(data['ac'], strict=False)
        self.ac.eval()
        ds = data.get('delta_scale', None)
        if ds is None:
            best_meta = Path(ckpt).with_name('best_model.pt')
            if best_meta.exists():
                meta = torch.load(best_meta, weights_only=False, map_location='cpu')
                ds = meta.get('delta_scale', None)
        self.delta_scale = float(ds) if ds is not None else DELTA_SCALE_DEFAULT

        self.n = 0
        self._h_act   = [0.0] * HIST_LEN
        self._h_lat   = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN

    def _push(self, action, current):
        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current]

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan,
                   error_integral, h_act, h_lat):
        error = target_lataccel - current_lataccel
        k_tgt = _curv(target_lataccel, state.roll_lataccel, state.v_ego)
        k_cur = _curv(current_lataccel, state.roll_lataccel, state.v_ego)
        _flat = getattr(future_plan, 'lataccel', None)
        fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target_lataccel
        fric = np.sqrt(current_lataccel**2 + state.a_ego**2) / 7.0

        core = np.array([
            target_lataccel / S_LAT,
            current_lataccel / S_LAT,
            error / S_LAT,
            k_tgt / S_CURV,
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            state.v_ego / S_VEGO,
            state.a_ego / S_AEGO,
            state.roll_lataccel / S_ROLL,
            h_act[-1] / S_STEER,
            error_integral / S_LAT,
            (fplan_lat0 - target_lataccel) / DEL_T / S_LAT,
            (current_lataccel - h_lat[-1]) / DEL_T / S_LAT,
            (h_act[-1] - h_act[-2]) / DEL_T / S_STEER,
            fric,
            max(0.0, 1.0 - fric),
        ], dtype=np.float32)

        obs = np.concatenate([
            core,
            np.array(h_act, np.float32) / S_STEER,
            np.array(h_lat, np.float32) / S_LAT,
            _future(future_plan, 'lataccel', target_lataccel) / S_LAT,
            _future(future_plan, 'roll_lataccel', state.roll_lataccel) / S_ROLL,
            _future(future_plan, 'v_ego', state.v_ego) / S_VEGO,
            _future(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ])
        return np.clip(obs, -5.0, 5.0)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        WARMUP_N = CONTROL_START_IDX - CONTEXT_LENGTH

        error = target_lataccel - current_lataccel
        self._h_error = self._h_error[1:] + [error]
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if self.n <= WARMUP_N:
            self._push(0.0, current_lataccel)
            return 0.0

        obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan,
                              error_integral, list(self._h_act), list(self._h_lat))

        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0)
            out = self.ac.actor(t)
            a_p = F.softplus(out[..., 0]).item() + 1.0
            b_p = F.softplus(out[..., 1]).item() + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = float(np.clip(raw * self.delta_scale, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))

        self._push(action, current_lataccel)
        return action
