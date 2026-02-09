"""exp050: Rich-obs PPO controller (207-dim, ReLU, 5-layer actor, deterministic)"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE

torch.set_num_threads(1)

HIST_LEN    = 20
STATE_DIM   = 248
HIDDEN      = 256
A_LAYERS    = 4
C_LAYERS    = 4
DELTA_SCALE = 0.3
MAX_DELTA   = 0.3
WARMUP_N    = CONTROL_START_IDX - CONTEXT_LENGTH
FUTURE_K    = 50

OBS_SCALE = torch.tensor(
    [1/0.75, 1/114, 1/0.04, 1/34, 1/0.65, 1/0.05, 1/2]
    + [667.0]*51 + [10000.0]*50
    + [1/0.65]*50
    + [1/2.0]*50
    + [1/2]*HIST_LEN + [1/0.75]*HIST_LEN,
    dtype=torch.float32)


def _kappa(lat, roll, v):
    return np.clip((lat - roll) / max(v * v, 25.0), -1.0, 1.0)


def _future_profile(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 1))
        self.actor = nn.Sequential(*a)
        self.log_std = nn.Parameter(torch.zeros(1))

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
        self.ac.load_state_dict(
            torch.load(checkpoint_path, weights_only=False, map_location='cpu'))
        self.ac.eval()
        self.n, self.prev_act = 0, 0.0
        self._ei, self._pe = 0.0, 0.0
        self._act_hist = [0.0] * HIST_LEN
        self._err_hist = [0.0] * HIST_LEN

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        e = target_lataccel - current_lataccel
        self._ei += e

        if self.n <= WARMUP_N:
            self._pe = e
            self._err_hist.append(e); self._err_hist.pop(0)
            return 0.0  # sim overrides with CSV steer; prev_act stays 0.0

        # Build 167-dim obs
        obs = np.empty(STATE_DIM, np.float32)
        obs[0] = e
        obs[1] = self._ei
        obs[2] = e - self._pe           # error_diff
        obs[3] = state.v_ego
        obs[4] = state.a_ego
        obs[5] = state.roll_lataccel
        obs[6] = self.prev_act
        obs[7] = _kappa(target_lataccel, state.roll_lataccel, state.v_ego)
        self._pe = e

        n_f = len(future_plan.lataccel) if future_plan else 0
        if n_f >= FUTURE_K:
            fl = np.asarray(future_plan.lataccel[:FUTURE_K], np.float32)
            fr = np.asarray(future_plan.roll_lataccel[:FUTURE_K], np.float32)
            fv = np.asarray(future_plan.v_ego[:FUTURE_K], np.float32)
            fk = np.clip((fl - fr) / np.maximum(fv**2, 25.0), -1.0, 1.0)
        elif n_f > 0:
            fl = np.array(future_plan.lataccel, np.float32)
            fr = np.array(future_plan.roll_lataccel, np.float32)
            fv = np.array(future_plan.v_ego, np.float32)
            p = FUTURE_K - n_f
            fl, fr, fv = [np.pad(x, (0, p), 'edge') for x in (fl, fr, fv)]
            fk = np.clip((fl - fr) / np.maximum(fv**2, 25.0), -1.0, 1.0)
        else:
            fk = np.full(FUTURE_K, obs[7], dtype=np.float32)

        obs[8:58] = fk
        obs[58:108] = np.diff(np.concatenate([[obs[7]], fk]))
        obs[108:158] = _future_profile(future_plan, 'a_ego', state.a_ego)
        obs[158:208] = _future_profile(future_plan, 'lataccel', target_lataccel) - current_lataccel
        obs[208:208+HIST_LEN] = self._act_hist
        obs[208+HIST_LEN:208+2*HIST_LEN] = self._err_hist
        np.clip(obs * OBS_SCALE.numpy(), -5.0, 5.0, out=obs)

        with torch.no_grad():
            raw = self.ac.actor(
                torch.from_numpy(obs).unsqueeze(0)).item()

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self.prev_act + delta, *STEER_RANGE))
        self.prev_act = action

        self._act_hist.append(action); self._act_hist.pop(0)
        self._err_hist.append(e); self._err_hist.pop(0)
        return action
