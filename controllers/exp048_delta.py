"""exp048 v5: Delta-action PPO controller (54-dim, Tanh, deterministic)"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE

torch.set_num_threads(1)

STATE_DIM   = 54
HIDDEN      = 128
A_LAYERS    = 3
C_LAYERS    = 2
DELTA_SCALE = 0.1
MAX_DELTA   = 0.3
WARMUP_N    = CONTROL_START_IDX - CONTEXT_LENGTH
FUTURE_K    = 50

OBS_SCALE = torch.tensor([1/3, 1/30, 1/2] + [100.0]*51, dtype=torch.float32)


def _kappa(lat, roll, v):
    return np.clip((lat - roll) / max(v * v, 25.0), -1.0, 1.0)


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
            exp = Path(__file__).parent.parent / 'experiments' / 'exp048_delta_ppo'
            for name in ('best_model.pt', 'final_model.pt', 'bc_init.pt'):
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

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        if self.n <= WARMUP_N:
            e = target_lataccel - current_lataccel
            self._ei += e; de = e - self._pe; self._pe = e
            self.prev_act = float(np.clip(
                0.195*e + 0.1*self._ei - 0.053*de, *STEER_RANGE))
            return 0.0

        # Build 54-dim obs
        obs = np.empty(STATE_DIM, np.float32)
        obs[0] = target_lataccel - current_lataccel
        obs[1] = state.v_ego
        obs[2] = self.prev_act
        obs[3] = _kappa(target_lataccel, state.roll_lataccel, state.v_ego)

        n_f = len(future_plan.lataccel) if future_plan else 0
        if n_f >= FUTURE_K:
            fl = np.asarray(future_plan.lataccel[:FUTURE_K], np.float32)
            fr = np.asarray(future_plan.roll_lataccel[:FUTURE_K], np.float32)
            fv = np.asarray(future_plan.v_ego[:FUTURE_K], np.float32)
            obs[4:54] = np.clip((fl - fr) / np.maximum(fv**2, 25.0), -1.0, 1.0)
        elif n_f > 0:
            fl = np.array(future_plan.lataccel, np.float32)
            fr = np.array(future_plan.roll_lataccel, np.float32)
            fv = np.array(future_plan.v_ego, np.float32)
            p = FUTURE_K - n_f
            fl, fr, fv = [np.pad(x, (0, p), 'edge') for x in (fl, fr, fv)]
            obs[4:54] = np.clip((fl - fr) / np.maximum(fv**2, 25.0), -1.0, 1.0)
        else:
            obs[4:54] = obs[3]

        with torch.no_grad():
            raw = self.ac.actor(
                torch.from_numpy(obs).unsqueeze(0) * OBS_SCALE).item()

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self.prev_act + delta, *STEER_RANGE))
        self.prev_act = action
        return action
