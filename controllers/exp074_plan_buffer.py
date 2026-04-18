import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, DEL_T

HIST_LEN, FUTURE_K = 20, 50
BASE_STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE_DEFAULT = 0.25
MAX_DELTA = 0.5
PLAN_H_DEFAULT = int(os.getenv("PLAN_H", "6"))
PLAN_COMMIT_DEFAULT = int(os.getenv("PLAN_COMMIT", "1"))
PLAN_BLEND_DEFAULT = float(os.getenv("PLAN_BLEND", "0.75"))

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

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


class ActorCritic(nn.Module):
    def __init__(self, plan_h):
        super().__init__()
        state_dim = BASE_STATE_DIM + plan_h
        actor = [nn.Linear(state_dim, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            actor += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        actor.append(nn.Linear(HIDDEN, 2 * plan_h))
        self.actor = nn.Sequential(*actor)


class Controller(BaseController):
    def __init__(self):
        exp = Path(__file__).parent.parent / "experiments" / "exp074_plan_buffer"
        ckpt = None

        model_env = os.getenv("MODEL", "").strip()
        if model_env:
            p = Path(model_env)
            if not p.exists():
                raise FileNotFoundError(f"MODEL path does not exist: {p}")
            ckpt = str(p)
        else:
            for name in ("best_model.pt", "final_model.pt"):
                p = exp / name
                if p.exists():
                    ckpt = str(p)
                    break

        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {exp}")

        data = torch.load(ckpt, weights_only=False, map_location="cpu")
        self.plan_h = int(data.get("plan_h", PLAN_H_DEFAULT))
        self.plan_commit = int(data.get("plan_commit", PLAN_COMMIT_DEFAULT))
        self.plan_blend = float(data.get("plan_blend", PLAN_BLEND_DEFAULT))
        if not (0 <= self.plan_commit < self.plan_h):
            raise ValueError(f"Invalid plan_commit={self.plan_commit} for plan_h={self.plan_h}")

        self.ac = ActorCritic(self.plan_h)
        self.ac.load_state_dict(data["ac"], strict=False)
        self.ac.eval()

        ds = data.get("delta_scale", None)
        if ds is None:
            best_meta = Path(ckpt).with_name("best_model.pt")
            if best_meta.exists():
                meta = torch.load(best_meta, weights_only=False, map_location="cpu")
                ds = meta.get("delta_scale", None)
        self.delta_scale = float(ds) if ds is not None else DELTA_SCALE_DEFAULT

        self.n = 0
        self._h_act = [0.0] * HIST_LEN
        self._h_lat = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN
        self._plan_raw = [0.0] * self.plan_h
        self._plan_primed = False

    def _push(self, action, current_lataccel):
        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current_lataccel]

    def _proposal_next(self, proposal):
        nxt = np.empty_like(proposal)
        nxt[:-1] = proposal[1:]
        nxt[-1] = proposal[-1]
        return nxt

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan, error_integral):
        prev_act = self._h_act[-1]
        prev_act2 = self._h_act[-2]
        prev_lat = self._h_lat[-1]
        fplan_lat = getattr(future_plan, "lataccel", None)
        fplan_lat0 = fplan_lat[0] if (fplan_lat and len(fplan_lat) > 0) else target_lataccel
        fric = np.sqrt(current_lataccel ** 2 + state.a_ego ** 2) / 7.0
        k_tgt = _curvature(target_lataccel, state.roll_lataccel, state.v_ego)
        k_cur = _curvature(current_lataccel, state.roll_lataccel, state.v_ego)

        core = np.array([
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
        ], dtype=np.float32)

        obs = np.concatenate([
            core,
            np.asarray(self._h_act, np.float32) / S_STEER,
            np.asarray(self._h_lat, np.float32) / S_LAT,
            _future(future_plan, "lataccel", target_lataccel) / S_LAT,
            _future(future_plan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
            _future(future_plan, "v_ego", state.v_ego) / S_VEGO,
            _future(future_plan, "a_ego", state.a_ego) / S_AEGO,
            np.asarray(self._plan_raw, np.float32),
        ])
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

        with torch.no_grad():
            logits = self.ac.actor(torch.from_numpy(obs).unsqueeze(0)).view(1, self.plan_h, 2)
            a_p = F.softplus(logits[..., 0]).squeeze(0).numpy() + 1.0
            b_p = F.softplus(logits[..., 1]).squeeze(0).numpy() + 1.0

        proposal = 2.0 * a_p / (a_p + b_p) - 1.0
        proposal_next = self._proposal_next(proposal)
        raw = float(proposal[0])

        if self._plan_primed:
            shifted = self._plan_raw[1:] + [self._plan_raw[-1]]
            if self.plan_commit < self.plan_h:
                base = np.asarray(shifted[self.plan_commit:], dtype=np.float32)
                tail = proposal_next[self.plan_commit:]
                blended = (1.0 - self.plan_blend) * base + self.plan_blend * tail
                shifted = shifted[:self.plan_commit] + blended.tolist()
            self._plan_raw = shifted
        else:
            self._plan_raw = proposal_next.tolist()
            self._plan_primed = True

        delta = float(np.clip(raw * self.delta_scale, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))

        self._push(action, current_lataccel)
        return action
