# exp052: Beta PPO controller (256-dim, delta actions, deterministic)

import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import (
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    DEL_T,
    State,
    FuturePlan,
    LATACCEL_RANGE,
    MAX_ACC_DELTA,
    LAT_ACCEL_COST_MULTIPLIER,
)

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN   = 256, 256
A_LAYERS            = 4
DELTA_SCALE_DEFAULT = 0.25

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

FUTURE_PLAN_STEPS = 50  # 10 FPS × 5 sec
MPC_SHOOT = os.getenv("MPC_SHOOT", "0") == "1"
MPC_SAMPLES = int(os.getenv("MPC_SAMPLES", "8"))
MPC_HORIZON = int(os.getenv("MPC_HORIZON", "5"))
# STOCH=1: stochastic shooting (Beta sampling + token sampling)
# STOCH=0: deterministic shooting (Beta mean + argmax token)
STOCH = os.getenv("STOCH", "1") == "1"

# obs layout
C     = 16
H1    = C + HIST_LEN
H2    = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V    = F_ROLL + FUTURE_K
F_A    = F_V + FUTURE_K


def _future(fplan, attr, fallback, k=FUTURE_K):
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
        # final_model.pt may contain only weights; recover scale from sibling best_model.pt metadata.
        if ds is None:
            best_meta = Path(ckpt).with_name('best_model.pt')
            if best_meta.exists():
                meta = torch.load(best_meta, weights_only=False, map_location='cpu')
                ds = meta.get('delta_scale', None)
        self.delta_scale = float(ds) if ds is not None else DELTA_SCALE_DEFAULT
        self.n = 0
        self._h_act = [0.0] * HIST_LEN
        self._h_lat = [0.0] * HIST_LEN
        self._h_state = [State(0.0, 0.0, 0.0) for _ in range(HIST_LEN)]
        self._h_error = [0.0] * HIST_LEN
        self.model = None

    def set_model(self, model):
        self.model = model

    def _build_obs(
        self,
        target_lataccel,
        current_lataccel,
        state,
        future_plan,
        h_act,
        h_lat,
        h_error,
    ):
        error = target_lataccel - current_lataccel
        ei = float(np.mean(h_error)) * DEL_T

        v2 = max(state.v_ego**2, 1.0)
        k_tgt = (target_lataccel - state.roll_lataccel) / v2
        k_cur = (current_lataccel - state.roll_lataccel) / v2
        flat = getattr(future_plan, 'lataccel', None)
        fp0 = flat[0] if (flat and len(flat) > 0) else target_lataccel
        fric = np.sqrt(current_lataccel**2 + state.a_ego**2) / 7.0

        ha = np.array(h_act, np.float32)
        hl = np.array(h_lat, np.float32)

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
            ha[-1] / S_STEER,
            ei / S_LAT,
            (fp0 - target_lataccel) / DEL_T / S_LAT,
            (current_lataccel - hl[-1]) / DEL_T / S_LAT,
            (ha[-1] - ha[-2]) / DEL_T / S_STEER,
            fric,
            max(0.0, 1.0 - fric),
        ], dtype=np.float32)

        obs = np.concatenate([
            core,
            ha / S_STEER,
            hl / S_LAT,
            _future(future_plan, 'lataccel', target_lataccel) / S_LAT,
            _future(future_plan, 'roll_lataccel', state.roll_lataccel) / S_ROLL,
            _future(future_plan, 'v_ego', state.v_ego) / S_VEGO,
            _future(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ])
        return np.clip(obs, -5.0, 5.0), error

    def _future_scalar(self, future_plan, attr, idx, fallback):
        vals = getattr(future_plan, attr, None)
        if vals is not None and len(vals) > 0:
            j = idx if idx < len(vals) else (len(vals) - 1)
            return float(vals[j])
        return float(fallback)

    def _sample_raw_batch(self, obs_batch):
        with torch.inference_mode():
            t = torch.from_numpy(obs_batch.astype(np.float32))
            out = self.ac.actor(t)
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            if STOCH:
                x = torch.distributions.Beta(a_p, b_p).sample()
                raw = 2.0 * x - 1.0
            else:
                raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return raw.cpu().numpy().astype(np.float64)

    def _predict_lataccel_batch(self, sim_states, actions, past_preds):
        # Batched TinyPhysics forward: (K, 20, 3), (K, 20), (K, 20)
        bins = self.model.tokenizer.bins
        tokens = np.digitize(np.clip(past_preds, LATACCEL_RANGE[0], LATACCEL_RANGE[1]), bins, right=True)
        states = np.concatenate([actions[..., None], sim_states], axis=2).astype(np.float32)
        inp = {"states": states, "tokens": tokens.astype(np.int64)}
        # Reuse simulator-provided model/session; avoid extra session creation.
        sess = self.model.ort_session
        res = sess.run(None, inp)[0]
        logits = res[:, -1, :]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits / 0.8)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        if STOCH:
            tok = np.array([np.random.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        else:
            tok = np.argmax(probs, axis=1).astype(np.int64)
        return bins[tok]

    def _mpc_shoot_action(self, target_lataccel, current_lataccel, state, future_plan):
        if self.model is None:
            return None
        k = max(1, MPC_SAMPLES)
        h = max(1, MPC_HORIZON)

        # Candidate trajectory state (K copies of controller memory).
        h_act = np.repeat(np.array(self._h_act, np.float64)[None, :], k, axis=0)
        h_lat = np.repeat(np.array(self._h_lat, np.float64)[None, :], k, axis=0)
        h_err = np.repeat(np.array(self._h_error, np.float64)[None, :], k, axis=0)
        h_state = np.repeat(
            np.array([[s.roll_lataccel, s.v_ego, s.a_ego] for s in self._h_state], np.float64)[None, :, :],
            k,
            axis=0,
        )
        cur = np.full(k, float(current_lataccel), dtype=np.float64)
        total = np.zeros(k, dtype=np.float64)
        first_action = None

        for t in range(h):
            tgt = self._future_scalar(future_plan, "lataccel", t - 1, target_lataccel) if t > 0 else float(target_lataccel)
            roll = self._future_scalar(future_plan, "roll_lataccel", t - 1, state.roll_lataccel) if t > 0 else float(state.roll_lataccel)
            vego = self._future_scalar(future_plan, "v_ego", t - 1, state.v_ego) if t > 0 else float(state.v_ego)
            aego = self._future_scalar(future_plan, "a_ego", t - 1, state.a_ego) if t > 0 else float(state.a_ego)
            st = State(roll, vego, aego)
            fp = FuturePlan(
                lataccel=list(getattr(future_plan, "lataccel", []) or []),
                roll_lataccel=list(getattr(future_plan, "roll_lataccel", []) or []),
                v_ego=list(getattr(future_plan, "v_ego", []) or []),
                a_ego=list(getattr(future_plan, "a_ego", []) or []),
            )

            obs_batch = np.empty((k, STATE_DIM), dtype=np.float32)
            for i in range(k):
                h_err[i, :-1] = h_err[i, 1:]
                h_err[i, -1] = tgt - cur[i]
                obs_i, _ = self._build_obs(tgt, cur[i], st, fp, h_act[i], h_lat[i], h_err[i])
                obs_batch[i] = obs_i

            raw = self._sample_raw_batch(obs_batch)
            delta = raw * self.delta_scale
            act = np.clip(h_act[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])
            if t == 0:
                first_action = act.copy()

            # Simulate one step ahead with TinyPhysics model.
            pred = self._predict_lataccel_batch(h_state[:, -CONTEXT_LENGTH:, :], h_act[:, -CONTEXT_LENGTH:], h_lat[:, -CONTEXT_LENGTH:])
            pred = np.clip(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)

            jerk = (pred - cur) / DEL_T
            total += (pred - tgt) ** 2 * (100.0 * LAT_ACCEL_COST_MULTIPLIER) + (jerk ** 2) * 100.0

            # shift histories
            h_act[:, :-1] = h_act[:, 1:]; h_act[:, -1] = act
            h_lat[:, :-1] = h_lat[:, 1:]; h_lat[:, -1] = pred
            h_state[:, :-1, :] = h_state[:, 1:, :]
            h_state[:, -1, 0] = roll
            h_state[:, -1, 1] = vego
            h_state[:, -1, 2] = aego
            cur = pred

        best = int(np.argmin(total))
        return float(first_action[best])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        step_idx = CONTEXT_LENGTH + self.n - 1

        error = target_lataccel - current_lataccel
        self._h_error = self._h_error[1:] + [error]
        ei = float(np.mean(self._h_error)) * DEL_T

        if step_idx < CONTROL_START_IDX:
            self._h_act = self._h_act[1:] + [0.0]
            self._h_lat = self._h_lat[1:] + [current_lataccel]
            self._h_state = self._h_state[1:] + [state]
            return 0.0

        obs, _ = self._build_obs(
            target_lataccel,
            current_lataccel,
            state,
            future_plan,
            self._h_act,
            self._h_lat,
            self._h_error,
        )

        action = None
        if MPC_SHOOT:
            action = self._mpc_shoot_action(target_lataccel, current_lataccel, state, future_plan)
        if action is None:
            with torch.inference_mode():
                t = torch.from_numpy(obs).unsqueeze(0)
                out = self.ac.actor(t)
                a_p = F.softplus(out[..., 0]).item() + 1.0
                b_p = F.softplus(out[..., 1]).item() + 1.0
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
            delta = raw * self.delta_scale
            action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))

        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current_lataccel]
        self._h_state = self._h_state[1:] + [state]
        return action
