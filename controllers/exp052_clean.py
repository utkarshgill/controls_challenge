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

    def _build_obs_batch(self, target, cur_la, state, future_plan, ei, h_act, h_lat):
        """Vectorized 256-dim obs for K candidates."""
        k = len(cur_la)
        v2 = max(state.v_ego ** 2, 1.0)
        k_tgt = (target - state.roll_lataccel) / v2
        k_cur = (cur_la - state.roll_lataccel) / v2
        flat = getattr(future_plan, "lataccel", None)
        fp0 = flat[0] if (flat and len(flat) > 0) else target
        fric = np.sqrt(cur_la ** 2 + state.a_ego ** 2) / 7.0
        error = target - cur_la

        core = np.column_stack([
            np.full(k, target / S_LAT, np.float32),
            cur_la / S_LAT,
            error / S_LAT,
            np.full(k, k_tgt / S_CURV, np.float32),
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            np.full(k, state.v_ego / S_VEGO, np.float32),
            np.full(k, state.a_ego / S_AEGO, np.float32),
            np.full(k, state.roll_lataccel / S_ROLL, np.float32),
            h_act[:, -1] / S_STEER,
            ei / S_LAT,
            np.full(k, (fp0 - target) / DEL_T / S_LAT, np.float32),
            (cur_la - h_lat[:, -1]) / DEL_T / S_LAT,
            (h_act[:, -1] - h_act[:, -2]) / DEL_T / S_STEER,
            fric,
            np.maximum(0.0, 1.0 - fric),
        ]).astype(np.float32)

        fp = np.concatenate([
            _future(future_plan, "lataccel", target) / S_LAT,
            _future(future_plan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
            _future(future_plan, "v_ego", state.v_ego) / S_VEGO,
            _future(future_plan, "a_ego", state.a_ego) / S_AEGO,
        ])
        obs = np.concatenate([
            core,
            h_act.astype(np.float32) / S_STEER,
            h_lat.astype(np.float32) / S_LAT,
            np.tile(fp, (k, 1)),
        ], axis=1)
        return np.clip(obs, -5.0, 5.0)

    def _sample_raw_batch(self, obs_batch):
        """Stochastic Beta sample for all K candidates; slot 0 = mean (no-regression)."""
        with torch.inference_mode():
            t = torch.from_numpy(obs_batch.astype(np.float32))
            out = self.ac.actor(t)
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            x = torch.distributions.Beta(a_p, b_p).sample()
            x[0] = a_p[0] / (a_p[0] + b_p[0])
            raw = 2.0 * x - 1.0
        return raw.cpu().numpy().astype(np.float64)

    def _predict_lataccel_batch(self, sim_states, actions, past_preds):
        bins = self.model.tokenizer.bins
        tokens = np.digitize(np.clip(past_preds, LATACCEL_RANGE[0], LATACCEL_RANGE[1]), bins, right=True)
        states = np.concatenate([actions[..., None], sim_states], axis=2).astype(np.float32)
        sess = self.model.ort_session
        res = sess.run(None, {"states": states, "tokens": tokens.astype(np.int64)})[0]
        logits = res[:, -1, :] / 0.8
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        return np.sum(probs * bins[None, :], axis=1)

    def _mpc_shoot_action(self, target_lataccel, current_lataccel, state, future_plan):
        if self.model is None:
            return None
        k = max(1, MPC_SAMPLES)
        h = max(1, MPC_HORIZON)
        cl = CONTEXT_LENGTH

        # Step-0 candidates from current-policy obs (same as exp050 shooting style).
        obs0, _ = self._build_obs(
            target_lataccel, current_lataccel, state, future_plan,
            self._h_act, self._h_lat, self._h_error,
        )
        obs0_batch = np.repeat(obs0[None, :], k, axis=0)
        raw0 = self._sample_raw_batch(obs0_batch)
        d0 = raw0 * self.delta_scale
        prev_steer = self._h_act[-1]
        actions_0 = np.clip(prev_steer + d0, STEER_RANGE[0], STEER_RANGE[1])

        # ONNX context tiled for K candidates.
        h_pr = np.array(self._h_lat + [current_lataccel], np.float64)[-cl:]
        h_st = np.array(
            list(zip([s.roll_lataccel for s in self._h_state],
                     [s.v_ego for s in self._h_state],
                     [s.a_ego for s in self._h_state]))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)], np.float64
        )[-cl:]
        h_ac = np.array(self._h_act, np.float64)[-cl:]
        all_pr = np.tile(h_pr, (k, 1))
        all_st = np.tile(h_st, (k, 1, 1))
        all_ac = np.tile(h_ac, (k, 1))

        fp_lat = np.array(getattr(future_plan, "lataccel", []) or [], np.float64)
        fp_roll = np.array(getattr(future_plan, "roll_lataccel", []) or [], np.float64)
        fp_v = np.array(getattr(future_plan, "v_ego", []) or [], np.float64)
        fp_a = np.array(getattr(future_plan, "a_ego", []) or [], np.float64)

        r_act = np.tile(self._h_act, (k, 1))
        r_lat = np.tile(self._h_lat, (k, 1))
        r_err = np.tile(self._h_error, (k, 1))

        costs = np.zeros(k, np.float64)
        prev_la = np.full(k, current_lataccel, np.float64)
        cur = actions_0.copy()
        first_action = actions_0.copy()

        for step in range(h):
            # Use candidate actions in ONNX sequence (fixes stale-action simulation).
            a_seqs = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            pred = self._predict_lataccel_batch(all_st, a_seqs, all_pr)
            pred = np.clip(pred, prev_la - MAX_ACC_DELTA, prev_la + MAX_ACC_DELTA)

            tgt = fp_lat[step] if step < len(fp_lat) else (fp_lat[-1] if len(fp_lat) > 0 else current_lataccel)
            costs += (tgt - pred) ** 2 * (100.0 * LAT_ACCEL_COST_MULTIPLIER)
            costs += ((pred - prev_la) / DEL_T) ** 2 * 100.0

            all_pr = np.concatenate([all_pr[:, 1:], pred[:, None]], axis=1)
            all_ac = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            if step < h - 1:
                r = fp_roll[step] if step < len(fp_roll) else float(all_st[0, -1, 0])
                v = fp_v[step] if step < len(fp_v) else float(all_st[0, -1, 1])
                a = fp_a[step] if step < len(fp_a) else float(all_st[0, -1, 2])
                ns = np.full((k, 1, 3), [r, v, a], np.float64)
                all_st = np.concatenate([all_st[:, 1:, :], ns], axis=1)

            r_act = np.concatenate([r_act[:, 1:], cur[:, None]], axis=1)
            r_lat = np.concatenate([r_lat[:, 1:], pred[:, None]], axis=1)
            r_err = np.concatenate([r_err[:, 1:], (tgt - pred)[:, None]], axis=1)
            prev_la = pred.copy()

            if step < h - 1:
                off = step + 1
                p_tgt = float(fp_lat[step]) if step < len(fp_lat) else float(tgt)
                p_st = State(
                    roll_lataccel=float(fp_roll[step]) if step < len(fp_roll) else float(all_st[0, -1, 0]),
                    v_ego=float(fp_v[step]) if step < len(fp_v) else float(all_st[0, -1, 1]),
                    a_ego=float(fp_a[step]) if step < len(fp_a) else float(all_st[0, -1, 2]),
                )
                sfp = FuturePlan(
                    lataccel=list(fp_lat[off:]),
                    roll_lataccel=list(fp_roll[off:]),
                    v_ego=list(fp_v[off:]),
                    a_ego=list(fp_a[off:]),
                )
                ei_batch = np.mean(r_err, axis=1).astype(np.float32) * DEL_T
                obs_batch = self._build_obs_batch(
                    p_tgt, pred.astype(np.float32), p_st, sfp,
                    ei_batch, r_act.astype(np.float32), r_lat.astype(np.float32),
                )
                raw = self._sample_raw_batch(obs_batch)
                delta = raw * self.delta_scale
                cur = np.clip(r_act[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])

        return float(first_action[int(np.argmin(costs))])

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
