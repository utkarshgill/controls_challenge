# exp052: Beta PPO controller (256-dim, delta actions, deterministic)
# MPC shooting ported verbatim from exp050_rich CPU path.

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
    LATACCEL_RANGE,
    MAX_ACC_DELTA,
    LAT_ACCEL_COST_MULTIPLIER,
    VOCAB_SIZE,
    LataccelTokenizer,
    State,
    FuturePlan,
)

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN   = 256, 256
A_LAYERS            = 4
DELTA_SCALE_DEFAULT = 0.25
MAX_DELTA           = 0.5

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

MPC_SHOOT   = os.getenv("MPC_SHOOT", "0") == "1"
MPC_SAMPLES = int(os.getenv("MPC_SAMPLES", "16"))
MPC_HORIZON = int(os.getenv("MPC_HORIZON", "5"))

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
        self._h_v     = [0.0] * HIST_LEN
        self._h_a     = [0.0] * HIST_LEN
        self._h_roll  = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN
        self._sim_model = None
        self._tokenizer = LataccelTokenizer()
        self._mpc_rng = np.random.RandomState(42)

    def set_model(self, model):
        self._sim_model = model

    def _push(self, action, current, state):
        self._h_act  = self._h_act[1:]  + [action]
        self._h_lat  = self._h_lat[1:]  + [current]
        self._h_v    = self._h_v[1:]    + [state.v_ego]
        self._h_a    = self._h_a[1:]    + [state.a_ego]
        self._h_roll = self._h_roll[1:] + [state.roll_lataccel]

    # ── obs builders (identical to exp050) ──────────────────────

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

    def _build_obs_batch(self, target, cur_la, state, future_plan, ei, h_act, h_lat):
        N = len(cur_la)
        k_tgt = _curv(target, state.roll_lataccel, state.v_ego)
        _flat = getattr(future_plan, 'lataccel', None)
        fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target
        v2 = max(state.v_ego ** 2, 1.0)
        error = target - cur_la
        k_cur = (cur_la - state.roll_lataccel) / v2
        fric = np.sqrt(cur_la ** 2 + state.a_ego ** 2) / 7.0

        core = np.column_stack([
            np.full(N, target / S_LAT, np.float32),
            cur_la / S_LAT,
            error / S_LAT,
            np.full(N, k_tgt / S_CURV, np.float32),
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            np.full(N, state.v_ego / S_VEGO, np.float32),
            np.full(N, state.a_ego / S_AEGO, np.float32),
            np.full(N, state.roll_lataccel / S_ROLL, np.float32),
            h_act[:, -1] / S_STEER,
            ei / S_LAT,
            np.full(N, (fplan_lat0 - target) / DEL_T / S_LAT, np.float32),
            (cur_la - h_lat[:, -1]) / DEL_T / S_LAT,
            (h_act[:, -1] - h_act[:, -2]) / DEL_T / S_STEER,
            fric,
            np.maximum(0.0, 1.0 - fric),
        ]).astype(np.float32)

        fp = np.concatenate([
            _future(future_plan, 'lataccel', target) / S_LAT,
            _future(future_plan, 'roll_lataccel', state.roll_lataccel) / S_ROLL,
            _future(future_plan, 'v_ego', state.v_ego) / S_VEGO,
            _future(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ])
        return np.clip(np.concatenate([
            core,
            h_act.astype(np.float32) / S_STEER,
            h_lat.astype(np.float32) / S_LAT,
            np.tile(fp, (N, 1)),
        ], axis=1), -5.0, 5.0)

    # ── policy helpers ──────────────────────────────────────────

    def _beta_params(self, obs):
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0)
            out = self.ac.actor(t)
            a_p = F.softplus(out[..., 0]).item() + 1.0
            b_p = F.softplus(out[..., 1]).item() + 1.0
        return a_p, b_p

    def _mean_action(self, obs):
        a_p, b_p = self._beta_params(obs)
        return 2.0 * a_p / (a_p + b_p) - 1.0

    # ── ONNX expected value (verbatim from exp050) ──────────────

    def _onnx_expected(self, p_actions, p_states, p_preds):
        tok = self._tokenizer
        tokenized = tok.encode(p_preds)
        states_in = np.concatenate(
            [p_actions[:, :, None], p_states], axis=-1)
        res = self._sim_model.ort_session.run(None, {
            'states': states_in.astype(np.float32),
            'tokens': tokenized.astype(np.int64),
        })[0]
        logits = res[:, -1, :] / 0.8
        e_x = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e_x / e_x.sum(axis=-1, keepdims=True)
        return np.sum(probs * tok.bins[None, :], axis=-1)

    # ── CPU shooting MPC (verbatim from exp050) ─────────────────

    def _mpc_shoot(self, obs, current_lataccel, state, future_plan):
        N, H, CL = MPC_SAMPLES, MPC_HORIZON, CONTEXT_LENGTH
        DS = self.delta_scale

        a_p, b_p = self._beta_params(obs)
        samp = self._mpc_rng.beta(a_p, b_p, size=N)
        samp[0] = a_p / (a_p + b_p)
        deltas = np.clip((2.0 * samp - 1.0) * DS, -MAX_DELTA, MAX_DELTA)
        prev_steer = self._h_act[-1]
        actions_0 = np.clip(prev_steer + deltas, STEER_RANGE[0], STEER_RANGE[1])

        h_pr = np.array(self._h_lat + [current_lataccel], np.float64)[-CL:]
        h_st = np.array(
            list(zip(self._h_roll, self._h_v, self._h_a))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)], np.float64)[-CL:]
        h_ac = np.array(self._h_act, np.float64)[-CL:]
        all_pr = np.tile(h_pr, (N, 1))
        all_st = np.tile(h_st, (N, 1, 1))
        all_ac = np.tile(h_ac, (N, 1))

        fp_lat  = np.array(getattr(future_plan, 'lataccel', []) or [], np.float64)
        fp_roll = np.array(getattr(future_plan, 'roll_lataccel', []) or [], np.float64)
        fp_v    = np.array(getattr(future_plan, 'v_ego', []) or [], np.float64)
        fp_a    = np.array(getattr(future_plan, 'a_ego', []) or [], np.float64)

        r_act = np.tile(self._h_act, (N, 1))
        r_lat = np.tile(self._h_lat, (N, 1))
        r_err = np.tile(self._h_error, (N, 1))

        costs = np.zeros(N, np.float64)
        prev_la = np.full(N, current_lataccel, np.float64)
        cur = actions_0.copy()

        for step in range(H):
            a_seqs = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            pred_la = self._onnx_expected(a_seqs, all_st, all_pr)

            tgt = fp_lat[step] if step < len(fp_lat) else (
                fp_lat[-1] if len(fp_lat) > 0 else current_lataccel)
            costs += (tgt - pred_la)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            costs += ((pred_la - prev_la) / DEL_T)**2 * 100

            all_pr = np.concatenate([all_pr[:, 1:], pred_la[:, None]], axis=1)
            all_ac = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            if step < H - 1:
                r = fp_roll[step] if step < len(fp_roll) else float(all_st[0, -1, 0])
                v = fp_v[step]    if step < len(fp_v)    else float(all_st[0, -1, 1])
                a = fp_a[step]    if step < len(fp_a)    else float(all_st[0, -1, 2])
                ns = np.full((N, 1, 3), [r, v, a], np.float64)
                all_st = np.concatenate([all_st[:, 1:, :], ns], axis=1)

            r_act = np.concatenate([r_act[:, 1:], cur[:, None]], axis=1)
            r_lat = np.concatenate([r_lat[:, 1:], pred_la[:, None]], axis=1)
            r_err = np.concatenate([r_err[:, 1:], (tgt - pred_la)[:, None]], axis=1)
            prev_la = pred_la.copy()

            if step < H - 1:
                off = step + 1
                p_tgt = float(fp_lat[step]) if step < len(fp_lat) else float(tgt)
                p_st = State(
                    roll_lataccel=float(fp_roll[step]) if step < len(fp_roll) else float(all_st[0, -1, 0]),
                    v_ego=float(fp_v[step]) if step < len(fp_v) else float(all_st[0, -1, 1]),
                    a_ego=float(fp_a[step]) if step < len(fp_a) else float(all_st[0, -1, 2]))
                sfp = FuturePlan(
                    lataccel=list(fp_lat[off:]),
                    roll_lataccel=list(fp_roll[off:]),
                    v_ego=list(fp_v[off:]),
                    a_ego=list(fp_a[off:]))

                ei_batch = np.mean(r_err, axis=1).astype(np.float32) * DEL_T
                obs_batch = self._build_obs_batch(
                    p_tgt, pred_la.astype(np.float32), p_st, sfp,
                    ei_batch, r_act.astype(np.float32), r_lat.astype(np.float32))
                with torch.no_grad():
                    out = self.ac.actor(torch.from_numpy(obs_batch))
                    a_ps = (F.softplus(out[..., 0]) + 1.0).cpu().numpy()
                    b_ps = (F.softplus(out[..., 1]) + 1.0).cpu().numpy()
                s = self._mpc_rng.beta(a_ps, b_ps)
                d = np.clip((2.0 * s - 1.0) * DS, -MAX_DELTA, MAX_DELTA)
                cur = np.clip(r_act[:, -1] + d, STEER_RANGE[0], STEER_RANGE[1])

        return float(actions_0[np.argmin(costs)])

    # ── update (main entry) ─────────────────────────────────────

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        WARMUP_N = CONTROL_START_IDX - CONTEXT_LENGTH

        error = target_lataccel - current_lataccel
        self._h_error = self._h_error[1:] + [error]
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if self.n <= WARMUP_N:
            self._push(0.0, current_lataccel, state)
            return 0.0

        obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan,
                              error_integral, list(self._h_act), list(self._h_lat))

        if MPC_SHOOT and self._sim_model is not None:
            action = self._mpc_shoot(obs, current_lataccel, state, future_plan)
        else:
            raw = self._mean_action(obs)
            delta = float(np.clip(raw * self.delta_scale, -MAX_DELTA, MAX_DELTA))
            action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))

        self._push(action, current_lataccel, state)
        return action
