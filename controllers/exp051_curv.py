"""exp050: Physics-Aligned PPO controller (256-dim, Beta, ReLU, 4+4 layers, deterministic)
With optional N-step MPC correction via ONNX physics model + future_plan lookahead.
CUDA=1 enables GPU-accelerated MPC (actor + ONNX on GPU, all tensors GPU-resident).
"""

import os
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import (CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, DEL_T,
                          MAX_ACC_DELTA, LATACCEL_RANGE, VOCAB_SIZE,
                          LAT_ACCEL_COST_MULTIPLIER, LataccelTokenizer,
                          State, FuturePlan)

USE_CUDA = os.getenv('CUDA', '0') == '1'
DEV = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
if not USE_CUDA:
    torch.set_num_threads(1)

HIST_LEN    = 20
STATE_DIM   = 156
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

MODEL      = os.getenv('MODEL', '')                   # path to controller weights (fallback: best_model.pt)
LPF_ALPHA  = float(os.getenv('LPF_ALPHA', '0'))      # low-pass: 0=off, 0.15=subtle
MPC        = int(os.getenv('MPC', '0'))               # predict-and-correct via ONNX
MPC_K      = float(os.getenv('MPC_K', '0.2'))         # correction gain
MPC_MAX    = float(os.getenv('MPC_MAX', '0.1'))       # max correction magnitude
MPC_H      = int(os.getenv('MPC_H', '1'))             # lookahead horizon (1=single, 2-5=multi-step)
MPC_ROLL   = int(os.getenv('MPC_ROLL', '0'))          # 1=policy-rolled unrolls, 0=zero-order hold
MPC_N      = int(os.getenv('MPC_N', '0'))             # shooting candidates (0=off, 16=recommended)
RATE_LIMIT = float(os.getenv('RATE_LIMIT', '0'))      # max |Δsteer|/step after all corrections (0=off)
VNORM      = int(os.getenv('VNORM', '0'))             # speed-normalize steer: steer *= v/v_ref



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
        if checkpoint_path is None and MODEL:
            checkpoint_path = MODEL
        if checkpoint_path is None:
            exp = Path(__file__).parent.parent / 'experiments' / 'exp050_rich_ppo'
            for name in ('best_model.pt', 'final_model.pt'):
                p = exp / name
                if p.exists():
                    checkpoint_path = str(p)
                    break
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint in {exp}")

        self.ac = ActorCritic()
        data = torch.load(checkpoint_path, weights_only=False, map_location=DEV)
        self.ac.load_state_dict(data['ac'])
        self.ac.to(DEV).eval()
        self.n = 0
        self._h_act   = [0.0] * HIST_LEN
        self._h_lat   = [0.0] * HIST_LEN
        self._h_v     = [0.0] * HIST_LEN
        self._h_a     = [0.0] * HIST_LEN
        self._h_roll  = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN
        self._sim_model = None    # set via set_model() hook
        self._tokenizer = LataccelTokenizer()
        self._gpu = USE_CUDA and DEV.type == 'cuda'
        self._gpu_ort = None      # GPU ONNX session for MPC (created in set_model)
        self._gpu_bins = None     # tokenizer bins as GPU tensor
        self._gpu_io_cache = {}   # pre-allocated IOBinding buffers

    # ── hook called by TinyPhysicsSimulator.__init__ ──
    def set_model(self, model):
        self._sim_model = model
        if self._gpu and model is not None:
            self._init_gpu_ort(model)

    def _init_gpu_ort(self, model):
        """Create a dedicated GPU ONNX session for MPC and pre-allocate buffers."""
        model_path = os.getenv('ONNX_MODEL', './models/tinyphysics.onnx')
        if not os.path.exists(model_path):
            print(f"[GPU-MPC] ONNX model not found at {model_path}, falling back to CPU")
            self._gpu = False
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        with open(str(model_path), 'rb') as f:
            self._gpu_ort = ort.InferenceSession(f.read(), opts, providers)
        actual = self._gpu_ort.get_providers()
        print(f"[GPU-MPC] providers={actual}")
        self._gpu_out_name = self._gpu_ort.get_outputs()[0].name
        self._gpu_bins = torch.from_numpy(self._tokenizer.bins).to(DEV)  # float64
        self._gpu_bins_f32 = self._gpu_bins.float()

    def _push(self, action, current, state):
        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current]
        self._h_v = self._h_v[1:] + [state.v_ego]
        self._h_a = self._h_a[1:] + [state.a_ego]
        self._h_roll = self._h_roll[1:] + [state.roll_lataccel]

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan, error_integral, h_act, h_lat):
        """Build the 156-dim observation vector."""
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
            (_future_raw(future_plan, 'lataccel', target_lataccel) - _future_raw(future_plan, 'roll_lataccel', state.roll_lataccel)) * 1000.0
                / np.maximum(_future_raw(future_plan, 'v_ego', state.v_ego) ** 2, 1.0) / S_CURV,
            _future_raw(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ])
        return np.clip(obs, -5.0, 5.0)

    def _build_obs_batch(self, target, cur_la, state, future_plan, ei, h_act, h_lat):
        """Vectorised obs for N candidates. cur_la/ei: (N,), h_act/h_lat: (N,20). Returns (N,156)."""
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

        f_lat  = _future_raw(future_plan, 'lataccel', target)
        f_roll = _future_raw(future_plan, 'roll_lataccel', state.roll_lataccel)
        f_v    = _future_raw(future_plan, 'v_ego', state.v_ego)
        f_curv = (f_lat - f_roll) * 1000.0 / np.maximum(f_v ** 2, 1.0) / S_CURV
        fp = np.concatenate([
            f_curv,
            _future_raw(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ])  # (100,) — shared, tile once
        return np.clip(np.concatenate([
            core,
            h_act.astype(np.float32) / S_STEER,
            h_lat.astype(np.float32) / S_LAT,
            np.tile(fp, (N, 1)),
        ], axis=1), -5.0, 5.0)

    def _beta_params(self, obs):
        """Get Beta distribution parameters from the actor."""
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(DEV)
            out = self.ac.actor(t)
            a_p = F.softplus(out[..., 0]).item() + 1.0
            b_p = F.softplus(out[..., 1]).item() + 1.0
        return a_p, b_p

    def _mean_action(self, obs):
        """Deterministic (mean) delta from the Beta policy."""
        a_p, b_p = self._beta_params(obs)
        return 2.0 * a_p / (a_p + b_p) - 1.0

    # ── MPC: predict-and-correct via ONNX with multi-step lookahead ──
    def _mpc_correct(self, action, current_lataccel, state, future_plan):
        """Newton correction validated by N-step ONNX lookahead using future_plan."""
        CL = CONTEXT_LENGTH
        H = MPC_H

        # Shared ONNX context
        h_preds = np.array(self._h_lat + [current_lataccel], np.float64)[-CL:]
        h_states = np.array(
            list(zip(self._h_roll, self._h_v, self._h_a))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)],
            np.float64)[-CL:]
        h_actions = np.array(self._h_act, np.float64)[-CL:]

        # Future targets + states from future_plan
        fp_lat = getattr(future_plan, 'lataccel', []) or []
        fp_roll = getattr(future_plan, 'roll_lataccel', []) or []
        fp_v = getattr(future_plan, 'v_ego', []) or []
        fp_a = getattr(future_plan, 'a_ego', []) or []

        # 1-step Newton correction
        a_seq = np.concatenate([h_actions[1:], [action]])
        pred_1 = self._onnx_expected(a_seq[None], h_states[None], h_preds[None])[0]
        target_0 = fp_lat[0] if len(fp_lat) > 0 else current_lataccel
        x_star = (target_0 + 2.0 * current_lataccel) / 3.0
        da = np.clip(MPC_K * (pred_1 - x_star), -MPC_MAX, MPC_MAX)
        corrected = float(np.clip(action - da, STEER_RANGE[0], STEER_RANGE[1]))

        if H <= 1:
            return corrected

        # Multi-step validation: unroll H steps, compare original vs corrected
        cost_orig = self._unroll_cost(action, h_preds, h_states, h_actions,
                                      current_lataccel, fp_lat, fp_roll, fp_v, fp_a, H)
        cost_corr = self._unroll_cost(corrected, h_preds, h_states, h_actions,
                                      current_lataccel, fp_lat, fp_roll, fp_v, fp_a, H)
        return corrected if cost_corr < cost_orig else float(action)

    def _unroll_cost(self, a_start, h_preds, h_states, h_actions,
                     current_la, fp_lat, fp_roll, fp_v, fp_a, horizon):
        """Unroll ONNX model for `horizon` steps, return total cost.
        MPC_ROLL=1: run policy at steps 1+ for realistic future actions.
        MPC_ROLL=0: hold a_start constant (zero-order hold).
        """
        preds = h_preds.copy()
        sts = h_states.copy()
        acts = h_actions.copy()
        prev_la = current_la
        cost = 0.0
        action = a_start

        # Rolling histories for policy rollout
        if MPC_ROLL:
            r_act = list(self._h_act)
            r_lat = list(self._h_lat)
            r_err = list(self._h_error)

        for step in range(horizon):
            a_seq = np.concatenate([acts[1:], [action]])
            pred_la = self._onnx_expected(a_seq[None], sts[None], preds[None])[0]

            tgt = fp_lat[step] if step < len(fp_lat) else prev_la
            cost += (tgt - pred_la)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            cost += ((pred_la - prev_la) / DEL_T)**2 * 100

            # Shift ONNX context
            preds = np.concatenate([preds[1:], [pred_la]])
            acts = np.concatenate([acts[1:], [action]])
            if step < horizon - 1:
                ns = np.array([
                    fp_roll[step] if step < len(fp_roll) else sts[-1, 0],
                    fp_v[step] if step < len(fp_v) else sts[-1, 1],
                    fp_a[step] if step < len(fp_a) else sts[-1, 2],
                ], np.float64)
                sts = np.concatenate([sts[1:], ns[None]], axis=0)

            # Update rolling histories & compute next action via policy
            if MPC_ROLL:
                r_act = r_act[1:] + [action]
                r_lat = r_lat[1:] + [float(pred_la)]
                r_err = r_err[1:] + [float(tgt - pred_la)]

            prev_la = pred_la

            if MPC_ROLL and step < horizon - 1:
                off = step + 1
                p_tgt = fp_lat[off - 1] if off - 1 < len(fp_lat) else tgt
                p_state = State(
                    roll_lataccel=fp_roll[step] if step < len(fp_roll) else float(sts[-1, 0]),
                    v_ego=fp_v[step] if step < len(fp_v) else float(sts[-1, 1]),
                    a_ego=fp_a[step] if step < len(fp_a) else float(sts[-1, 2]),
                )
                shifted_fp = FuturePlan(
                    lataccel=list(fp_lat[off:]) if off < len(fp_lat) else [],
                    roll_lataccel=list(fp_roll[off:]) if off < len(fp_roll) else [],
                    v_ego=list(fp_v[off:]) if off < len(fp_v) else [],
                    a_ego=list(fp_a[off:]) if off < len(fp_a) else [],
                )
                ei = float(np.mean(r_err)) * DEL_T
                obs = self._build_obs(p_tgt, float(pred_la), p_state,
                                      shifted_fp, ei, r_act, r_lat)
                raw = self._mean_action(obs)
                delta = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
                action = float(np.clip(r_act[-1] + delta, *STEER_RANGE))

        return cost

    # ── Shooting MPC: sample N trajectories from policy, score via ONNX ──
    def _mpc_shoot(self, obs, current_lataccel, state, future_plan):
        if self._gpu and self._gpu_ort is not None:
            return self._mpc_shoot_gpu(obs, current_lataccel, state, future_plan)
        return self._mpc_shoot_cpu(obs, current_lataccel, state, future_plan)

    def _mpc_shoot_gpu(self, obs, current_lataccel, state, future_plan):
        """GPU-accelerated shooting MPC. All arrays as CUDA tensors."""
        N, H, CL = MPC_N, MPC_H, CONTEXT_LENGTH

        # Step-0 candidates: sample N deltas from policy Beta (CPU — tiny N)
        a_p, b_p = self._beta_params(obs)
        samp = np.random.beta(a_p, b_p, size=N)
        samp[0] = a_p / (a_p + b_p)
        deltas = np.clip((2.0 * samp - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        prev_steer = self._h_act[-1]
        actions_0 = torch.from_numpy(
            np.clip(prev_steer + deltas, STEER_RANGE[0], STEER_RANGE[1])
        ).to(DEV, dtype=torch.float64)

        # ONNX context → GPU tensors
        h_pr = torch.tensor(self._h_lat + [current_lataccel], dtype=torch.float64, device=DEV)[-CL:]
        h_st = torch.tensor(
            list(zip(self._h_roll, self._h_v, self._h_a))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)],
            dtype=torch.float64, device=DEV)[-CL:]
        h_ac = torch.tensor(self._h_act, dtype=torch.float64, device=DEV)[-CL:]
        all_pr = h_pr.unsqueeze(0).expand(N, -1).contiguous()       # (N, CL)
        all_st = h_st.unsqueeze(0).expand(N, -1, -1).contiguous()   # (N, CL, 3)
        all_ac = h_ac.unsqueeze(0).expand(N, -1).contiguous()       # (N, CL)

        # Future-plan arrays → GPU
        fp_lat = torch.tensor(getattr(future_plan, 'lataccel', []) or [], dtype=torch.float64, device=DEV)
        fp_roll = torch.tensor(getattr(future_plan, 'roll_lataccel', []) or [], dtype=torch.float64, device=DEV)
        fp_v = torch.tensor(getattr(future_plan, 'v_ego', []) or [], dtype=torch.float64, device=DEV)
        fp_a = torch.tensor(getattr(future_plan, 'a_ego', []) or [], dtype=torch.float64, device=DEV)

        # Rolling histories → GPU (N, 20)
        r_act = torch.tensor(self._h_act, dtype=torch.float64, device=DEV).unsqueeze(0).expand(N, -1).contiguous()
        r_lat = torch.tensor(self._h_lat, dtype=torch.float64, device=DEV).unsqueeze(0).expand(N, -1).contiguous()
        r_err = torch.tensor(self._h_error, dtype=torch.float64, device=DEV).unsqueeze(0).expand(N, -1).contiguous()

        costs = torch.zeros(N, dtype=torch.float64, device=DEV)
        prev_la = torch.full((N,), current_lataccel, dtype=torch.float64, device=DEV)
        cur = actions_0.clone()

        for step in range(H):
            a_seqs = torch.cat([all_ac[:, 1:], cur.unsqueeze(1)], dim=1)
            pred_la = self._onnx_expected_gpu(a_seqs, all_st, all_pr)  # (N,) float32

            # Cost
            tgt_val = fp_lat[step].item() if step < len(fp_lat) else (
                fp_lat[-1].item() if len(fp_lat) > 0 else current_lataccel)
            costs += (tgt_val - pred_la.double()) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            costs += ((pred_la.double() - prev_la) / DEL_T) ** 2 * 100

            # Shift ONNX context
            all_pr = torch.cat([all_pr[:, 1:], pred_la.double().unsqueeze(1)], dim=1)
            all_ac = torch.cat([all_ac[:, 1:], cur.unsqueeze(1)], dim=1)
            if step < H - 1:
                r = fp_roll[step].item() if step < len(fp_roll) else all_st[0, -1, 0].item()
                v = fp_v[step].item() if step < len(fp_v) else all_st[0, -1, 1].item()
                a = fp_a[step].item() if step < len(fp_a) else all_st[0, -1, 2].item()
                ns = torch.tensor([[r, v, a]], dtype=torch.float64, device=DEV).expand(N, 1, 3)
                all_st = torch.cat([all_st[:, 1:, :], ns], dim=1)

            # History shift
            r_act = torch.cat([r_act[:, 1:], cur.unsqueeze(1)], dim=1)
            r_lat = torch.cat([r_lat[:, 1:], pred_la.double().unsqueeze(1)], dim=1)
            r_err = torch.cat([r_err[:, 1:], (tgt_val - pred_la.double()).unsqueeze(1)], dim=1)
            prev_la = pred_la.double()

            # Policy-rolled next actions for steps 1+
            if step < H - 1:
                off = step + 1
                p_tgt = fp_lat[step].item() if step < len(fp_lat) else tgt_val
                p_st = State(
                    roll_lataccel=fp_roll[step].item() if step < len(fp_roll) else all_st[0, -1, 0].item(),
                    v_ego=fp_v[step].item() if step < len(fp_v) else all_st[0, -1, 1].item(),
                    a_ego=fp_a[step].item() if step < len(fp_a) else all_st[0, -1, 2].item())
                sfp = FuturePlan(
                    lataccel=fp_lat[off:].tolist(),
                    roll_lataccel=fp_roll[off:].tolist(),
                    v_ego=fp_v[off:].tolist(),
                    a_ego=fp_a[off:].tolist())

                # Batched obs + actor on GPU
                ei_batch = r_err.mean(dim=1).float() * DEL_T
                obs_batch = self._build_obs_batch_gpu(
                    p_tgt, pred_la.float(), p_st, sfp,
                    ei_batch, r_act.float(), r_lat.float())
                with torch.no_grad():
                    out = self.ac.actor(obs_batch)
                    a_ps = (F.softplus(out[..., 0]) + 1.0)
                    b_ps = (F.softplus(out[..., 1]) + 1.0)
                # Sample on CPU (Beta dist, tiny N)
                s = np.random.beta(a_ps.cpu().numpy(), b_ps.cpu().numpy())
                d = np.clip((2.0 * s - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
                cur = torch.from_numpy(
                    np.clip(r_act[:, -1].cpu().numpy() + d, STEER_RANGE[0], STEER_RANGE[1])
                ).to(DEV, dtype=torch.float64)

        return float(actions_0[costs.argmin()].item())

    def _build_obs_batch_gpu(self, target, cur_la, state, future_plan, ei, h_act, h_lat):
        """GPU vectorised obs for N candidates. cur_la/ei: (N,) GPU, h_act/h_lat: (N,20) GPU.
        Returns (N, 256) float32 GPU tensor."""
        N = cur_la.shape[0]
        k_tgt = _curv(target, state.roll_lataccel, state.v_ego)
        _flat = getattr(future_plan, 'lataccel', None)
        fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target
        v2 = max(state.v_ego ** 2, 1.0)
        error = target - cur_la
        k_cur = (cur_la - state.roll_lataccel) / v2
        fric = torch.sqrt(cur_la ** 2 + state.a_ego ** 2) / 7.0

        core = torch.stack([
            torch.full((N,), target / S_LAT, dtype=torch.float32, device=DEV),
            cur_la / S_LAT,
            error / S_LAT,
            torch.full((N,), k_tgt / S_CURV, dtype=torch.float32, device=DEV),
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            torch.full((N,), state.v_ego / S_VEGO, dtype=torch.float32, device=DEV),
            torch.full((N,), state.a_ego / S_AEGO, dtype=torch.float32, device=DEV),
            torch.full((N,), state.roll_lataccel / S_ROLL, dtype=torch.float32, device=DEV),
            h_act[:, -1] / S_STEER,
            ei / S_LAT,
            torch.full((N,), (fplan_lat0 - target) / DEL_T / S_LAT, dtype=torch.float32, device=DEV),
            (cur_la - h_lat[:, -1]) / DEL_T / S_LAT,
            (h_act[:, -1] - h_act[:, -2]) / DEL_T / S_STEER,
            fric,
            torch.clamp(1.0 - fric, min=0.0),
        ], dim=1)  # (N, 16)

        f_lat  = _future_raw(future_plan, 'lataccel', target)
        f_roll = _future_raw(future_plan, 'roll_lataccel', state.roll_lataccel)
        f_v    = _future_raw(future_plan, 'v_ego', state.v_ego)
        f_curv = (f_lat - f_roll) * 1000.0 / np.maximum(f_v ** 2, 1.0) / S_CURV
        fp = torch.tensor(np.concatenate([
            f_curv,
            _future_raw(future_plan, 'a_ego', state.a_ego) / S_AEGO,
        ]), dtype=torch.float32, device=DEV).unsqueeze(0).expand(N, -1)  # (N, 100)

        obs = torch.cat([core, h_act / S_STEER, h_lat / S_LAT, fp], dim=1)
        return obs.clamp(-5.0, 5.0)

    def _mpc_shoot_cpu(self, obs, current_lataccel, state, future_plan):
        """Original CPU shooting MPC."""
        N, H, CL = MPC_N, MPC_H, CONTEXT_LENGTH

        # Step-0 candidates: sample N deltas from policy Beta
        a_p, b_p = self._beta_params(obs)
        samp = np.random.beta(a_p, b_p, size=N)
        samp[0] = a_p / (a_p + b_p)                              # slot 0 = mean (no-regression)
        deltas = np.clip((2.0 * samp - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        prev_steer = self._h_act[-1]
        actions_0 = np.clip(prev_steer + deltas, STEER_RANGE[0], STEER_RANGE[1])

        # ONNX context tiled for N candidates
        h_pr = np.array(self._h_lat + [current_lataccel], np.float64)[-CL:]
        h_st = np.array(
            list(zip(self._h_roll, self._h_v, self._h_a))
            + [(state.roll_lataccel, state.v_ego, state.a_ego)], np.float64)[-CL:]
        h_ac = np.array(self._h_act, np.float64)[-CL:]
        all_pr = np.tile(h_pr, (N, 1))                           # (N, CL)
        all_st = np.tile(h_st, (N, 1, 1))                        # (N, CL, 3)
        all_ac = np.tile(h_ac, (N, 1))                            # (N, CL)

        # Future-plan arrays
        fp_lat = np.array(getattr(future_plan, 'lataccel', []) or [], np.float64)
        fp_roll = np.array(getattr(future_plan, 'roll_lataccel', []) or [], np.float64)
        fp_v = np.array(getattr(future_plan, 'v_ego', []) or [], np.float64)
        fp_a = np.array(getattr(future_plan, 'a_ego', []) or [], np.float64)

        # Per-candidate rolling histories as (N, 20) arrays
        r_act = np.tile(self._h_act, (N, 1))        # (N, 20)
        r_lat = np.tile(self._h_lat, (N, 1))        # (N, 20)
        r_err = np.tile(self._h_error, (N, 1))      # (N, 20)

        costs = np.zeros(N, np.float64)
        prev_la = np.full(N, current_lataccel, np.float64)
        cur = actions_0.copy()

        for step in range(H):
            # Batched ONNX call (batch=N)
            a_seqs = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            pred_la = self._onnx_expected(a_seqs, all_st, all_pr)  # (N,)

            # Real cost function
            tgt = fp_lat[step] if step < len(fp_lat) else (
                fp_lat[-1] if len(fp_lat) > 0 else current_lataccel)
            costs += (tgt - pred_la)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            costs += ((pred_la - prev_la) / DEL_T)**2 * 100

            # Shift ONNX context
            all_pr = np.concatenate([all_pr[:, 1:], pred_la[:, None]], axis=1)
            all_ac = np.concatenate([all_ac[:, 1:], cur[:, None]], axis=1)
            if step < H - 1:
                r = fp_roll[step] if step < len(fp_roll) else float(all_st[0, -1, 0])
                v = fp_v[step] if step < len(fp_v) else float(all_st[0, -1, 1])
                a = fp_a[step] if step < len(fp_a) else float(all_st[0, -1, 2])
                ns = np.full((N, 1, 3), [r, v, a], np.float64)
                all_st = np.concatenate([all_st[:, 1:, :], ns], axis=1)

            # Vectorised history shift
            r_act = np.concatenate([r_act[:, 1:], cur[:, None]], axis=1)
            r_lat = np.concatenate([r_lat[:, 1:], pred_la[:, None]], axis=1)
            r_err = np.concatenate([r_err[:, 1:], (tgt - pred_la)[:, None]], axis=1)
            prev_la = pred_la.copy()

            # Policy-rolled next actions for steps 1+
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

                # Batched obs + batched actor forward pass
                ei_batch = np.mean(r_err, axis=1).astype(np.float32) * DEL_T
                obs_batch = self._build_obs_batch(
                    p_tgt, pred_la.astype(np.float32), p_st, sfp,
                    ei_batch, r_act.astype(np.float32), r_lat.astype(np.float32))
                with torch.no_grad():
                    out = self.ac.actor(torch.from_numpy(obs_batch).to(DEV))
                    a_ps = (F.softplus(out[..., 0]) + 1.0).cpu().numpy()
                    b_ps = (F.softplus(out[..., 1]) + 1.0).cpu().numpy()
                s = np.random.beta(a_ps, b_ps)
                d = np.clip((2.0 * s - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
                cur = np.clip(r_act[:, -1] + d, STEER_RANGE[0], STEER_RANGE[1])

        return float(actions_0[np.argmin(costs)])

    def _onnx_expected_gpu(self, p_actions, p_states, p_preds):
        """GPU-accelerated batched ONNX call → E[next_lataccel] as GPU tensor.
        Inputs are torch GPU tensors: p_actions (P, CL), p_states (P, CL, 3), p_preds (P, CL)."""
        P, CL = p_actions.shape
        # Tokenize on GPU: clamp + bucketize
        clamped = p_preds.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        tokens = torch.bucketize(clamped, self._gpu_bins_f32, right=False).long()  # (P, CL)

        # Build states: (P, CL, 4) = [actions, roll, v, a]
        states_in = torch.cat([p_actions.unsqueeze(-1).float(), p_states.float()], dim=-1)

        # ONNX via IOBinding
        io = self._gpu_ort.io_binding()
        io.bind_input('states', 'cuda', 0, np.float32, list(states_in.shape),
                       states_in.data_ptr())
        io.bind_input('tokens', 'cuda', 0, np.int64, list(tokens.shape),
                       tokens.data_ptr())
        # Allocate output on GPU
        out_shape = [P, CL, VOCAB_SIZE]
        if not hasattr(self, '_gpu_out') or self._gpu_out.shape[0] < P:
            self._gpu_out = torch.empty(out_shape, dtype=torch.float32, device=DEV)
        out_buf = self._gpu_out[:P]
        io.bind_output(self._gpu_out_name, 'cuda', 0, np.float32, out_shape, out_buf.data_ptr())
        self._gpu_ort.run_with_iobinding(io)

        # Softmax + expected value on GPU
        logits = out_buf[:, -1, :] / 0.8
        probs = torch.softmax(logits, dim=-1)                   # (P, VOCAB)
        return (probs * self._gpu_bins_f32.unsqueeze(0)).sum(dim=-1)  # (P,) E[la]

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
        self._h_error = self._h_error[1:] + [error]
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if self.n <= WARMUP_N:
            self._push(0.0, current_lataccel, state)
            return 0.0

        obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan,
                              error_integral, list(self._h_act), list(self._h_lat))

        if MPC_N > 0 and self._sim_model is not None:
            action = self._mpc_shoot(obs, current_lataccel, state, future_plan)
        else:
            raw = self._mean_action(obs)
            delta = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
            action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))
            if MPC and self._sim_model is not None:
                action = self._mpc_correct(action, current_lataccel, state, future_plan)

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
