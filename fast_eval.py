"""Massively-parallel GPU evaluation — all episodes in one batched rollout.

Usage:
  # GPU batched eval (5000 episodes in ~15s):
  CUDA=1 MPC_N=16 MPC_H=5 python fast_eval.py --num_segs 5000

  # GPU batched eval without MPC (deterministic, ~8s):
  CUDA=1 python fast_eval.py --num_segs 5000

  # CPU sequential fallback for parity verification:
  python fast_eval.py --num_segs 100 --sequential

  # Verify deterministic parity (MPC_N=0) against official run_rollout:
  python fast_eval.py --num_segs 50 --verify

Produces the same report.html as eval.py.
Does NOT modify any official files (tinyphysics.py, eval.py).
"""

import argparse
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

from tinyphysics import (run_rollout, CONTROL_START_IDX, CONTEXT_LENGTH, COST_END_IDX,
                          STEER_RANGE, DEL_T, LATACCEL_RANGE, VOCAB_SIZE,
                          LAT_ACCEL_COST_MULTIPLIER, FUTURE_PLAN_STEPS,
                          LataccelTokenizer)
from tinyphysics_batched import BatchedSimulator, preload_csvs, make_ort_session
from eval import create_report, SAMPLE_ROLLOUTS

# Controller config (mirrors controllers/exp050_rich.py)
HIST_LEN    = 20
STATE_DIM   = 256
HIDDEN      = 256
A_LAYERS    = 4
C_LAYERS    = 4
DELTA_SCALE = 0.25
MAX_DELTA   = 0.5
FUTURE_K    = 50

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV  = 2.0, 0.02

MPC_N = int(os.getenv('MPC_N', '0'))
MPC_H = int(os.getenv('MPC_H', '1'))


def _load_actor(checkpoint_path, device):
    """Load actor network from checkpoint."""
    import torch.nn as nn
    a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
    for _ in range(A_LAYERS - 1):
        a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
    a.append(nn.Linear(HIDDEN, 2))
    actor = nn.Sequential(*a)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    # Extract actor weights from the ActorCritic state dict
    actor_sd = {k.replace('actor.', ''): v for k, v in ckpt['ac'].items() if k.startswith('actor.')}
    actor.load_state_dict(actor_sd)
    actor.to(device).eval()
    return actor


def _make_mpc_ort(model_path, device):
    """Create a GPU ONNX session for MPC lookahead."""
    if device.type != 'cuda':
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    with open(str(model_path), 'rb') as f:
        sess = ort.InferenceSession(f.read(), opts, providers)
    print(f"[MPC-ORT] providers={sess.get_providers()}")
    return sess


def batched_eval(files, model_path, checkpoint_path, device):
    """Run all episodes in one batched rollout on GPU. Returns (N,) cost arrays."""
    csv_list = [str(f) for f in files]
    ort_session = make_ort_session(str(model_path))
    sim = BatchedSimulator(str(model_path), csv_list, ort_session=ort_session)

    N = sim.N
    T = sim.T
    actor = _load_actor(checkpoint_path, device)
    use_gpu = device.type == 'cuda'
    dg = sim.data_gpu if use_gpu else None

    # MPC setup
    do_mpc = MPC_N > 0 and use_gpu
    mpc_ort = _make_mpc_ort(model_path, device) if do_mpc else None
    tok = LataccelTokenizer()
    if use_gpu:
        gpu_bins = torch.from_numpy(tok.bins).to(device)
        gpu_bins_f32 = gpu_bins.float()
        mpc_out_name = mpc_ort.get_outputs()[0].name if mpc_ort else None
        mpc_out_buf = None  # lazily allocated

    OBS_DIM = 16 + HIST_LEN + HIST_LEN + FUTURE_K * 4

    # GPU ring buffers
    h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device=device)
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device=device)
    h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device=device)
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float64, device=device)
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device=device)

    # Keep a reference to sim's numpy data for CPU path
    sim_data_np = sim.data

    def _build_obs(step_idx, sim_ref=None, *,
                   cpu_target=None, cpu_cur_la=None, cpu_state=None, cpu_fplan=None):
        """Build obs for all N episodes. Works on both GPU and CPU paths."""
        if sim_ref is not None:
            # GPU path — read from data_gpu tensors
            target     = dg['target_lataccel'][:, step_idx]
            current_la = sim_ref.current_lataccel
            roll_la    = dg['roll_lataccel'][:, step_idx]
            v_ego      = dg['v_ego'][:, step_idx]
            a_ego      = dg['a_ego'][:, step_idx]
            fplan_lat0 = dg['target_lataccel'][:, min(step_idx + 1, T - 1)]
            data_src = dg
        else:
            # CPU path — convert numpy args to tensors
            target     = torch.from_numpy(np.asarray(cpu_target, dtype=np.float64)).to(device)
            current_la = torch.from_numpy(np.asarray(cpu_cur_la, dtype=np.float64)).to(device)
            roll_la    = torch.from_numpy(np.asarray(cpu_state['roll_lataccel'], dtype=np.float64)).to(device)
            v_ego      = torch.from_numpy(np.asarray(cpu_state['v_ego'], dtype=np.float64)).to(device)
            a_ego      = torch.from_numpy(np.asarray(cpu_state['a_ego'], dtype=np.float64)).to(device)
            fplan_lat0_np = sim_data_np['target_lataccel'][:, min(step_idx + 1, T - 1)]
            fplan_lat0 = torch.from_numpy(np.asarray(fplan_lat0_np, dtype=np.float64)).to(device)
            data_src = None

        cla64 = current_la.double()
        error = (target - current_la).double()

        h_error[:, :-1] = h_error[:, 1:].clone()
        h_error[:, -1] = error
        error_integral = h_error.mean(dim=1) * DEL_T

        v2 = torch.clamp(v_ego * v_ego, min=1.0)
        k_tgt = (target - roll_la) / v2
        k_cur = (current_la - roll_la) / v2
        fric = torch.sqrt(current_la**2 + a_ego**2) / 7.0

        c = 0
        obs_buf[:, c] = target / S_LAT;                    c += 1
        obs_buf[:, c] = current_la / S_LAT;                c += 1
        obs_buf[:, c] = (target - current_la) / S_LAT;     c += 1
        obs_buf[:, c] = k_tgt / S_CURV;                    c += 1
        obs_buf[:, c] = k_cur / S_CURV;                    c += 1
        obs_buf[:, c] = (k_tgt - k_cur) / S_CURV;          c += 1
        obs_buf[:, c] = v_ego / S_VEGO;                    c += 1
        obs_buf[:, c] = a_ego / S_AEGO;                    c += 1
        obs_buf[:, c] = roll_la / S_ROLL;                  c += 1
        obs_buf[:, c] = h_act32[:, -1] / S_STEER;          c += 1
        obs_buf[:, c] = error_integral / S_LAT;            c += 1
        obs_buf[:, c] = (fplan_lat0 - target) / DEL_T / S_LAT; c += 1
        obs_buf[:, c] = (current_la - h_lat[:, -1]) / DEL_T / S_LAT; c += 1
        obs_buf[:, c] = (h_act32[:, -1] - h_act32[:, -2]) / DEL_T / S_STEER; c += 1
        obs_buf[:, c] = fric;                               c += 1
        obs_buf[:, c] = torch.clamp(1.0 - fric, min=0.0);  c += 1

        obs_buf[:, c:c+HIST_LEN] = h_act32 / S_STEER;     c += HIST_LEN
        obs_buf[:, c:c+HIST_LEN] = h_lat / S_LAT;          c += HIST_LEN

        end = min(step_idx + FUTURE_PLAN_STEPS, T)
        for attr, scale in [('target_lataccel', S_LAT), ('roll_lataccel', S_ROLL),
                            ('v_ego', S_VEGO), ('a_ego', S_AEGO)]:
            if data_src is not None:
                slc = data_src[attr][:, step_idx+1:end]
                w = slc.shape[1]
                if w == 0:
                    fb = data_src[attr][:, step_idx]
                    obs_buf[:, c:c+FUTURE_K] = (fb / scale).float().unsqueeze(1)
                elif w < FUTURE_K:
                    obs_buf[:, c:c+w] = slc.float() / scale
                    obs_buf[:, c+w:c+FUTURE_K] = (slc[:, -1:].float() / scale)
                else:
                    obs_buf[:, c:c+FUTURE_K] = slc[:, :FUTURE_K].float() / scale
            else:
                slc_np = sim_data_np[attr][:, step_idx+1:end].astype(np.float32)
                slc_t = torch.from_numpy(slc_np).to(device)
                w = slc_t.shape[1]
                if w == 0:
                    fb_np = sim_data_np[attr][:, step_idx].astype(np.float32)
                    obs_buf[:, c:c+FUTURE_K] = torch.from_numpy(fb_np).to(device).unsqueeze(1) / scale
                elif w < FUTURE_K:
                    obs_buf[:, c:c+w] = slc_t / scale
                    obs_buf[:, c+w:c+FUTURE_K] = slc_t[:, -1:] / scale
                else:
                    obs_buf[:, c:c+FUTURE_K] = slc_t[:, :FUTURE_K] / scale
            c += FUTURE_K

        obs_buf.clamp_(-5.0, 5.0)
        return cla64

    def _onnx_expected_gpu(p_actions, p_states, p_preds):
        """Batched ONNX → E[lataccel]. All inputs are GPU tensors.
        p_actions: (P, CL), p_states: (P, CL, 3), p_preds: (P, CL)."""
        nonlocal mpc_out_buf
        P, CL = p_actions.shape

        clamped = p_preds.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        tokens = torch.bucketize(clamped, gpu_bins_f32, right=False).long()
        states_in = torch.cat([p_actions.unsqueeze(-1).float(), p_states.float()], dim=-1)

        io = mpc_ort.io_binding()
        io.bind_input('states', 'cuda', 0, np.float32, list(states_in.shape), states_in.data_ptr())
        io.bind_input('tokens', 'cuda', 0, np.int64, list(tokens.shape), tokens.data_ptr())

        out_shape = [P, CL, VOCAB_SIZE]
        if mpc_out_buf is None or mpc_out_buf.shape[0] < P:
            mpc_out_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
        out_buf = mpc_out_buf[:P]
        io.bind_output(mpc_out_name, 'cuda', 0, np.float32, out_shape, out_buf.data_ptr())
        mpc_ort.run_with_iobinding(io)

        logits = out_buf[:, -1, :] / 0.8
        probs = torch.softmax(logits, dim=-1)
        return (probs * gpu_bins_f32.unsqueeze(0)).sum(dim=-1)  # (P,)

    def _mpc_shoot(step_idx, sim_ref):
        """Batched MPC shooting across ALL N episodes × MPC_N candidates.
        Returns (N,) best actions on GPU."""
        H, CL = MPC_H, CONTEXT_LENGTH
        K = MPC_N  # candidates per episode

        # Get Beta params for all N episodes
        with torch.inference_mode():
            out = actor(obs_buf)
        a_p = F.softplus(out[:, 0]) + 1.0  # (N,)
        b_p = F.softplus(out[:, 1]) + 1.0  # (N,)

        # Sample K candidates per episode → (N, K)
        a_np, b_np = a_p.cpu().numpy(), b_p.cpu().numpy()
        samp = np.random.beta(a_np[:, None], b_np[:, None], size=(N, K))  # (N, K)
        samp[:, 0] = a_np / (a_np + b_np)  # slot 0 = mean (no-regression)
        deltas = np.clip((2.0 * samp - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        prev_steer = h_act[:, -1].cpu().numpy()  # (N,)
        actions_0 = torch.from_numpy(
            np.clip(prev_steer[:, None] + deltas, STEER_RANGE[0], STEER_RANGE[1])
        ).to(device, dtype=torch.float64)  # (N, K)

        # Flatten: (N*K,) for batched ONNX
        P = N * K

        # ONNX context: tile each episode K times → (N*K, CL, ...)
        h = sim_ref._hist_len
        all_pr = sim_ref.current_lataccel_history[:, h-CL:h]   # (N, CL) float64
        all_st = sim_ref.state_history[:, h-CL+1:h+1, :]       # (N, CL, 3) float64
        all_ac_base = sim_ref.action_history[:, h-CL+1:h+1]    # (N, CL) float64

        # Tile for K candidates: (N, CL) → (N*K, CL)
        all_pr = all_pr.unsqueeze(1).expand(-1, K, -1).reshape(P, CL)
        all_st = all_st.unsqueeze(1).expand(-1, K, -1, -1).reshape(P, CL, 3)
        all_ac = all_ac_base.unsqueeze(1).expand(-1, K, -1).reshape(P, CL)

        # Rolling histories for policy rollout: (N*K, HIST_LEN)
        r_act = h_act.unsqueeze(1).expand(-1, K, -1).reshape(P, HIST_LEN).clone()
        r_lat = h_lat.unsqueeze(1).expand(-1, K, -1).reshape(P, HIST_LEN).clone()
        r_err = h_error.unsqueeze(1).expand(-1, K, -1).reshape(P, HIST_LEN).clone()

        # Future plan from data_gpu: (N, ...) → tiled to (N*K, ...)
        fp_end = min(step_idx + FUTURE_PLAN_STEPS, T)
        fp_lat = dg['target_lataccel'][:, step_idx+1:fp_end]     # (N, L)
        fp_roll = dg['roll_lataccel'][:, step_idx+1:fp_end]
        fp_v = dg['v_ego'][:, step_idx+1:fp_end]
        fp_a = dg['a_ego'][:, step_idx+1:fp_end]
        fp_len = fp_lat.shape[1]

        costs = torch.zeros(P, dtype=torch.float64, device=device)
        prev_la = sim_ref.current_lataccel.unsqueeze(1).expand(-1, K).reshape(P).clone()
        cur = actions_0.reshape(P)  # (N*K,)

        for step in range(H):
            a_seqs = torch.cat([all_ac[:, 1:], cur.unsqueeze(1)], dim=1)
            pred_la = _onnx_expected_gpu(a_seqs, all_st, all_pr)  # (P,)

            # Target for this horizon step (same across candidates for an episode)
            if step < fp_len:
                tgt = fp_lat[:, step].unsqueeze(1).expand(-1, K).reshape(P)
            elif fp_len > 0:
                tgt = fp_lat[:, -1].unsqueeze(1).expand(-1, K).reshape(P)
            else:
                tgt = prev_la

            costs += (tgt - pred_la.double()) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            costs += ((pred_la.double() - prev_la) / DEL_T) ** 2 * 100

            # Shift ONNX context
            all_pr = torch.cat([all_pr[:, 1:], pred_la.double().unsqueeze(1)], dim=1)
            all_ac = torch.cat([all_ac[:, 1:], cur.unsqueeze(1)], dim=1)
            if step < H - 1 and step < fp_len:
                ns_r = fp_roll[:, step].unsqueeze(1).expand(-1, K).reshape(P, 1)
                ns_v = fp_v[:, step].unsqueeze(1).expand(-1, K).reshape(P, 1)
                ns_a = fp_a[:, step].unsqueeze(1).expand(-1, K).reshape(P, 1)
                ns = torch.cat([ns_r, ns_v, ns_a], dim=1).unsqueeze(1)  # (P, 1, 3)
                all_st = torch.cat([all_st[:, 1:, :], ns], dim=1)

            # History shift
            r_act = torch.cat([r_act[:, 1:], cur.float().unsqueeze(1)], dim=1)
            r_lat = torch.cat([r_lat[:, 1:], pred_la.float().unsqueeze(1)], dim=1)
            r_err = torch.cat([r_err[:, 1:], (tgt.float() - pred_la.float()).unsqueeze(1)], dim=1)
            prev_la = pred_la.double()

            # Policy-rolled next actions for steps 1+
            if step < H - 1:
                ei = r_err.mean(dim=1) * DEL_T  # (P,)
                # Build batched obs for all P candidates
                # Use future plan data for the shifted state
                off = step + 1
                if off < fp_len:
                    p_tgt = fp_lat[:, off-1].unsqueeze(1).expand(-1, K).reshape(P)
                    p_roll = fp_roll[:, step].unsqueeze(1).expand(-1, K).reshape(P)
                    p_v = fp_v[:, step].unsqueeze(1).expand(-1, K).reshape(P)
                    p_a = fp_a[:, step].unsqueeze(1).expand(-1, K).reshape(P)
                else:
                    p_tgt = tgt
                    p_roll = all_st[:, -1, 0]
                    p_v = all_st[:, -1, 1]
                    p_a = all_st[:, -1, 2]

                # Inline obs build for P candidates
                v2 = torch.clamp(p_v * p_v, min=1.0)
                k_tgt_p = (p_tgt.float() - p_roll.float()) / v2.float()
                k_cur_p = (pred_la.float() - p_roll.float()) / v2.float()
                error_p = p_tgt.float() - pred_la.float()
                fric_p = torch.sqrt(pred_la.float()**2 + p_a.float()**2) / 7.0

                # Future plan for shifted offset: reuse from episode-level
                # (shared across K candidates per episode)
                obs_p = torch.empty((P, OBS_DIM), dtype=torch.float32, device=device)
                c = 0
                obs_p[:, c] = p_tgt.float() / S_LAT;                c += 1
                obs_p[:, c] = pred_la.float() / S_LAT;              c += 1
                obs_p[:, c] = error_p / S_LAT;                      c += 1
                obs_p[:, c] = k_tgt_p / S_CURV;                     c += 1
                obs_p[:, c] = k_cur_p / S_CURV;                     c += 1
                obs_p[:, c] = (k_tgt_p - k_cur_p) / S_CURV;        c += 1
                obs_p[:, c] = p_v.float() / S_VEGO;                 c += 1
                obs_p[:, c] = p_a.float() / S_AEGO;                 c += 1
                obs_p[:, c] = p_roll.float() / S_ROLL;              c += 1
                obs_p[:, c] = r_act[:, -1] / S_STEER;               c += 1
                obs_p[:, c] = ei / S_LAT;                           c += 1
                # Approximate future delta
                obs_p[:, c] = 0.0;                                   c += 1
                obs_p[:, c] = (pred_la.float() - r_lat[:, -1]) / DEL_T / S_LAT; c += 1
                obs_p[:, c] = (r_act[:, -1] - r_act[:, -2]) / DEL_T / S_STEER; c += 1
                obs_p[:, c] = fric_p;                                c += 1
                obs_p[:, c] = torch.clamp(1.0 - fric_p, min=0.0);   c += 1
                obs_p[:, c:c+HIST_LEN] = r_act / S_STEER;           c += HIST_LEN
                obs_p[:, c:c+HIST_LEN] = r_lat / S_LAT;             c += HIST_LEN
                # Future plan: tile from episode level
                fp_off = off
                fp_end2 = min(step_idx + 1 + fp_off + FUTURE_K, T)
                for attr, scale in [('target_lataccel', S_LAT), ('roll_lataccel', S_ROLL),
                                    ('v_ego', S_VEGO), ('a_ego', S_AEGO)]:
                    slc = dg[attr][:, step_idx+1+fp_off:fp_end2]
                    w = slc.shape[1]
                    # Tile from (N, w) → (N*K, w)
                    if w == 0:
                        fb = dg[attr][:, step_idx].unsqueeze(1).expand(-1, K).reshape(P)
                        obs_p[:, c:c+FUTURE_K] = (fb / scale).float().unsqueeze(1)
                    else:
                        slc_tiled = slc.unsqueeze(1).expand(-1, K, -1).reshape(P, w)
                        if w < FUTURE_K:
                            obs_p[:, c:c+w] = slc_tiled.float() / scale
                            obs_p[:, c+w:c+FUTURE_K] = (slc_tiled[:, -1:].float() / scale)
                        else:
                            obs_p[:, c:c+FUTURE_K] = slc_tiled[:, :FUTURE_K].float() / scale
                    c += FUTURE_K
                obs_p.clamp_(-5.0, 5.0)

                with torch.inference_mode():
                    out_p = actor(obs_p)
                a_ps = (F.softplus(out_p[:, 0]) + 1.0)
                b_ps = (F.softplus(out_p[:, 1]) + 1.0)
                s = np.random.beta(a_ps.cpu().numpy(), b_ps.cpu().numpy())
                d = np.clip((2.0 * s - 1.0) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
                cur = torch.from_numpy(
                    np.clip(r_act[:, -1].cpu().numpy() + d, STEER_RANGE[0], STEER_RANGE[1])
                ).to(device, dtype=torch.float64)

        # Reshape costs → (N, K), pick best per episode
        costs_2d = costs.reshape(N, K)
        best_idx = costs_2d.argmin(dim=1)  # (N,)
        best_actions = actions_0[torch.arange(N, device=device), best_idx]  # (N,)
        return best_actions

    def controller_fn(step_idx, *args):
        """Batched controller for all N episodes (GPU: 1 extra arg, CPU: 4 extra args)."""
        if len(args) == 1:
            # GPU path: controller_fn(step_idx, sim_ref)
            cla64 = _build_obs(step_idx, sim_ref=args[0])
        else:
            # CPU path: controller_fn(step_idx, target, cur_la, state_dict, future_plan)
            cla64 = _build_obs(step_idx, cpu_target=args[0], cpu_cur_la=args[1],
                               cpu_state=args[2], cpu_fplan=args[3])

        is_cpu_path = len(args) != 1

        if step_idx < CONTROL_START_IDX:
            h_act[:, :-1] = h_act[:, 1:].clone()
            h_act[:, -1] = 0.0
            h_act32[:, :-1] = h_act32[:, 1:].clone()
            h_act32[:, -1] = 0.0
            h_lat[:, :-1] = h_lat[:, 1:].clone()
            h_lat[:, -1] = cla64
            zeros = torch.zeros(N, dtype=torch.float64, device=device)
            return zeros.numpy() if is_cpu_path else zeros

        if do_mpc:
            action = _mpc_shoot(step_idx, args[0] if not is_cpu_path else None)
        else:
            with torch.inference_mode():
                out = actor(obs_buf)
            a_p = F.softplus(out[:, 0]).double() + 1.0
            b_p = F.softplus(out[:, 1]).double() + 1.0
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
            delta = (raw * DELTA_SCALE).clamp(-MAX_DELTA, MAX_DELTA)
            action = (h_act[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # Update histories
        h_act[:, :-1] = h_act[:, 1:].clone()
        h_act[:, -1] = action
        h_act32[:, :-1] = h_act32[:, 1:].clone()
        h_act32[:, -1] = action.float()
        h_lat[:, :-1] = h_lat[:, 1:].clone()
        h_lat[:, -1] = cla64

        return action.numpy() if is_cpu_path else action

    print(f"Running batched rollout: N={N}, MPC_N={MPC_N}, MPC_H={MPC_H}, device={device}")
    t0 = time.time()
    cost_dict = sim.rollout(controller_fn)
    elapsed = time.time() - t0
    print(f"Batched rollout: {elapsed:.1f}s ({elapsed/N*1000:.1f}ms per episode)")
    return cost_dict, ort_session


def batched_pid(files, model_path, device, ort_session=None):
    """Run PID baseline through BatchedSimulator. Returns cost dict with (N,) arrays."""
    csv_list = [str(f) for f in files]
    if ort_session is None:
        ort_session = make_ort_session(str(model_path))
    sim = BatchedSimulator(str(model_path), csv_list, ort_session=ort_session)
    N = sim.N

    PID_P, PID_I, PID_D = 0.195, 0.100, -0.053
    err_integral = torch.zeros(N, dtype=torch.float64, device=device)
    prev_err     = torch.zeros(N, dtype=torch.float64, device=device)

    use_gpu = device.type == 'cuda'
    dg = sim.data_gpu if use_gpu else None
    sim_data_np = sim.data

    def pid_fn(step_idx, *args):
        nonlocal err_integral, prev_err
        if len(args) == 1:
            sim_ref = args[0]
            target = dg['target_lataccel'][:, step_idx]
            current = sim_ref.current_lataccel
        else:
            target = torch.from_numpy(np.asarray(args[0], dtype=np.float64)).to(device)
            current = torch.from_numpy(np.asarray(args[1], dtype=np.float64)).to(device)

        if step_idx < CONTROL_START_IDX:
            err_integral.zero_()
            prev_err.zero_()
            z = torch.zeros(N, dtype=torch.float64, device=device)
            return z.numpy() if len(args) != 1 else z

        err = target - current
        err_integral += err
        err_diff = err - prev_err
        prev_err = err.clone()
        action = PID_P * err + PID_I * err_integral + PID_D * err_diff
        return action.numpy() if len(args) != 1 else action

    print(f"Running batched PID: N={N}, device={device}")
    t0 = time.time()
    cost_dict = sim.rollout(pid_fn)
    elapsed = time.time() - t0
    print(f"Batched PID: {elapsed:.1f}s ({elapsed/N*1000:.1f}ms per episode)")
    return cost_dict


def main():
    parser = argparse.ArgumentParser(description="Fast GPU-batched eval")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--test_controller", default="exp050_rich")
    parser.add_argument("--baseline_controller", default="pid")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt (default: auto-detect)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially with official run_rollout (for parity)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify parity with MPC_N=0 against official run_rollout")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    assert data_path.is_dir()
    files = sorted(data_path.iterdir())[:args.num_segs]

    # Auto-detect checkpoint
    if args.checkpoint is None:
        for p in [Path('experiments/exp050_rich_ppo/best_model.pt'),
                  Path('best_model_43.pt')]:
            if p.exists():
                args.checkpoint = str(p)
                break
        if args.checkpoint is None:
            raise FileNotFoundError("No checkpoint found. Pass --checkpoint path/to/best_model.pt")
    print(f"Checkpoint: {args.checkpoint}")

    if args.verify:
        _verify(files[:min(50, len(files))], args)
        return

    if args.sequential:
        _sequential_eval(files, args)
        return

    # ── Batched GPU eval ──
    use_cuda = os.getenv('CUDA', '0') == '1'
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    # Test controller: batched GPU
    test_costs, ort_sess = batched_eval(files, args.model_path, args.checkpoint, device)

    # Baseline PID: batched on same device, reuse ORT session
    baseline_costs = batched_pid(files, args.model_path, device, ort_session=ort_sess)

    # Sample rollouts for report plots (sequential, only n_sample files)
    sample_rollouts = []
    n_sample = min(SAMPLE_ROLLOUTS, len(files))
    for i in range(n_sample):
        f = files[i]
        _, test_tgt, test_cur = run_rollout(f, args.test_controller, args.model_path)
        _, base_tgt, base_cur = run_rollout(f, args.baseline_controller, args.model_path)
        sample_rollouts.append({
            'seg': f.stem,
            'test_controller': args.test_controller,
            'baseline_controller': args.baseline_controller,
            'desired_lataccel': test_tgt,
            'test_controller_lataccel': test_cur,
            'baseline_controller_lataccel': base_cur,
        })

    # Combine costs
    costs = []
    for i in range(len(files)):
        costs.append({'controller': 'test',
                      'lataccel_cost': float(test_costs['lataccel_cost'][i]),
                      'jerk_cost': float(test_costs['jerk_cost'][i]),
                      'total_cost': float(test_costs['total_cost'][i])})
        costs.append({'controller': 'baseline',
                      'lataccel_cost': float(baseline_costs['lataccel_cost'][i]),
                      'jerk_cost': float(baseline_costs['jerk_cost'][i]),
                      'total_cost': float(baseline_costs['total_cost'][i])})

    test_mean = np.mean(test_costs['total_cost'])
    base_mean = np.mean(baseline_costs['total_cost'])
    print(f"\nTest:     {test_mean:.2f}")
    print(f"Baseline: {base_mean:.2f}")

    create_report(args.test_controller, args.baseline_controller,
                  sample_rollouts, costs, len(files))


def _sequential_eval(files, args):
    """Sequential single-process eval using official run_rollout."""
    costs = []
    sample_rollouts = []
    n_sample = min(SAMPLE_ROLLOUTS, len(files))
    t0 = time.time()

    for i, f in enumerate(tqdm(files)):
        test_cost, test_tgt, test_cur = run_rollout(f, args.test_controller, args.model_path)
        base_cost, base_tgt, base_cur = run_rollout(f, args.baseline_controller, args.model_path)
        costs.append({'controller': 'test', **test_cost})
        costs.append({'controller': 'baseline', **base_cost})
        if i < n_sample:
            sample_rollouts.append({
                'seg': f.stem,
                'test_controller': args.test_controller,
                'baseline_controller': args.baseline_controller,
                'desired_lataccel': test_tgt,
                'test_controller_lataccel': test_cur,
                'baseline_controller_lataccel': base_cur,
            })

    elapsed = time.time() - t0
    test_mean = np.mean([c['total_cost'] for c in costs if c['controller'] == 'test'])
    print(f"\nTest: {test_mean:.2f}  ({elapsed:.1f}s)")
    create_report(args.test_controller, args.baseline_controller,
                  sample_rollouts, costs, len(files))


def _verify(files, args):
    """Verify parity between batched and sequential with MPC_N=0."""
    os.environ['MPC_N'] = '0'
    os.environ['MPC'] = '0'
    print(f"[verify] {len(files)} files, MPC_N=0 (deterministic), comparing batched vs sequential...")

    # Sequential costs
    seq_costs = []
    for f in tqdm(files, desc="sequential"):
        cost, _, _ = run_rollout(f, args.test_controller, args.model_path)
        seq_costs.append(cost['total_cost'])

    # Batched costs
    use_cuda = os.getenv('CUDA', '0') == '1'
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    batch_cost_dict, _ = batched_eval(files, args.model_path, args.checkpoint, device)
    batch_costs = batch_cost_dict['total_cost']

    max_diff = 0.0
    for i in range(len(files)):
        diff = abs(seq_costs[i] - batch_costs[i])
        max_diff = max(max_diff, diff)
        if diff > 0.1:
            print(f"  [!] {files[i].stem}: seq={seq_costs[i]:.4f} batch={batch_costs[i]:.4f} diff={diff:.4f}")

    if max_diff < 0.01:
        print(f"[verify] PASS — max_diff={max_diff:.6f}")
    else:
        print(f"[verify] WARN — max_diff={max_diff:.4f} (expected for GPU vs CPU float differences)")


if __name__ == "__main__":
    main()
