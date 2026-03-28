# exp061 — MPC-Teacher PPO
#
# Base: exp055 (256-dim Beta PPO, batch-of-batch, GPU-only)
# Twist: during TRAIN rollouts, each route is run MPC_N times in parallel
#        through the same TRT-backed BatchedSimulator.  Candidates diverge
#        through stochastic physics.  After the full episode the lowest-cost
#        candidate per route is selected and its trajectory is fed to PPO.
#        Test/submission controller is clean (no MPC) — pure policy inference.
#
# Summary of what changes vs exp055:
#   • batch in train() tiled by MPC_N; rollout() takes mpc_n arg
#   • Post-rollout winner selection; raw/logp recomputed from action_history
#   • No shadow ONNX session; no MPCShooter class; MPC_H removed
#   • Everything else (obs layout, BC pretrain, PPO update) unchanged.

import numpy as np, pandas as pd, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    ACC_G,
    State,
    FuturePlan,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS, C_LAYERS = 4, 4
DELTA_SCALE_MAX = float(os.getenv("DELTA_SCALE_MAX", "0.25"))
DELTA_SCALE_MIN = float(os.getenv("DELTA_SCALE_MIN", "0.25"))
MAX_DELTA = 0.5

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── MPC (train-time only) ─────────────────────────────────────
# MPC_N > 1: tile each route MPC_N times, run full episodes in parallel via
# the shared TRT-backed BatchedSimulator, select best trajectory per group.
# MPC_N = 1: pure PPO (no MPC).
MPC_N = int(os.getenv("MPC_N", "4"))  # candidates per route (1 = pure PPO)
MPC_EVERY = int(os.getenv("MPC_EVERY", "1"))  # run MPC every N epochs (1 = always)
MPC_EPOCH_START = int(os.getenv("MPC_EPOCH_START", "0"))  # first epoch to enable MPC
MPC_TEMP = float(os.getenv("MPC_TEMP", "3.0"))  # inflate Beta variance on MPC epochs
MPC_H = int(os.getenv("MPC_H", "6"))  # lookahead horizon (steps before reconverge)

# ── PPO ───────────────────────────────────────────────────────
PI_LR = float(os.getenv("PI_LR", "3e-4"))
VF_LR = float(os.getenv("VF_LR", "3e-4"))
LR_MIN = 5e-5
GAMMA = float(os.getenv("GAMMA", "0.95"))
LAMDA = float(os.getenv("LAMDA", "0.9"))
K_EPOCHS = 4
EPS_CLIP = 0.2
VF_COEF = 1.0
ENT_COEF = float(os.getenv("ENT_COEF", "0.003"))
SIGMA_FLOOR = float(os.getenv("SIGMA_FLOOR", "0.01"))
SIGMA_FLOOR_COEF = float(os.getenv("SIGMA_FLOOR_COEF", "0.5"))
ACT_SMOOTH = float(os.getenv("ACT_SMOOTH", "0.0"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1.0"))
MINI_BS = int(os.getenv("MINI_BS", "25_000"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))


# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "20"))
BC_LR = float(os.getenv("BC_LR", "0.01"))
BC_BS = int(os.getenv("BC_BS", "2048"))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "10"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = 5
EVAL_N = 100
RESUME = os.getenv("RESUME", "0") == "1"
RESUME_OPT = os.getenv("RESUME_OPT", "1") == "1"
RESUME_DS = os.getenv("RESUME_DS", "0") == "1"
RESET_CRITIC = os.getenv("RESET_CRITIC", "0") == "1"
RESUME_WARMUP = os.getenv("RESUME_WARMUP", "0") == "1"
LR_DECAY = os.getenv("LR_DECAY", "1") == "1"
DELTA_SCALE_DECAY = os.getenv("DELTA_SCALE_DECAY", "0") == "1"
REWARD_RMS_NORM = os.getenv("REWARD_RMS_NORM", "1") == "1"
ADV_NORM = os.getenv("ADV_NORM", "1") == "1"
COMPILE = os.getenv("COMPILE", "0") == "1"


def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout offsets ────────────────────────────────────────
C = 16
H1 = C + HIST_LEN  # 36
H2 = H1 + HIST_LEN  # 56
F_LAT = H2  # 56
F_ROLL = F_LAT + FUTURE_K  # 106
F_V = F_ROLL + FUTURE_K  # 156
F_A = F_V + FUTURE_K  # 206
OBS_DIM = F_A + FUTURE_K  # 256


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        for layer in self.actor[:-1]:
            _ortho(layer)
        _ortho(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]:
            _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

    def beta_params(self, obs):
        out = self.actor(obs)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0


# ══════════════════════════════════════════════════════════════
#  Observation builder (GPU, batched) — identical to exp055
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _windows(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        "target_lataccel": _windows(dg["target_lataccel"]),
        "roll_lataccel": _windows(dg["roll_lataccel"]),
        "v_ego": _windows(dg["v_ego"]),
        "a_ego": _windows(dg["a_ego"]),
    }


def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale
        return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale


def fill_obs(
    buf,
    target,
    current,
    roll_la,
    v_ego,
    a_ego,
    h_act,
    h_lat,
    hist_head,
    error_integral,
    future,
    step_idx,
):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    prev_act = h_act[:, hist_head]
    prev_act2 = h_act[:, (hist_head - 1) % HIST_LEN]
    prev_lat = h_lat[:, hist_head]

    buf[:, 0] = target / S_LAT
    buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV
    buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO
    buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL
    buf[:, 9] = prev_act / S_STEER
    buf[:, 10] = error_integral / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - prev_lat) / DEL_T / S_LAT
    buf[:, 13] = (prev_act - prev_act2) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)

    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)

    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO

    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  GPU Rollout  (uniform — MPC is just tiling, no shadow session)
# ══════════════════════════════════════════════════════════════
#
# MPC design
# ──────────
# Pass mpc_n > 1 to tile each CSV mpc_n times in the BatchedSimulator.
# The sim runs all N = N_real * mpc_n rows through TRT in one batched ONNX
# call per step — identical to a pure-PPO rollout, just larger.
#
# At each ctrl() step the actor runs once on N_real obs (candidate 0 of each
# group), samples mpc_n actions per group, and broadcasts them across the
# N rows.  Candidates diverge naturally through the stochastic physics.
#
# After the full episode, per-group costs are computed from sim history,
# the lowest-cost candidate per group is the winner, and PPO receives
# only the winner's trajectory — obs, raw delta, logp, rewards.
#
# There is no shadow ONNX session, no MPCShooter class, no expected-value
# approximation, no horizon limit.  The full episode is the horizon.
# TRT compiles once for the tiled batch size and is reused every epoch.


def rollout(
    csv_files,
    ac,
    mdl_path,
    ort_session,
    csv_cache,
    deterministic=False,
    ds=DELTA_SCALE_MAX,
    mpc_n=1,
    mpc_temp=1.0,
):
    """Run a full batched episode rollout.

    MPC via tiling
    ──────────────
    Pass mpc_n > 1 to run MPC-style best-of-K selection.  csv_files must
    already be tiled (each route repeated mpc_n times consecutively) so that
    rows [r*mpc_n : r*mpc_n+mpc_n] are mpc_n independent candidates for
    route r, all driven by the same policy but diverging through stochastic
    physics.

    The actor runs on N_real = N // mpc_n observations (one per route group)
    and samples mpc_n distinct actions per group.  After the full episode,
    the lowest-cost candidate per group is selected and only its trajectory
    is returned to PPO.

    Everything — physics, ONNX, TRT — runs through the single shared
    BatchedSimulator session.  No shadow session.  No expected-value
    approximation.  The full episode is the scoring horizon.
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    # Deterministic physics during training: use E[lataccel] instead of sampling.
    # This gives PPO clean, noise-free reward signal and makes MPC candidate
    # comparison deterministic (like PGTO). Eval stays stochastic.
    if not deterministic:
        sim.use_expected = True
    N, T = sim.N, sim.T
    N_real = N // mpc_n  # number of distinct routes
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    # future windows only for N_real rows (candidate-0 of each group).
    # c0 indices are [0, mpc_n, 2*mpc_n, ...] — one per route group.
    # All MPC_N copies of a route share identical CSV data so there is no
    # information loss. This avoids an (N * T * FUTURE_K) allocation that
    # would be N_real * mpc_n * 600 * 50 * 4B = ~38 GB at mpc_n=16.
    dg_real = {k: v[::mpc_n] for k, v in dg.items()} if mpc_n > 1 else dg
    future = _precompute_future_windows(dg_real)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    # obs_buf: only N_real rows — obs is identical across mpc_n copies of a route
    # (they share the same current_lataccel until after sim_step diverges them)
    obs_buf = torch.empty((N_real, OBS_DIM), dtype=torch.float32, device="cuda")

    if not deterministic:
        if mpc_n > 1:
            # MPC: pinned CPU to keep GPU headroom for the N*mpc_n sim batch.
            # Store raw/logp for ALL candidates — (steps, N_real, mpc_n).
            all_obs = torch.empty(
                (max_steps, N_real, OBS_DIM), dtype=torch.float32
            ).pin_memory()
            all_val = torch.empty((max_steps, N_real), dtype=torch.float32).pin_memory()
            all_raw_m = torch.empty(
                (max_steps, N_real, mpc_n), dtype=torch.float32
            ).pin_memory()
            all_logp_m = torch.empty(
                (max_steps, N_real, mpc_n), dtype=torch.float32
            ).pin_memory()
        else:
            # Non-MPC: pinned CPU to leave GPU headroom for use_expected physics.
            all_obs = torch.empty(
                (max_steps, N_real, OBS_DIM), dtype=torch.float32
            ).pin_memory()
            all_val = torch.empty((max_steps, N_real), dtype=torch.float32).pin_memory()
            all_raw = torch.empty((max_steps, N_real), dtype=torch.float32).pin_memory()
            all_logp = torch.empty(
                (max_steps, N_real), dtype=torch.float32
            ).pin_memory()

    si = 0
    hist_head = HIST_LEN - 1
    route_idx = torch.arange(N_real, device="cuda")
    # Per-step winner indices for gathering raw/logp after rollout
    all_winners = (
        torch.empty((max_steps, N_real), dtype=torch.long, device="cuda")
        if mpc_n > 1
        else None
    )
    # MPC horizon state: accumulate cost over MPC_H steps, reconverge at the end
    mpc_h = MPC_H if mpc_n > 1 else 1
    horizon_cost = torch.zeros(N, dtype=torch.float32, device="cuda")  # running cost
    horizon_start = -1  # step_idx where current horizon window started
    horizon_count = 0  # how many steps into the current horizon

    def _reconverge_horizon(sim_ref):
        """H-step MPC reconvergence: pick winner by cumulative horizon cost,
        scatter winner's state for the ENTIRE horizon window back into all
        candidates, then reset cost accumulator."""
        nonlocal horizon_cost, horizon_start, horizon_count

        if horizon_count == 0:
            return

        # Winner by cumulative cost over the horizon window
        cost_2d = horizon_cost.view(N_real, mpc_n)  # (N_real, mpc_n)
        winner = cost_2d.argmin(dim=1)  # (N_real,)
        win_flat = route_idx * mpc_n + winner  # (N_real,)

        # Record the winner for ALL steps in this horizon window
        # (the winning candidate's action at each step is what PPO learns from)
        for s in range(horizon_count):
            rec_idx = (horizon_start + s) - CONTROL_START_IDX
            if 0 <= rec_idx < max_steps:
                all_winners[rec_idx] = winner

        # Scatter winner's sim state for each step in the horizon window
        h_end = sim_ref._hist_len - 1  # last slot written
        h_beg = h_end - horizon_count + 1
        for h_slot in range(h_beg, h_end + 1):
            win_act = sim_ref.action_history[win_flat, h_slot]
            win_lat = sim_ref.current_lataccel_history[win_flat, h_slot]
            sim_ref.action_history[:, h_slot] = win_act.repeat_interleave(mpc_n)
            sim_ref.current_lataccel_history[:, h_slot] = win_lat.repeat_interleave(
                mpc_n
            )

        # Scatter final state
        win_cur = sim_ref.current_lataccel[win_flat]
        sim_ref.current_lataccel = win_cur.repeat_interleave(mpc_n)

        # Reconverge local ring buffers — scatter the last H slots
        for offset in range(horizon_count):
            ring_idx = (hist_head - offset) % HIST_LEN
            h_act[:, ring_idx] = h_act[win_flat, ring_idx].repeat_interleave(mpc_n)
            h_act32[:, ring_idx] = h_act32[win_flat, ring_idx].repeat_interleave(mpc_n)
            h_lat[:, ring_idx] = h_lat[win_flat, ring_idx].repeat_interleave(mpc_n)
            h_error[:, ring_idx] = h_error[win_flat, ring_idx].repeat_interleave(mpc_n)

        win_err_sum = err_sum[win_flat]
        err_sum[:] = win_err_sum.repeat_interleave(mpc_n)

        # Reset for next horizon
        horizon_cost.zero_()
        horizon_count = 0

    def _accumulate_cost(sim_ref, step_idx_cur):
        """Add 1-step cost to the horizon accumulator."""
        if sim_ref.expected_lataccel is not None:
            cur_la = sim_ref.expected_lataccel
        else:
            cur_la = sim_ref.current_lataccel
        tgt_la = dg["target_lataccel"][:, step_idx_cur]
        h = sim_ref._hist_len - 1
        if h >= 1:
            prev_la = sim_ref.current_lataccel_history[:, h - 1]
            jerk = ((cur_la - prev_la) / DEL_T).float()
        else:
            jerk = torch.zeros(N, device="cuda")
        lat_err = ((tgt_la - cur_la) ** 2).float() * (100 * LAT_ACCEL_COST_MULTIPLIER)
        horizon_cost.add_(lat_err + jerk**2 * 100)

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
        nonlocal horizon_cost, horizon_start, horizon_count
        c0 = route_idx * mpc_n

        # ── Accumulate cost from previous step and reconverge if horizon complete ──
        if mpc_n > 1 and step_idx > CONTEXT_LENGTH and horizon_count > 0:
            _accumulate_cost(sim_ref, step_idx - 1)
            if horizon_count >= mpc_h:
                _reconverge_horizon(sim_ref)

        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        roll_la = dg["roll_lataccel"][:, step_idx]
        v_ego = dg["v_ego"][:, step_idx]
        a_ego = dg["a_ego"][:, step_idx]

        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_error = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_error
        ei_all = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        # Build obs from candidate-0 rows only.
        # After reconvergence, all candidates in a group are identical,
        # so candidate-0 IS the shared state.
        fill_obs(
            obs_buf,
            target[c0].float(),
            cur32[c0],
            roll_la[c0].float(),
            v_ego[c0].float(),
            a_ego[c0].float(),
            h_act32[c0],
            h_lat[c0],
            hist_head,
            ei_all[c0],
            future,
            step_idx,
        )

        with torch.inference_mode():
            val = ac.critic(obs_buf).squeeze(-1)

        if deterministic:
            with torch.inference_mode():
                logits = ac.actor(obs_buf)
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            raw_r = 2.0 * a_p / (a_p + b_p) - 1.0
            logp = None
            delta = raw_r.to(h_act.dtype) * ds
            act_r = (h_act[c0, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
            action = act_r.repeat_interleave(mpc_n)
        else:
            with torch.no_grad():
                logits = ac.actor(obs_buf)
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            dist = torch.distributions.Beta(a_p, b_p)
            if mpc_n > 1:
                if horizon_count == 0:
                    # ── Step 1 of horizon: sample MPC_N diverse candidates ──
                    horizon_start = step_idx
                    a_wide = a_p / mpc_temp
                    b_wide = b_p / mpc_temp
                    dist_wide = torch.distributions.Beta(a_wide, b_wide)
                    x_mat = dist_wide.sample((mpc_n,)).T
                    x_mat[:, 0] = a_p / (a_p + b_p)  # slot 0 = mean
                    raw_mat = 2.0 * x_mat - 1.0
                    delta_mat = raw_mat.to(h_act.dtype) * ds
                    delta_mat = delta_mat.clamp(-MAX_DELTA, MAX_DELTA)
                    prev_act = h_act[c0, hist_head]
                    act_mat = (prev_act.unsqueeze(1) + delta_mat).clamp(
                        STEER_RANGE[0], STEER_RANGE[1]
                    )
                    action = act_mat.reshape(N)
                    # logp under original policy for PPO
                    dist_m = torch.distributions.Beta(
                        a_p.unsqueeze(1).expand_as(x_mat),
                        b_p.unsqueeze(1).expand_as(x_mat),
                    )
                    logp_mat = dist_m.log_prob(x_mat.clamp(1e-6, 1 - 1e-6))
                    raw_r = raw_mat[:, 0]
                    logp_r = logp_mat[:, 0]
                else:
                    # ── Steps 2..H of horizon: each candidate uses policy mean ──
                    # Candidates have diverged — build obs for ALL N rows.
                    # Tile future slices per-step (cheap) instead of full tensor (2+ GB).
                    future_step = {
                        k: v[:, step_idx].repeat_interleave(mpc_n, dim=0)
                        for k, v in future.items()
                    }  # each is (N, FUTURE_K) — tiny, transient
                    # Wrap as indexable so fill_obs can do [:, step_idx]
                    future_n = {k: v.unsqueeze(1) for k, v in future_step.items()}
                    obs_all = torch.empty(
                        (N, OBS_DIM), dtype=torch.float32, device="cuda"
                    )
                    fill_obs(
                        obs_all,
                        target.float(),
                        cur32,
                        roll_la.float(),
                        v_ego.float(),
                        a_ego.float(),
                        h_act32,
                        h_lat,
                        hist_head,
                        ei_all,
                        future_n,
                        0,  # step_idx=0 into the (N,1,FUTURE_K) wrapper
                    )
                    with torch.no_grad():
                        logits_all = ac.actor(obs_all)
                    a_all = F.softplus(logits_all[..., 0]) + 1.0
                    b_all = F.softplus(logits_all[..., 1]) + 1.0
                    raw_all = 2.0 * a_all / (a_all + b_all) - 1.0  # deterministic mean
                    delta_all = raw_all.to(h_act.dtype) * ds
                    delta_all = delta_all.clamp(-MAX_DELTA, MAX_DELTA)
                    action = (h_act[:, hist_head] + delta_all).clamp(
                        STEER_RANGE[0], STEER_RANGE[1]
                    )
                    # Compute raw_mat/logp_mat for all candidates at this step.
                    # Each candidate used its own policy mean — reshape (N,) → (N_real, mpc_n).
                    raw_mat = raw_all.view(N_real, mpc_n)
                    x_all = ((raw_mat + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)
                    # a_all/b_all are (N,) — reshape to (N_real, mpc_n) for logp
                    logp_mat = torch.distributions.Beta(
                        a_all.view(N_real, mpc_n),
                        b_all.view(N_real, mpc_n),
                    ).log_prob(x_all)
                    raw_r = raw_mat[:, 0]
                    logp_r = logp_mat[:, 0]
                horizon_count += 1
            else:
                x = dist.sample()
                raw_r = 2.0 * x - 1.0
                logp_r = dist.log_prob(x)
                delta = raw_r.to(h_act.dtype) * ds
                action = (
                    (h_act[c0, hist_head] + delta)
                    .clamp(STEER_RANGE[0], STEER_RANGE[1])
                    .repeat_interleave(mpc_n)
                )
                logp = logp_r

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if not deterministic and step_idx < COST_END_IDX:
            if mpc_n > 1:
                # MPC: pinned CPU buffers — GPU→CPU copy
                all_obs[si] = obs_buf.cpu()
                all_val[si] = val.cpu()
                all_raw_m[si] = raw_mat.float().cpu()
                all_logp_m[si] = logp_mat.float().cpu()
            else:
                # Non-MPC: pinned CPU
                all_obs[si] = obs_buf.cpu()
                all_val[si] = val.cpu()
                all_raw[si] = raw_r.float().cpu()
                all_logp[si] = logp.float().cpu()
            si += 1
        return action

    costs_dict = sim.rollout(ctrl)

    # Final reconverge for any remaining horizon window
    if mpc_n > 1 and horizon_count > 0:
        _accumulate_cost(sim, sim.T - 1)
        _reconverge_horizon(sim)

    costs_all = torch.from_numpy(costs_dict["total_cost"]).to(DEV)  # (N,)

    if deterministic:
        return costs_all[route_idx * mpc_n].tolist()

    S = si
    c0 = route_idx * mpc_n

    # ── Post-rollout data extraction ──────────────────────────────────────────
    # Extract what we need from sim, then free its GPU memory before transfers.
    start_t, end_t = CONTROL_START_IDX, CONTROL_START_IDX + S
    if mpc_n > 1:
        pred = sim.current_lataccel_history[c0, start_t:end_t].float().clone()
        target_t = dg["target_lataccel"][c0, start_t:end_t].float().clone()
        act_t = sim.action_history[c0, start_t:end_t].float().clone()
        costs_out = costs_all[c0].cpu().numpy()
    else:
        pred = sim.current_lataccel_history[:, start_t:end_t].float().clone()
        target_t = dg["target_lataccel"][:, start_t:end_t].float().clone()
        act_t = sim.action_history[:, start_t:end_t].float().clone()
        costs_out = costs_dict["total_cost"]

    # Free sim GPU memory (~10+ GB) before pinned CPU → GPU transfers
    del sim, dg, future
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    if mpc_n > 1:
        raw_m = all_raw_m[:S]
        logp_m = all_logp_m[:S]
        w_idx = all_winners[:S].cpu().unsqueeze(-1)
        raw_win = raw_m.gather(2, w_idx).squeeze(2)
        logp_win = logp_m.gather(2, w_idx).squeeze(2)

        obs_gpu = (
            all_obs[:S].to(DEV, non_blocking=True).permute(1, 0, 2).reshape(-1, OBS_DIM)
        )
        raw_flat = raw_win.to(DEV, non_blocking=True).T.reshape(-1)
        logp_flat = logp_win.to(DEV, non_blocking=True).T.reshape(-1)
        torch.cuda.synchronize()
    else:
        obs_gpu = (
            all_obs[:S].to(DEV, non_blocking=True).permute(1, 0, 2).reshape(-1, OBS_DIM)
        )
        raw_gpu = all_raw[:S].to(DEV, non_blocking=True).T.reshape(-1)
        old_logp_gpu = all_logp[:S].to(DEV, non_blocking=True).T.reshape(-1)
        val_2d_gpu = all_val[:S].to(DEV, non_blocking=True).T
        torch.cuda.synchronize()

    lat_r = (target_t - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act_t, dim=1, prepend=act_t[:, :1]) / DEL_T
    rew = (
        -(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(REWARD_SCALE, 1e-8)
    ).float()
    dones = torch.zeros((N_real, S), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    if mpc_n > 1:
        val_2d_gpu = all_val[:S].to(DEV, non_blocking=True).T
        torch.cuda.synchronize()
        return dict(
            obs=obs_gpu,
            raw=raw_flat,
            old_logp=logp_flat,
            val_2d=val_2d_gpu,
            rew=rew,
            done=dones,
            costs=costs_out,
            spr=1,  # MPC winner selection already filtered; no SPR grouping in PPO
        )

    return dict(
        obs=obs_gpu,
        raw=raw_gpu,
        old_logp=old_logp_gpu,
        val_2d=val_2d_gpu,
        rew=rew,
        done=dones,
        costs=costs_out,
        spr=SAMPLES_PER_ROUTE,
    )


# ══════════════════════════════════════════════════════════════
#  BC Pretrain — identical to exp055
# ══════════════════════════════════════════════════════════════


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), "edge")
    return np.full(k, fallback, dtype=np.float32)


def _build_obs_bc(target, current, state, fplan, hist_act, hist_lat):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    _flat = getattr(fplan, "lataccel", None)
    fp0 = _flat[0] if (_flat and len(_flat) > 0) else target
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0

    core = np.array(
        [
            target / S_LAT,
            current / S_LAT,
            (target - current) / S_LAT,
            k_tgt / S_CURV,
            k_cur / S_CURV,
            (k_tgt - k_cur) / S_CURV,
            state.v_ego / S_VEGO,
            state.a_ego / S_AEGO,
            state.roll_lataccel / S_ROLL,
            hist_act[-1] / S_STEER,
            0.0,
            (fp0 - target) / DEL_T / S_LAT,
            (current - hist_lat[-1]) / DEL_T / S_LAT,
            (hist_act[-1] - hist_act[-2]) / DEL_T / S_STEER,
            fric,
            max(0.0, 1.0 - fric),
        ],
        dtype=np.float32,
    )

    obs = np.concatenate(
        [
            core,
            np.array(hist_act, np.float32) / S_STEER,
            np.array(hist_lat, np.float32) / S_LAT,
            _future_raw(fplan, "lataccel", target) / S_LAT,
            _future_raw(fplan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
            _future_raw(fplan, "v_ego", state.v_ego) / S_VEGO,
            _future_raw(fplan, "a_ego", state.a_ego) / S_AEGO,
        ]
    )
    return np.clip(obs, -5.0, 5.0)


def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values
    a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values
    steer = -df["steerCommand"].values

    obs_list, raw_list = [], []
    h_act = [0.0] * HIST_LEN
    h_lat = [0.0] * HIST_LEN

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        target_la = tgt[step_idx]
        state = State(
            roll_lataccel=roll_la[step_idx],
            v_ego=v_ego[step_idx],
            a_ego=a_ego[step_idx],
        )
        fplan = FuturePlan(
            lataccel=tgt[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS].tolist(),
        )

        obs = _build_obs_bc(target_la, target_la, state, fplan, h_act, h_lat)
        raw_target = np.clip((steer[step_idx] - h_act[-1]) / DELTA_SCALE_MAX, -1.0, 1.0)
        obs_list.append(obs)
        raw_list.append(raw_target)
        h_act = h_act[1:] + [steer[step_idx]]
        h_lat = h_lat[1:] + [tgt[step_idx]]

    return (np.array(obs_list, np.float32), np.array(raw_list, np.float32))


def pretrain_bc(ac, all_csvs):
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(
        _bc_worker,
        [str(f) for f in all_csvs],
        max_workers=10,
        chunksize=50,
        disable=False,
    )
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    N = len(all_obs)
    print(f"BC pretrain: {N} samples, {BC_EPOCHS} epochs")

    obs_t = torch.FloatTensor(all_obs).to(DEV)
    raw_t = torch.FloatTensor(all_raw).to(DEV)
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            a_out = ac.actor(obs_t[idx])
            a_p = F.softplus(a_out[..., 0]) + 1.0
            b_p = F.softplus(a_out[..., 1]) + 1.0
            x = ((raw_t[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        print(
            f"  BC epoch {ep}: loss={total / nb:.6f}  lr={opt.param_groups[0]['lr']:.1e}"
        )
    print("BC pretrain done.\n")


# ══════════════════════════════════════════════════════════════
#  PPO — identical to exp055
# ══════════════════════════════════════════════════════════════


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / tot
        ) / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(ac.actor.parameters(), lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._rms = RunningMeanStd()

    @staticmethod
    def _beta_sigma_raw(alpha, beta):
        return 2.0 * torch.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )

    def _gae(self, rew, val, done):
        if REWARD_RMS_NORM:
            with torch.no_grad():
                flat = rew.reshape(-1)
                self._rms.update(flat.mean().item(), flat.var().item(), flat.numel())
            rew = rew / max(self._rms.std, 1e-8)
        N, S = rew.shape
        adv = torch.empty_like(rew)
        g = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(S - 1, -1, -1):
            nv = val[:, t + 1] if t < S - 1 else g
            mask = 1.0 - done[:, t]
            g = (rew[:, t] + GAMMA * nv * mask - val[:, t]) + GAMMA * LAMDA * mask * g
            adv[:, t] = g
        return adv.reshape(-1), (adv + val).reshape(-1)

    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX):
        obs = gd["obs"]
        raw = gd["raw"].unsqueeze(-1)
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])
        # spr: effective samples-per-route seen by PPO.
        # In MPC mode (mpc_n>1) winner selection already filtered to 1 winner
        # per route — no within-group normalisation applies.
        spr = gd.get("spr", SAMPLES_PER_ROUTE)
        if spr > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // spr
            adv_2d = adv_t.reshape(n_routes, spr, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)
        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)
        sigma_pen = torch.tensor(0.0, device="cuda")

        n_vf = n_actor = 0
        vf_sum = pi_sum = ent_sum = sigma_pen_sum = 0.0

        with torch.no_grad():
            if "old_logp" in gd:
                old_lp = gd["old_logp"]
            else:
                old_out = self.ac.actor(obs)
                a_old = F.softplus(old_out[..., 0]) + 1.0
                b_old = F.softplus(old_out[..., 1]) + 1.0
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(
                    x_t.squeeze(-1)
                )

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device="cuda").split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                val = self.ac.critic(obs[idx]).squeeze(-1)
                vf_loss = F.mse_loss(val, ret_t[idx], reduction="mean")
                bs = int(idx.numel())
                vf_sum += vf_loss.detach().item() * bs
                n_vf += bs

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                else:
                    a_out = self.ac.actor(obs[idx])
                    a_c = F.softplus(a_out[..., 0]) + 1.0
                    b_c = F.softplus(a_out[..., 1]) + 1.0
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv, ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv
                    ).mean()
                    ent = dist.entropy().mean()
                    sigma_raw_mb = self._beta_sigma_raw(a_c, b_c).mean()
                    sigma_eff_mb = sigma_raw_mb * ds
                    sigma_pen = F.relu(SIGMA_FLOOR - sigma_eff_mb)
                    loss = (
                        pi_loss
                        + VF_COEF * vf_loss
                        - ENT_COEF * ent
                        + SIGMA_FLOOR_COEF * sigma_pen
                    )
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.pi_opt.step()
                    self.vf_opt.step()
                    pi_sum += pi_loss.detach().item() * bs
                    ent_sum += ent.detach().item() * bs
                    sigma_pen_sum += sigma_pen.detach().item() * bs
                    n_actor += bs

        with torch.no_grad():
            diag_out = self.ac.actor(obs[:1000])
            a_d = F.softplus(diag_out[..., 0]) + 1.0
            b_d = F.softplus(diag_out[..., 1]) + 1.0
            sigma_raw = self._beta_sigma_raw(a_d, b_d).mean().item()
            sigma_eff = sigma_raw * ds
        return dict(
            pi=(pi_sum / max(1, n_actor)) if not critic_only else 0.0,
            vf=(vf_sum / max(1, n_vf)),
            ent=(ent_sum / max(1, n_actor)) if not critic_only else 0.0,
            σ=sigma_eff,
            σraw=sigma_raw,
            σpen=(sigma_pen_sum / max(1, n_actor)) if not critic_only else 0.0,
            lr=self.pi_opt.param_groups[0]["lr"],
        )


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(
        files,
        ac,
        mdl_path,
        ort_session,
        csv_cache,
        deterministic=True,
        ds=ds,
        mpc_n=1,  # eval is always pure policy — no tiling
    )
    return float(np.mean(costs)), float(np.std(costs))


def train():
    ac = ActorCritic().to(DEV)
    ppo = PPO(ac)
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    all_csv = sorted((ROOT / "data").glob("*.csv"))
    tr_f = all_csv
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    print(
        f"MPC: mpc_n={MPC_N}  mpc_h={MPC_H}  mpc_every={MPC_EVERY}  mpc_start={MPC_EPOCH_START}  mpc_temp={MPC_TEMP}"
        f"  (1 = pure PPO, >1 = best-of-K via tiled sim)"
    )

    warmup_off = 0
    resumed_ds = None
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt["ac"])
        if RESUME_OPT and "pi_opt" in ckpt:
            ppo.pi_opt.load_state_dict(ckpt["pi_opt"])
            ppo.vf_opt.load_state_dict(ckpt["vf_opt"])
            if "ret_rms" in ckpt:
                r = ckpt["ret_rms"]
                ppo._rms.mean, ppo._rms.var, ppo._rms.count = (
                    r["mean"],
                    r["var"],
                    r["count"],
                )
        elif RESUME_OPT:
            print(
                "RESUME_OPT=1 but optimizer state missing in checkpoint; using fresh state"
            )
        warmup_off = 0 if RESUME_WARMUP else CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
        if RESUME_DS:
            ds_ckpt = ckpt.get("delta_scale", None)
            if ds_ckpt is not None:
                resumed_ds = float(ds_ckpt)
                print(f"Resumed delta_scale={resumed_ds:.6f} from checkpoint")
        if RESET_CRITIC:
            for layer in ac.critic[:-1]:
                if isinstance(layer, nn.Linear):
                    _ortho(layer)
            if isinstance(ac.critic[-1], nn.Linear):
                _ortho(ac.critic[-1], gain=1.0)
            ppo.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic, vf_opt, and ret_rms reset")
    else:
        pretrain_bc(ac, all_csv)

    if COMPILE:
        ac.actor = torch.compile(
            ac.actor, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        ac.critic = torch.compile(
            ac.critic, mode="max-autotune-no-cudagraphs", dynamic=True
        )

    ds_max_run = DELTA_SCALE_MAX
    ds_min_run = DELTA_SCALE_MIN
    if RESUME_DS and resumed_ds is not None:
        ds_max_run = resumed_ds
        ds_min_run = min(ds_min_run, ds_max_run)

    baseline_ds = (
        ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(0.0))
        if DELTA_SCALE_DECAY
        else ds_max_run
    )
    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=baseline_ds)
    best, best_ep = vm, "init"
    print(f"Baseline: {vm:.1f} ± {vs:.1f}  (Δs={baseline_ds:.4f})")

    cur_ds = ds_max_run

    def save_best():
        torch.save(
            {
                "ac": ac.state_dict(),
                "pi_opt": ppo.pi_opt.state_dict(),
                "vf_opt": ppo.vf_opt.state_dict(),
                "ret_rms": {
                    "mean": ppo._rms.mean,
                    "var": ppo._rms.var,
                    "count": ppo._rms.count,
                },
                "delta_scale": cur_ds,
            },
            BEST_PT,
        )

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}  mpc_n={MPC_N}")
    _n_r = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
    print(
        f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {_n_r} routes × {SAMPLES_PER_ROUTE} = {_n_r * SAMPLES_PER_ROUTE} rollouts/epoch"
    )
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
        f"  rew_scale={REWARD_SCALE:g}"
        f"  lr_decay={'on' if LR_DECAY else 'off'}"
        f"  σfloor_eff={SIGMA_FLOOR} coef={SIGMA_FLOOR_COEF}"
        f"  rew_rms_norm={'on' if REWARD_RMS_NORM else 'off'}"
        f"  adv_norm={'on' if ADV_NORM else 'off'}"
        f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}\n"
    )

    for epoch in range(MAX_EP):
        if DELTA_SCALE_DECAY:
            ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (
                1 + np.cos(np.pi * epoch / MAX_EP)
            )
        else:
            ds = ds_max_run
        cur_ds = ds
        if RESUME and RESUME_OPT and epoch == 0:
            pi_lr = ppo.pi_opt.param_groups[0]["lr"]
            vf_lr = ppo.vf_opt.param_groups[0]["lr"]
        elif LR_DECAY:
            pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
            vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
        else:
            pi_lr = PI_LR
            vf_lr = VF_LR
        for pg in ppo.pi_opt.param_groups:
            pg["lr"] = pi_lr
        for pg in ppo.vf_opt.param_groups:
            pg["lr"] = vf_lr

        t0 = time.time()
        # MPC_EVERY: use MPC tiling every N epochs, pure PPO in between.
        # This amortises the MPC_N× ONNX cost while still getting high-quality
        # trajectories periodically.
        use_mpc = MPC_N > 1 and epoch >= MPC_EPOCH_START and (epoch % MPC_EVERY == 0)
        if use_mpc:
            # MPC mode: keep total sim rows ≈ CSVS_EPOCH so epoch time is constant.
            # n_routes = CSVS_EPOCH / MPC_N, each tiled MPC_N times → total = CSVS_EPOCH.
            n_routes = max(1, min(CSVS_EPOCH, len(tr_f)) // MPC_N)
            batch = random.sample(tr_f, n_routes)
            batch = [f for f in batch for _ in range(MPC_N)]
            epoch_mpc_n = MPC_N
        else:
            # Pure PPO: SPR tiling for within-group advantage normalisation.
            n_routes = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
            batch = random.sample(tr_f, max(n_routes, 1))
            batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]
            epoch_mpc_n = 1
        res = rollout(
            batch,
            ac,
            mdl_path,
            ort_sess,
            csv_cache,
            deterministic=False,
            ds=ds,
            mpc_n=epoch_mpc_n,
            mpc_temp=MPC_TEMP if use_mpc else 1.0,
        )

        t1 = time.time()
        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co, ds=ds)
        tu = time.time() - t1

        mpc_tag = " [MPC]" if use_mpc else ""
        phase = "  [critic warmup]" if co else mpc_tag
        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}  σraw={info['σraw']:.4f}"
            f"  σpen={info['σpen']:.4f}  π={info['pi']:+.4f}"
            f"  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  Δs={ds:.4f}  lr={info['lr']:.1e}  ⏱ {t1 - t0:.0f}+{tu:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    torch.save({"ac": ac.state_dict()}, EXP_DIR / "final_model.pt")


if __name__ == "__main__":
    train()
