#!/usr/bin/env python3
# exp057 — exp055 + GPU train-time shooting teacher
#
# Keep the exp055 actor/critic and inference path unchanged.
# The only new ingredient is a train-time first-action shooting teacher:
# sample many first actions from the current policy, roll them forward under
# deterministic policy continuation and expected plant dynamics, then distill
# the best first action back into the actor.

import importlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (  # noqa: E402
    ACC_G,
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
    COST_END_IDX,
    DEL_T,
    FUTURE_PLAN_STEPS,
    LAT_ACCEL_COST_MULTIPLIER,
    MAX_ACC_DELTA,
    STEER_RANGE,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session  # noqa: E402

base = importlib.import_module("experiments.exp055_batch_of_batch.train")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

DEV = base.DEV

HIST_LEN = base.HIST_LEN
FUTURE_K = base.FUTURE_K
STATE_DIM = base.STATE_DIM
HIDDEN = base.HIDDEN
A_LAYERS = base.A_LAYERS
C_LAYERS = base.C_LAYERS
DELTA_SCALE_MAX = base.DELTA_SCALE_MAX
DELTA_SCALE_MIN = base.DELTA_SCALE_MIN

S_LAT = base.S_LAT
S_STEER = base.S_STEER
S_VEGO = base.S_VEGO
S_AEGO = base.S_AEGO
S_ROLL = base.S_ROLL
S_CURV = base.S_CURV

PI_LR = base.PI_LR
VF_LR = base.VF_LR
LR_MIN = base.LR_MIN
GAMMA = base.GAMMA
LAMDA = base.LAMDA
K_EPOCHS = base.K_EPOCHS
EPS_CLIP = base.EPS_CLIP
VF_COEF = base.VF_COEF
ENT_COEF = base.ENT_COEF
SIGMA_FLOOR = base.SIGMA_FLOOR
SIGMA_FLOOR_COEF = base.SIGMA_FLOOR_COEF
ACT_SMOOTH = base.ACT_SMOOTH
REWARD_SCALE = base.REWARD_SCALE
MINI_BS = base.MINI_BS
CRITIC_WARMUP = base.CRITIC_WARMUP

BC_EPOCHS = base.BC_EPOCHS
BC_LR = base.BC_LR
BC_BS = base.BC_BS
BC_GRAD_CLIP = base.BC_GRAD_CLIP

CSVS_EPOCH = base.CSVS_EPOCH
SAMPLES_PER_ROUTE = base.SAMPLES_PER_ROUTE
MAX_EP = base.MAX_EP
EVAL_EVERY = base.EVAL_EVERY
EVAL_N = base.EVAL_N
RESUME = base.RESUME
RESUME_OPT = base.RESUME_OPT
RESUME_DS = base.RESUME_DS
RESET_CRITIC = base.RESET_CRITIC
RESUME_WARMUP = base.RESUME_WARMUP
LR_DECAY = base.LR_DECAY
DELTA_SCALE_DECAY = base.DELTA_SCALE_DECAY
REWARD_RMS_NORM = base.REWARD_RMS_NORM
ADV_NORM = base.ADV_NORM
COMPILE = base.COMPILE

C = base.C
H1 = base.H1
H2 = base.H2
F_LAT = base.F_LAT
F_ROLL = base.F_ROLL
F_V = base.F_V
F_A = base.F_A
OBS_DIM = base.OBS_DIM

ActorCritic = base.ActorCritic
pretrain_bc = base.pretrain_bc
lr_schedule = base.lr_schedule
RunningMeanStd = base.RunningMeanStd
_precompute_future_windows = base._precompute_future_windows
fill_obs = base.fill_obs

TEACHER = os.getenv("TEACHER", "1") == "1"
TEACHER_EVERY = int(os.getenv("TEACHER_EVERY", "8"))
TEACHER_H = int(os.getenv("TEACHER_H", "8"))
TEACHER_K = int(os.getenv("TEACHER_K", "32"))
TEACHER_ROUTES = int(os.getenv("TEACHER_ROUTES", "1024"))
TEACHER_COEF = float(os.getenv("TEACHER_COEF", "0.5"))
TEACHER_BS = int(os.getenv("TEACHER_BS", "8192"))
TEACHER_MIN_IMPROV = float(os.getenv("TEACHER_MIN_IMPROV", "0.0"))

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"


def _ring_to_seq(ring, head):
    split = head + 1
    if split >= HIST_LEN:
        return ring.clone()
    return torch.cat([ring[:, split:], ring[:, :split]], dim=1)


def _build_obs_teacher(
    target,
    current,
    roll_la,
    v_ego,
    a_ego,
    act_hist,
    lat_hist,
    err_sum,
    future_target,
    future_roll,
    future_v,
    future_a,
):
    current_f = current.float()
    target_f = target.float()
    roll_f = roll_la.float()
    v_f = v_ego.float()
    a_f = a_ego.float()
    prev_act = act_hist[:, -1]
    prev_act2 = act_hist[:, -2]
    prev_lat = lat_hist[:, -1]
    v2 = torch.clamp(v_f * v_f, min=1.0)
    k_tgt = (target_f - roll_f) / v2
    k_cur = (current_f - roll_f) / v2
    fp0 = future_target[:, 0]
    fric = torch.sqrt(current_f**2 + a_f**2) / 7.0
    ei = err_sum * (DEL_T / HIST_LEN)

    obs = torch.empty((target.shape[0], OBS_DIM), dtype=torch.float32, device="cuda")
    obs[:, 0] = target_f / S_LAT
    obs[:, 1] = current_f / S_LAT
    obs[:, 2] = (target_f - current_f) / S_LAT
    obs[:, 3] = k_tgt / S_CURV
    obs[:, 4] = k_cur / S_CURV
    obs[:, 5] = (k_tgt - k_cur) / S_CURV
    obs[:, 6] = v_f / S_VEGO
    obs[:, 7] = a_f / S_AEGO
    obs[:, 8] = roll_f / S_ROLL
    obs[:, 9] = prev_act / S_STEER
    obs[:, 10] = ei / S_LAT
    obs[:, 11] = (fp0 - target_f) / DEL_T / S_LAT
    obs[:, 12] = (current_f - prev_lat) / DEL_T / S_LAT
    obs[:, 13] = (prev_act - prev_act2) / DEL_T / S_STEER
    obs[:, 14] = fric
    obs[:, 15] = torch.clamp(1.0 - fric, min=0.0)

    obs[:, C:H1] = act_hist / S_STEER
    obs[:, H1:H2] = lat_hist / S_LAT
    obs[:, F_LAT:F_ROLL] = future_target / S_LAT
    obs[:, F_ROLL:F_V] = future_roll / S_ROLL
    obs[:, F_V:F_A] = future_v / S_VEGO
    obs[:, F_A:OBS_DIM] = future_a / S_AEGO
    obs.clamp_(-5.0, 5.0)
    return obs


def _teacher_targets(
    sim_ref,
    step_idx,
    ac,
    ds,
    obs_buf,
    h_act32,
    h_lat,
    h_error,
    err_sum,
    hist_head,
    future,
):
    if not TEACHER or TEACHER_COEF <= 0.0 or TEACHER_K < 2:
        return None
    if step_idx >= COST_END_IDX:
        return None
    if (step_idx - CONTROL_START_IDX) % max(TEACHER_EVERY, 1) != 0:
        return None

    horizon = min(TEACHER_H, sim_ref.T - step_idx)
    if horizon <= 0:
        return None

    n_routes = obs_buf.shape[0]
    m = min(TEACHER_ROUTES, n_routes)
    if m <= 0:
        return None

    ridx = torch.randperm(n_routes, device="cuda")[:m]
    k = TEACHER_K
    b = m * k
    base_mask = (torch.arange(b, device="cuda") % k) == 0
    rep = ridx.repeat_interleave(k)
    h = sim_ref._hist_len

    act_hist_prev = sim_ref.action_history[ridx, h - CONTEXT_LENGTH:h].clone().repeat_interleave(k, dim=0)
    state_hist_prev = sim_ref.state_history[ridx, h - CONTEXT_LENGTH:h, :].clone().repeat_interleave(k, dim=0)
    pred_hist_prev = sim_ref.current_lataccel_history[ridx, h - CONTEXT_LENGTH:h].clone().repeat_interleave(k, dim=0)

    act_hist_ctrl = _ring_to_seq(h_act32[ridx], hist_head).clone().repeat_interleave(k, dim=0)
    lat_hist_ctrl = _ring_to_seq(h_lat[ridx], hist_head).clone().repeat_interleave(k, dim=0)
    err_head = (hist_head + 1) % HIST_LEN
    err_hist = _ring_to_seq(h_error[ridx], err_head).clone().repeat_interleave(k, dim=0)
    err_sum_cur = err_sum[ridx].clone().repeat_interleave(k, dim=0)
    current = sim_ref.current_lataccel[ridx].clone().repeat_interleave(k, dim=0)

    first_raw = None
    total_cost = torch.zeros(b, dtype=torch.float64, device="cuda")

    with torch.inference_mode():
        current_obs = obs_buf[ridx].clone().repeat_interleave(k, dim=0)
        for j in range(horizon):
            target_j = sim_ref.data_gpu["target_lataccel"][rep, step_idx + j]
            roll_j = sim_ref.data_gpu["roll_lataccel"][rep, step_idx + j]
            v_j = sim_ref.data_gpu["v_ego"][rep, step_idx + j]
            a_j = sim_ref.data_gpu["a_ego"][rep, step_idx + j]

            if j == 0:
                obs_j = current_obs
            else:
                err_j = (target_j.float() - current.float())
                old_err = err_hist[:, 0]
                err_hist = torch.cat([err_hist[:, 1:], err_j.unsqueeze(1)], dim=1)
                err_sum_cur = err_sum_cur + err_j - old_err
                obs_j = _build_obs_teacher(
                    target=target_j,
                    current=current,
                    roll_la=roll_j,
                    v_ego=v_j,
                    a_ego=a_j,
                    act_hist=act_hist_ctrl,
                    lat_hist=lat_hist_ctrl,
                    err_sum=err_sum_cur,
                    future_target=future["target_lataccel"][rep, step_idx + j],
                    future_roll=future["roll_lataccel"][rep, step_idx + j],
                    future_v=future["v_ego"][rep, step_idx + j],
                    future_a=future["a_ego"][rep, step_idx + j],
                )

            logits = ac.actor(obs_j)
            alpha = F.softplus(logits[..., 0]) + 1.0
            beta = F.softplus(logits[..., 1]) + 1.0
            raw_mu = 2.0 * alpha / (alpha + beta) - 1.0

            if j == 0:
                dist = torch.distributions.Beta(alpha, beta)
                raw = 2.0 * dist.sample() - 1.0
                raw[base_mask] = raw_mu[base_mask]
                first_raw = raw.clone()
            else:
                raw = raw_mu

            prev_action = act_hist_ctrl[:, -1].double()
            current_before = current
            action = (prev_action + raw.double() * ds).clamp(STEER_RANGE[0], STEER_RANGE[1])
            state_now = torch.stack([roll_j, v_j, a_j], dim=1).double()
            actions_in = torch.cat([act_hist_prev[:, 1:], action.unsqueeze(1)], dim=1)
            states_in = torch.cat([state_hist_prev[:, 1:, :], state_now.unsqueeze(1)], dim=1)
            _, expected = sim_ref.sim_model.get_current_lataccel(
                sim_states=states_in,
                actions=actions_in,
                past_preds=pred_hist_prev,
                rng_u=None,
                return_expected=True,
            )
            pred = expected.clamp(current_before - MAX_ACC_DELTA, current_before + MAX_ACC_DELTA)

            lat_cost = (target_j.double() - pred).pow(2) * (100.0 * LAT_ACCEL_COST_MULTIPLIER)
            jerk = (pred - current_before) / DEL_T
            total_cost += lat_cost + jerk.pow(2) * 100.0
            if ACT_SMOOTH > 0.0:
                act_d = (action.float() - act_hist_ctrl[:, -1]) / DEL_T
                total_cost += act_d.double().pow(2) * ACT_SMOOTH

            current = pred
            act_hist_prev = actions_in
            state_hist_prev = states_in
            pred_hist_prev = torch.cat([pred_hist_prev[:, 1:], pred.unsqueeze(1)], dim=1)
            act_hist_ctrl = torch.cat([act_hist_ctrl[:, 1:], action.float().unsqueeze(1)], dim=1)
            lat_hist_ctrl = torch.cat([lat_hist_ctrl[:, 1:], current_before.float().unsqueeze(1)], dim=1)

    if first_raw is None:
        return None

    total_cost = total_cost.view(m, k)
    first_raw = first_raw.view(m, k)
    best_idx = total_cost.argmin(dim=1)
    row_idx = torch.arange(m, device="cuda")
    best_cost = total_cost[row_idx, best_idx]
    base_cost = total_cost[:, 0]
    improvement = (base_cost - best_cost).float().clamp_min(0.0)
    keep = improvement > TEACHER_MIN_IMPROV
    if not bool(keep.any()):
        return None

    best_raw = first_raw[row_idx, best_idx]
    weights = improvement[keep]
    weights = weights / (weights.mean() + 1e-8)
    return (
        obs_buf[ridx][keep].clone(),
        best_raw[keep].float().clone(),
        weights.float().clone(),
    )


def rollout(csv_files, ac, mdl_path, ort_session, csv_cache, deterministic=False, ds=DELTA_SCALE_MAX):
    t_setup0 = time.perf_counter()
    data, rng = csv_cache.slice(csv_files)
    t_slice = time.perf_counter()
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng)
    t_sim_init = time.perf_counter()
    n, _ = sim.N, sim.T
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    future = _precompute_future_windows(dg)
    t_future = time.perf_counter()

    h_act = torch.zeros((n, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((n, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((n, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((n, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(n, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((n, OBS_DIM), dtype=torch.float32, device="cuda")

    if not deterministic:
        all_obs = torch.empty((max_steps, n, OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw = torch.empty((max_steps, n), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((max_steps, n), dtype=torch.float32, device="cuda")
        all_val = torch.empty((max_steps, n), dtype=torch.float32, device="cuda")
        teacher_obs = []
        teacher_raw = []
        teacher_w = []

    if int(os.environ.get("DEBUG", "0")) >= 2:
        torch.cuda.synchronize()
        t_alloc = time.perf_counter()
        print(
            f"  [rollout setup N={n}] slice={t_slice-t_setup0:.3f}s  "
            f"sim_init={t_sim_init-t_slice:.3f}s  future={t_future-t_sim_init:.3f}s  "
            f"alloc={t_alloc-t_future:.3f}s",
            flush=True,
        )

    si = 0
    hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
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
        ei = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(n, dtype=h_act.dtype, device="cuda")

        fill_obs(
            obs_buf,
            target.float(),
            cur32,
            roll_la.float(),
            v_ego.float(),
            a_ego.float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )

        with torch.inference_mode():
            logits = ac.actor(obs_buf)
            val = ac.critic(obs_buf).squeeze(-1)

        if not deterministic:
            teacher_pack = _teacher_targets(
                sim_ref=sim_ref,
                step_idx=step_idx,
                ac=ac,
                ds=float(ds),
                obs_buf=obs_buf,
                h_act32=h_act32,
                h_lat=h_lat,
                h_error=h_error,
                err_sum=err_sum,
                hist_head=hist_head,
                future=future,
            )
            if teacher_pack is not None:
                teacher_obs.append(teacher_pack[0])
                teacher_raw.append(teacher_pack[1])
                teacher_w.append(teacher_pack[2])

        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0

        if deterministic:
            raw_policy = 2.0 * a_p / (a_p + b_p) - 1.0
            logp = None
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            x = dist.sample()
            raw_policy = 2.0 * x - 1.0
            logp = dist.log_prob(x)

        delta = raw_policy.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_raw[si] = raw_policy
            all_logp[si] = logp
            all_val[si] = val
            si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    if deterministic:
        return costs.tolist()

    s = si
    start = CONTROL_START_IDX
    end = start + s
    if sim._gpu:
        pred = sim.current_lataccel_history[:, start:end].float()
        target = dg["target_lataccel"][:, start:end].float()
        act = sim.action_history[:, start:end].float()
    else:
        pred = torch.from_numpy(sim.current_lataccel_history[:, start:end]).to(device="cuda", dtype=torch.float32)
        target = torch.from_numpy(sim.data["target_lataccel"][:, start:end]).to(device="cuda", dtype=torch.float32)
        act = torch.from_numpy(sim.action_history[:, start:end]).to(device="cuda", dtype=torch.float32)

    lat_r = (target - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    rew = (-(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(REWARD_SCALE, 1e-8)).float()
    dones = torch.zeros((n, s), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    out = dict(
        obs=all_obs[:s].permute(1, 0, 2).reshape(-1, OBS_DIM),
        raw=all_raw[:s].T.reshape(-1),
        old_logp=all_logp[:s].T.reshape(-1),
        val_2d=all_val[:s].T,
        rew=rew,
        done=dones,
        costs=costs,
    )
    if teacher_obs:
        out["teacher_obs"] = torch.cat(teacher_obs, dim=0)
        out["teacher_raw"] = torch.cat(teacher_raw, dim=0)
        out["teacher_w"] = torch.cat(teacher_w, dim=0)
    return out


class PPO(base.PPO):
    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX):
        info = super().update(gd, critic_only=critic_only, ds=ds)
        teacher_metric = 0.0

        if critic_only or TEACHER_COEF <= 0.0:
            info["teacher"] = teacher_metric
            return info

        t_obs = gd.get("teacher_obs")
        if t_obs is None or t_obs.numel() == 0:
            info["teacher"] = teacher_metric
            return info

        t_raw = gd["teacher_raw"]
        t_w = gd["teacher_w"]
        t_w = t_w / (t_w.mean() + 1e-8)
        t_w = t_w.clamp(max=10.0)

        total = 0.0
        n = 0
        for idx in torch.randperm(len(t_obs), device="cuda").split(TEACHER_BS):
            a_out = self.ac.actor(t_obs[idx])
            alpha = F.softplus(a_out[..., 0]) + 1.0
            beta = F.softplus(a_out[..., 1]) + 1.0
            x = ((t_raw[idx] + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)
            dist = torch.distributions.Beta(alpha, beta)
            loss = -(dist.log_prob(x) * t_w[idx]).mean()

            self.pi_opt.zero_grad(set_to_none=True)
            (TEACHER_COEF * loss).backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
            self.pi_opt.step()

            bs = int(idx.numel())
            total += loss.detach().item() * bs
            n += bs

        teacher_metric = total / max(1, n)
        info["teacher"] = teacher_metric
        return info


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
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
                ppo._rms.mean, ppo._rms.var, ppo._rms.count = r["mean"], r["var"], r["count"]
        elif RESUME_OPT:
            print("RESUME_OPT=1 but optimizer state missing in checkpoint; using fresh optimizer/RMS state")
        warmup_off = 0 if RESUME_WARMUP else CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
        if RESUME_OPT:
            print("RESUME_OPT=1: optimizer state, LR/eps, and RMS restored from checkpoint")
        else:
            print("RESUME_OPT=0: resumed weights only; optimizer and RMS use fresh state")
        if RESUME_DS:
            ds_ckpt = ckpt.get("delta_scale", None)
            if ds_ckpt is not None:
                resumed_ds = float(ds_ckpt)
                print(f"Resumed delta_scale={resumed_ds:.6f} from checkpoint")
            else:
                print("RESUME_DS=1 but checkpoint has no delta_scale; using schedule/env")
        if RESET_CRITIC:
            for layer in ac.critic[:-1]:
                if isinstance(layer, torch.nn.Linear):
                    base._ortho(layer)
            if isinstance(ac.critic[-1], torch.nn.Linear):
                base._ortho(ac.critic[-1], gain=1.0)
            ppo.vf_opt = torch.optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic, vf_opt, and ret_rms reset; critic warmup re-enabled")
    else:
        pretrain_bc(ac, all_csv)

    if COMPILE:
        ac.actor = torch.compile(ac.actor, mode="max-autotune-no-cudagraphs", dynamic=True)
        ac.critic = torch.compile(ac.critic, mode="max-autotune-no-cudagraphs", dynamic=True)

    ds_max_run = DELTA_SCALE_MAX
    ds_min_run = DELTA_SCALE_MIN
    if RESUME_DS and resumed_ds is not None:
        ds_max_run = resumed_ds
        ds_min_run = min(ds_min_run, ds_max_run)

    baseline_ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(0.0)) if DELTA_SCALE_DECAY else ds_max_run
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
                "ret_rms": {"mean": ppo._rms.mean, "var": ppo._rms.var, "count": ppo._rms.count},
                "delta_scale": cur_ds,
            },
            BEST_PT,
        )

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    n_roll = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
    print(f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {n_roll} routes × {SAMPLES_PER_ROUTE} = {n_roll * SAMPLES_PER_ROUTE} rollouts/epoch")
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
        f"  rew_scale={REWARD_SCALE:g}"
        f"  lr_decay={'on' if LR_DECAY else 'off'}"
        f"  resume_opt={'on' if RESUME_OPT else 'off'}"
        f"  reset_critic={'on' if RESET_CRITIC else 'off'}"
        f"  resume_warmup={'on' if RESUME_WARMUP else 'off'}"
        f"  σfloor_eff={SIGMA_FLOOR} coef={SIGMA_FLOOR_COEF}"
        f"  rew_rms_norm={'on' if REWARD_RMS_NORM else 'off'}"
        f"  adv_norm={'on' if ADV_NORM else 'off'}"
        f"  compile={'on' if COMPILE else 'off'}"
        f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}  K={K_EPOCHS}  dim={STATE_DIM}"
    )
    print(
        f"  teacher={'on' if TEACHER else 'off'}"
        f"  coef={TEACHER_COEF:g}  every={TEACHER_EVERY}  H={TEACHER_H}  K={TEACHER_K}"
        f"  routes={TEACHER_ROUTES}  bs={TEACHER_BS}\n"
    )

    for epoch in range(MAX_EP):
        if DELTA_SCALE_DECAY:
            ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(np.pi * epoch / MAX_EP))
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
        n_routes = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
        batch = random.sample(tr_f, max(n_routes, 1))
        batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]
        res = rollout(batch, ac, mdl_path, ort_sess, csv_cache, deterministic=False, ds=ds)

        t1 = time.time()
        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co, ds=ds)
        tu = time.time() - t1

        phase = "  [critic warmup]" if co else ""
        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}  σraw={info['σraw']:.4f}"
            f"  σpen={info['σpen']:.4f}  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  T={info.get('teacher', 0.0):.4f}  Δs={ds:.4f}  lr={info['lr']:.1e}  ⏱{t1-t0:.0f}+{tu:.0f}s{phase}"
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
    torch.save({"ac": ac.state_dict(), "delta_scale": cur_ds}, EXP_DIR / "final_model.pt")


if __name__ == "__main__":
    train()
