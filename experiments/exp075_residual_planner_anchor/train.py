# exp075 — frozen exp055 base actor + anchored residual planner
#
# Base path is exactly exp055 and stays frozen.
# A separate planner head proposes a bounded residual sequence over a short
# horizon. The current action executes base_raw + residual[0], while the tail
# persists in a small internal plan buffer appended to the planner observation.
# No BC path: this experiment is intentionally a warm-started surgery.

import os, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (  # noqa: E402
    ACC_G,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    DEL_T,
    FUTURE_PLAN_STEPS,
    LAT_ACCEL_COST_MULTIPLIER,
    STEER_RANGE,
    FuturePlan,
    State,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
BASE_OBS_DIM, HIDDEN = 256, 256
PLAN_H = int(os.getenv("PLAN_H", "6"))
PLAN_COMMIT = int(os.getenv("PLAN_COMMIT", "1"))
PLAN_BLEND = float(os.getenv("PLAN_BLEND", "0.75"))
MAX_PLAN_RESIDUAL = float(os.getenv("MAX_PLAN_RESIDUAL", "0.20"))
INIT_CONC = float(os.getenv("INIT_CONC", "40.0"))
A_LAYERS, C_LAYERS = 4, 4

if not (0 <= PLAN_COMMIT < PLAN_H):
    raise ValueError(f"PLAN_COMMIT must be in [0, PLAN_H), got {PLAN_COMMIT=} {PLAN_H=}")
if INIT_CONC <= 1.0:
    raise ValueError(f"INIT_CONC must be > 1.0, got {INIT_CONC}")
if MAX_PLAN_RESIDUAL < 0.0:
    raise ValueError(f"MAX_PLAN_RESIDUAL must be >= 0, got {MAX_PLAN_RESIDUAL}")

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── PPO ───────────────────────────────────────────────────────
PI_LR = float(os.getenv("PI_LR", "3e-4"))
VF_LR = float(os.getenv("VF_LR", "3e-4"))
LR_MIN = 5e-5
GAMMA = float(os.getenv("GAMMA", "0.95"))
LAMDA = float(os.getenv("LAMDA", "0.9"))
K_EPOCHS = int(os.getenv("K_EPOCHS", "4"))
EPS_CLIP = float(os.getenv("EPS_CLIP", "0.2"))
VF_COEF = float(os.getenv("VF_COEF", "1.0"))
ENT_COEF = float(os.getenv("ENT_COEF", "0.001"))
SIGMA_FLOOR = float(os.getenv("SIGMA_FLOOR", "0.0"))
SIGMA_FLOOR_COEF = float(os.getenv("SIGMA_FLOOR_COEF", "0.0"))
RESIDUAL_L2_COEF = float(os.getenv("RESIDUAL_L2_COEF", "1.0"))
ACT_SMOOTH = float(os.getenv("ACT_SMOOTH", "0.0"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1.0"))
MINI_BS = int(os.getenv("MINI_BS", "25_000"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))

# ── runtime ───────────────────────────────────────────────────
DELTA_SCALE_MAX = float(os.getenv("DELTA_SCALE_MAX", "0.25"))
DELTA_SCALE_MIN = float(os.getenv("DELTA_SCALE_MIN", "0.25"))
CSVS_EPOCH = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "10"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = int(os.getenv("EVAL_N", "100"))
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
BASE_INIT = os.getenv("BASE_INIT", "1") == "1"

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = Path(os.getenv("BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")))

# ── obs layout offsets ────────────────────────────────────────
C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
PLAN_OBS_DIM = BASE_OBS_DIM + PLAN_H + 1


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════

def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


def _mlp(input_dim, out_dim, layers):
    seq = [nn.Linear(input_dim, HIDDEN), nn.ReLU()]
    for _ in range(layers - 1):
        seq += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
    seq.append(nn.Linear(HIDDEN, out_dim))
    return nn.Sequential(*seq)


def _safe_beta_params_from_logits(logits):
    a = torch.nan_to_num(F.softplus(logits[..., 0]) + 1.0, nan=1.0, posinf=1000.0, neginf=1.0)
    b = torch.nan_to_num(F.softplus(logits[..., 1]) + 1.0, nan=1.0, posinf=1000.0, neginf=1.0)
    return a.clamp_min(1e-4), b.clamp_min(1e-4)


def _safe_beta_dist(a, b):
    a = torch.nan_to_num(a, nan=1.0, posinf=1000.0, neginf=1.0).clamp_min(1e-4)
    b = torch.nan_to_num(b, nan=1.0, posinf=1000.0, neginf=1.0).clamp_min(1e-4)
    return torch.distributions.Beta(a, b, validate_args=False)


def _softplus_inverse(y):
    return float(np.log(np.expm1(max(y, 1e-6))))


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_actor = _mlp(BASE_OBS_DIM, 2, A_LAYERS)
        self.planner_actor = _mlp(PLAN_OBS_DIM, 2 * PLAN_H, A_LAYERS)
        self.critic = _mlp(PLAN_OBS_DIM, 1, C_LAYERS)

        for layer in self.base_actor[:-1]:
            _ortho(layer)
        _ortho(self.base_actor[-1], gain=0.01)

        for layer in self.planner_actor[:-1]:
            _ortho(layer)
        for layer in self.critic[:-1]:
            _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

        with torch.no_grad():
            self.planner_actor[-1].weight.zero_()
            self.planner_actor[-1].bias.fill_(_softplus_inverse(INIT_CONC - 1.0))

    def base_beta_params(self, base_obs):
        return _safe_beta_params_from_logits(self.base_actor(base_obs))

    def planner_beta_params(self, plan_obs):
        out = self.planner_actor(plan_obs).view(-1, PLAN_H, 2)
        return _safe_beta_params_from_logits(out)


def _freeze_base_actor(ac):
    ac.base_actor.requires_grad_(False)
    ac.base_actor.eval()


def _copy_exp055_actor(ac, state):
    actor_sd = {k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")}
    ac.base_actor.load_state_dict(actor_sd, strict=True)


def _copy_exp055_critic(ac, state):
    with torch.no_grad():
        ac.critic[0].weight[:, :BASE_OBS_DIM].copy_(state["critic.0.weight"])
        ac.critic[0].weight[:, BASE_OBS_DIM:].zero_()
        ac.critic[0].bias.copy_(state["critic.0.bias"])
        for idx in (2, 4, 6, 8):
            ac.critic[idx].weight.copy_(state[f"critic.{idx}.weight"])
            ac.critic[idx].bias.copy_(state[f"critic.{idx}.bias"])


def init_from_exp055(ac):
    if not BASE_PT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_PT}")
    ckpt = torch.load(BASE_PT, weights_only=False, map_location="cpu")
    state = ckpt["ac"]
    _copy_exp055_actor(ac, state)
    _copy_exp055_critic(ac, state)
    _freeze_base_actor(ac)
    return float(ckpt.get("delta_scale", DELTA_SCALE_MAX))


# ══════════════════════════════════════════════════════════════
#  Observation builder
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


def _proposal_next(proposal):
    nxt = torch.empty_like(proposal)
    nxt[:, :-1] = proposal[:, 1:]
    nxt[:, -1] = proposal[:, -1]
    return nxt


def _shift_plan(plan):
    shifted = torch.empty_like(plan)
    shifted[:, :-1] = plan[:, 1:]
    shifted[:, -1] = plan[:, -1]
    return shifted


def _plan_active_mask(primed, device):
    mask = torch.zeros((primed.shape[0], PLAN_H), dtype=torch.float32, device=device)
    mask[:, 0] = 1.0
    if PLAN_COMMIT + 1 < PLAN_H:
        mask[:, PLAN_COMMIT + 1 :] = 1.0
    mask[~primed] = 1.0
    return mask


def fill_base_obs(
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
    buf[:, F_A:BASE_OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    buf.clamp_(-5.0, 5.0)


def fill_plan_obs(buf, base_obs, plan_resid, base_raw):
    buf[:, :BASE_OBS_DIM] = base_obs
    scale = max(MAX_PLAN_RESIDUAL, 1e-6)
    buf[:, BASE_OBS_DIM : BASE_OBS_DIM + PLAN_H] = plan_resid / scale
    buf[:, BASE_OBS_DIM + PLAN_H] = base_raw
    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  GPU Rollout
# ══════════════════════════════════════════════════════════════

def rollout(csv_files, ac, mdl_path, ort_session, csv_cache, deterministic=False, ds=DELTA_SCALE_MAX):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng)
    N = sim.N
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    plan_resid = torch.zeros((N, PLAN_H), dtype=torch.float32, device="cuda")
    plan_primed = torch.zeros(N, dtype=torch.bool, device="cuda")
    base_obs_buf = torch.empty((N, BASE_OBS_DIM), dtype=torch.float32, device="cuda")
    plan_obs_buf = torch.empty((N, PLAN_OBS_DIM), dtype=torch.float32, device="cuda")

    if not deterministic:
        all_obs = torch.empty((max_steps, N, PLAN_OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw = torch.empty((max_steps, N, PLAN_H), dtype=torch.float32, device="cuda")
        all_mask = torch.empty((max_steps, N, PLAN_H), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_val = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")

    si = 0
    hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum, plan_resid, plan_primed
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
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        fill_base_obs(
            base_obs_buf,
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
            base_a, base_b = ac.base_beta_params(base_obs_buf)
            base_raw = (2.0 * base_a / (base_a + base_b) - 1.0).float()
            fill_plan_obs(plan_obs_buf, base_obs_buf, plan_resid, base_raw)
            planner_a, planner_b = ac.planner_beta_params(plan_obs_buf)
            val = ac.critic(plan_obs_buf).squeeze(-1)

        if deterministic:
            resid_raw = 2.0 * planner_a / (planner_a + planner_b) - 1.0
            mask = None
            logp = None
        else:
            dist = _safe_beta_dist(planner_a, planner_b)
            x = dist.sample()
            resid_raw = 2.0 * x - 1.0
            mask = _plan_active_mask(plan_primed, resid_raw.device)
            logp = (dist.log_prob(x) * mask).sum(dim=-1)

        resid_eff = resid_raw * MAX_PLAN_RESIDUAL
        raw = (base_raw + resid_eff[:, 0]).clamp(-1.0, 1.0).to(h_act.dtype)
        delta = raw * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        proposal_next = _proposal_next(resid_eff)
        shifted_plan = _shift_plan(plan_resid)
        next_plan = proposal_next.clone()
        if plan_primed.any():
            next_plan[plan_primed] = shifted_plan[plan_primed]
            if PLAN_COMMIT < PLAN_H:
                next_plan[plan_primed, PLAN_COMMIT:] = (
                    (1.0 - PLAN_BLEND) * shifted_plan[plan_primed, PLAN_COMMIT:]
                    + PLAN_BLEND * proposal_next[plan_primed, PLAN_COMMIT:]
                )
        plan_resid = next_plan
        plan_primed = torch.ones_like(plan_primed)

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[si] = plan_obs_buf
            all_raw[si] = resid_raw
            all_mask[si] = mask
            all_logp[si] = logp
            all_val[si] = val
            si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    if deterministic:
        return costs.tolist()

    S = si
    start = CONTROL_START_IDX
    end = start + S
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
    dones = torch.zeros((N, S), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    return dict(
        obs=all_obs[:S].permute(1, 0, 2).reshape(-1, PLAN_OBS_DIM),
        raw=all_raw[:S].permute(1, 0, 2).reshape(-1, PLAN_H),
        mask=all_mask[:S].permute(1, 0, 2).reshape(-1, PLAN_H),
        old_logp=all_logp[:S].T.reshape(-1),
        val_2d=all_val[:S].T,
        rew=rew,
        done=dones,
        costs=costs,
    )


# ══════════════════════════════════════════════════════════════
#  PPO
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
        self.pi_opt = optim.Adam(ac.planner_actor.parameters(), lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._rms = RunningMeanStd()

    @staticmethod
    def _beta_sigma_raw(alpha, beta):
        return 2.0 * torch.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

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
        raw = gd["raw"]
        mask = gd["mask"]
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])
        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_2d = adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)

        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)

        with torch.no_grad():
            if "old_logp" in gd:
                old_lp = gd["old_logp"]
            else:
                a_old, b_old = self.ac.planner_beta_params(obs)
                old_lp = (_safe_beta_dist(a_old, b_old).log_prob(x_t) * mask).sum(dim=-1)

        n_vf = 0
        n_actor = 0
        vf_sum = 0.0
        pi_sum = 0.0
        ent_sum = 0.0
        sigma_pen_sum = 0.0
        resid_l2_sum = 0.0

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
                    continue

                a_c, b_c = self.ac.planner_beta_params(obs[idx])
                dist = _safe_beta_dist(a_c, b_c)
                mb_mask = mask[idx]
                active = mb_mask.sum(dim=-1).clamp_min(1.0)
                lp = (dist.log_prob(x_t[idx]) * mb_mask).sum(dim=-1)
                ratio = (lp - old_lp[idx]).exp()
                pi_loss = -torch.min(
                    ratio * mb_adv,
                    ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv,
                ).mean()
                ent = ((dist.entropy() * mb_mask).sum(dim=-1) / active).mean()

                mu_raw = 2.0 * a_c / (a_c + b_c) - 1.0
                mu_eff = mu_raw * MAX_PLAN_RESIDUAL
                resid_l2 = (((mu_eff.square()) * mb_mask).sum(dim=-1) / active).mean()

                sigma_raw = self._beta_sigma_raw(a_c, b_c)
                sigma_raw_mb = ((sigma_raw * mb_mask).sum(dim=-1) / active).mean()
                sigma_eff_mb = sigma_raw_mb * MAX_PLAN_RESIDUAL * ds
                sigma_pen = F.relu(SIGMA_FLOOR - sigma_eff_mb)

                loss = (
                    pi_loss
                    + VF_COEF * vf_loss
                    - ENT_COEF * ent
                    + SIGMA_FLOOR_COEF * sigma_pen
                    + RESIDUAL_L2_COEF * resid_l2
                )

                self.pi_opt.zero_grad(set_to_none=True)
                self.vf_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.planner_actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                self.pi_opt.step()
                self.vf_opt.step()

                pi_sum += pi_loss.detach().item() * bs
                ent_sum += ent.detach().item() * bs
                sigma_pen_sum += sigma_pen.detach().item() * bs
                resid_l2_sum += resid_l2.detach().item() * bs
                n_actor += bs

        with torch.no_grad():
            diag_obs = obs[:1000]
            diag_mask = mask[:1000]
            a_d, b_d = self.ac.planner_beta_params(diag_obs)
            active = diag_mask.sum(dim=-1).clamp_min(1.0)
            sigma_raw = ((self._beta_sigma_raw(a_d, b_d) * diag_mask).sum(dim=-1) / active).mean().item()
            mu_raw = ((torch.abs(2.0 * a_d / (a_d + b_d) - 1.0) * diag_mask).sum(dim=-1) / active).mean().item()

        return dict(
            pi=(pi_sum / max(1, n_actor)) if not critic_only else 0.0,
            vf=vf_sum / max(1, n_vf),
            ent=(ent_sum / max(1, n_actor)) if not critic_only else 0.0,
            sigma=sigma_raw * MAX_PLAN_RESIDUAL * ds,
            sigma_raw=sigma_raw,
            sigma_pen=(sigma_pen_sum / max(1, n_actor)) if not critic_only else 0.0,
            resid_l2=(resid_l2_sum / max(1, n_actor)) if not critic_only else 0.0,
            resid_abs=mu_raw * MAX_PLAN_RESIDUAL,
            lr=self.pi_opt.param_groups[0]["lr"],
        )


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════

def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
    return float(np.mean(costs)), float(np.std(costs))


def train():
    ac = ActorCritic().to(DEV)
    _freeze_base_actor(ac)
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
        _freeze_base_actor(ac)
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
            _freeze_base_actor(ac)
            ppo.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic, vf_opt, and ret_rms reset; critic warmup re-enabled")
    else:
        if not BASE_INIT:
            raise ValueError("exp075 is a warm-start experiment; use BASE_INIT=1 or RESUME=1")
        init_ds = init_from_exp055(ac)
        _freeze_base_actor(ac)
        print(f"BASE_INIT=1 loaded {BASE_PT}; no BC pretrain")
        if RESUME_DS:
            resumed_ds = init_ds

    if COMPILE:
        ac.planner_actor = torch.compile(ac.planner_actor, mode="max-autotune-no-cudagraphs", dynamic=True)
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
                "plan_h": PLAN_H,
                "plan_commit": PLAN_COMMIT,
                "plan_blend": PLAN_BLEND,
                "max_plan_residual": MAX_PLAN_RESIDUAL,
            },
            BEST_PT,
        )

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    n_routes = min(CSVS_EPOCH, len(tr_f)) // SAMPLES_PER_ROUTE
    print(f"  batch_of_batch: K={SAMPLES_PER_ROUTE}  → {n_routes} routes × {SAMPLES_PER_ROUTE} = {n_routes * SAMPLES_PER_ROUTE} rollouts/epoch")
    print(
        f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
        f"  rew_scale={REWARD_SCALE:g}"
        f"  lr_decay={'on' if LR_DECAY else 'off'}"
        f"  resume_opt={'on' if RESUME_OPT else 'off'}"
        f"  reset_critic={'on' if RESET_CRITIC else 'off'}"
        f"  resume_warmup={'on' if RESUME_WARMUP else 'off'}"
        f"  σfloor_eff={SIGMA_FLOOR} coef={SIGMA_FLOOR_COEF}"
        f"  resid_l2={RESIDUAL_L2_COEF}"
        f"  rew_rms_norm={'on' if REWARD_RMS_NORM else 'off'}"
        f"  adv_norm={'on' if ADV_NORM else 'off'}"
        f"  compile={'on' if COMPILE else 'off'}"
        f"  plan_h={PLAN_H} commit={PLAN_COMMIT} blend={PLAN_BLEND}"
        f"  max_resid={MAX_PLAN_RESIDUAL}"
        f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}"
        f"  K={K_EPOCHS}  base_dim={BASE_OBS_DIM}  planner_dim={PLAN_OBS_DIM}\n"
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

        critic_only = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=critic_only, ds=ds)
        tu = time.time() - t1

        phase = "  [critic warmup]" if critic_only else ""
        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}"
            f"  σ={info['sigma']:.4f}  σraw={info['sigma_raw']:.4f}"
            f"  σpen={info['sigma_pen']:.4f}"
            f"  rmag={info['resid_abs']:.4f}  rpen={info['resid_l2']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  Δs={ds:.4f}  lr={info['lr']:.1e}  ⏱{t1 - t0:.0f}+{tu:.0f}s{phase}"
        )

        if epoch % EVAL_EVERY == 0:
            vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
            mark = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mark = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}{mark}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    torch.save(
        {
            "ac": ac.state_dict(),
            "delta_scale": cur_ds,
            "plan_h": PLAN_H,
            "plan_commit": PLAN_COMMIT,
            "plan_blend": PLAN_BLEND,
            "max_plan_residual": MAX_PLAN_RESIDUAL,
        },
        EXP_DIR / "final_model.pt",
    )


if __name__ == "__main__":
    train()
