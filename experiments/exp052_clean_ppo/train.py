# exp052 — GPU-only Beta PPO with NLL BC pretrain
#
# 256-dim obs: 16 core + 20 h_act + 20 h_lat + 50×4 future
# Delta actions: steer = prev + delta * DELTA_SCALE
# All tensors GPU-resident.  No CPU fallback.  No remote.  No workers.

import numpy as np, pandas as pd, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    ACC_G, State, FuturePlan)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEV = torch.device('cuda')

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN   = 256, 256
A_LAYERS, C_LAYERS  = 4, 4
DELTA_SCALE_MAX     = float(os.getenv('DELTA_SCALE_MAX', '0.25'))
DELTA_SCALE_MIN     = float(os.getenv('DELTA_SCALE_MIN', '0.25'))

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── PPO ───────────────────────────────────────────────────────
PI_LR      = float(os.getenv('PI_LR', '3e-4'))
VF_LR      = float(os.getenv('VF_LR', '3e-4'))
LR_MIN     = 5e-5
GAMMA       = float(os.getenv('GAMMA', '0.95'))
LAMDA       = float(os.getenv('LAMDA', '0.9'))
K_EPOCHS    = 4
EPS_CLIP    = 0.2
VF_COEF     = 1.0
ENT_COEF    = float(os.getenv('ENT_COEF', '0.003'))
# Floor is enforced on effective delta-action sigma (post DELTA_SCALE).
SIGMA_FLOOR = float(os.getenv('SIGMA_FLOOR', '0.005'))
SIGMA_FLOOR_COEF = float(os.getenv('SIGMA_FLOOR_COEF', '0.60'))
ACT_SMOOTH  = float(os.getenv('ACT_SMOOTH', '0.0'))
REWARD_SCALE = float(os.getenv('REWARD_SCALE', '1.0'))
MINI_BS     = int(os.getenv('MINI_BS', '25_000'))
CRITIC_WARMUP = int(os.getenv('CRITIC_WARMUP', '3'))

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS   = int(os.getenv('BC_EPOCHS', '20'))
BC_LR       = float(os.getenv('BC_LR', '0.01'))
BC_BS       = int(os.getenv('BC_BS', '2048'))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv('CSVS', '5000'))
MAX_EP     = int(os.getenv('EPOCHS', '5000'))
EVAL_EVERY = 5
EVAL_N     = 100  # files to use for val metrics (subset of full, not held out)
RESUME     = os.getenv('RESUME', '0') == '1'
RESUME_OPT = os.getenv('RESUME_OPT', '1') == '1'
RESUME_DS  = os.getenv('RESUME_DS', '0') == '1'
RESET_CRITIC = os.getenv('RESET_CRITIC', '0') == '1'
RESUME_WARMUP = os.getenv('RESUME_WARMUP', '0') == '1'
LR_DECAY = os.getenv('LR_DECAY', '1') == '1'
DELTA_SCALE_DECAY = os.getenv('DELTA_SCALE_DECAY', '0') == '1'
REWARD_RMS_NORM = os.getenv('REWARD_RMS_NORM', '1') == '1'
ADV_NORM = os.getenv('ADV_NORM', '1') == '1'
def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / 'best_model.pt'

# ── obs layout offsets ────────────────────────────────────────
C     = 16
H1    = C + HIST_LEN          # 36
H2    = H1 + HIST_LEN         # 56
F_LAT = H2                    # 56
F_ROLL = F_LAT + FUTURE_K     # 106
F_V    = F_ROLL + FUTURE_K    # 156
F_A    = F_V + FUTURE_K       # 206
OBS_DIM = F_A + FUTURE_K      # 256


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

        for layer in self.actor[:-1]: _ortho(layer)
        _ortho(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]: _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

    def beta_params(self, obs):
        out = self.actor(obs)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0


# ══════════════════════════════════════════════════════════════
#  Observation builder (GPU, batched)
# ══════════════════════════════════════════════════════════════

def fill_obs(buf, target, current, roll_la, v_ego, a_ego,
             h_act, h_lat, error_integral, dg, step_idx, T):
    v2   = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0  = dg['target_lataccel'][:, min(step_idx + 1, T - 1)]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0

    buf[:, 0]  = target / S_LAT
    buf[:, 1]  = current / S_LAT
    buf[:, 2]  = (target - current) / S_LAT
    buf[:, 3]  = k_tgt / S_CURV
    buf[:, 4]  = k_cur / S_CURV
    buf[:, 5]  = (k_tgt - k_cur) / S_CURV
    buf[:, 6]  = v_ego / S_VEGO
    buf[:, 7]  = a_ego / S_AEGO
    buf[:, 8]  = roll_la / S_ROLL
    buf[:, 9]  = h_act[:, -1] / S_STEER
    buf[:, 10] = error_integral / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - h_lat[:, -1]) / DEL_T / S_LAT
    buf[:, 13] = (h_act[:, -1] - h_act[:, -2]) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)

    buf[:, C:H1]  = h_act / S_STEER
    buf[:, H1:H2] = h_lat / S_LAT

    end = min(step_idx + FUTURE_PLAN_STEPS, T)
    for off, key, sc in [(F_LAT, 'target_lataccel', S_LAT),
                         (F_ROLL, 'roll_lataccel', S_ROLL),
                         (F_V, 'v_ego', S_VEGO),
                         (F_A, 'a_ego', S_AEGO)]:
        slc = dg[key][:, step_idx+1:end]
        w = slc.shape[1]
        if w == 0:
            buf[:, off:off+FUTURE_K] = (dg[key][:, step_idx] / sc).float().unsqueeze(1)
        elif w < FUTURE_K:
            buf[:, off:off+w] = slc.float() / sc
            buf[:, off+w:off+FUTURE_K] = slc[:, -1:].float() / sc
        else:
            buf[:, off:off+FUTURE_K] = slc[:, :FUTURE_K].float() / sc

    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  GPU Rollout
# ══════════════════════════════════════════════════════════════

def rollout(csv_files, ac, mdl_path, ort_session, csv_cache, deterministic=False, ds=DELTA_SCALE_MAX):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session,
                           cached_data=data, cached_rng=rng)
    N, T = sim.N, sim.T
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX

    h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device='cuda')

    if not deterministic:
        all_obs = torch.empty((max_steps, N, OBS_DIM), dtype=torch.float32, device='cuda')
        all_raw = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        all_logp = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        all_val = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si
        target  = dg['target_lataccel'][:, step_idx]
        current = sim_ref.current_lataccel
        roll_la = dg['roll_lataccel'][:, step_idx]
        v_ego   = dg['v_ego'][:, step_idx]
        a_ego   = dg['a_ego'][:, step_idx]

        cur32 = current.float()
        error = (target - current).float()
        h_error[:, :-1] = h_error[:, 1:]; h_error[:, -1] = error
        ei = h_error.mean(dim=1) * DEL_T

        if step_idx < CONTROL_START_IDX:
            h_act[:, :-1] = h_act[:, 1:]; h_act[:, -1] = 0.0
            h_act32[:, :-1] = h_act32[:, 1:]; h_act32[:, -1] = 0.0
            h_lat[:, :-1] = h_lat[:, 1:]; h_lat[:, -1] = cur32
            return torch.zeros(N, dtype=torch.float64, device='cuda')

        fill_obs(obs_buf, target.float(), cur32, roll_la.float(), v_ego.float(),
                 a_ego.float(), h_act32, h_lat, ei, dg, step_idx, T)

        with torch.inference_mode():
            logits = ac.actor(obs_buf)
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            val = ac.critic(obs_buf).squeeze(-1)

        if deterministic:
            raw_policy = 2.0 * a_p / (a_p + b_p) - 1.0
            logp = None
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            x = dist.sample()
            raw_policy = 2.0 * x - 1.0
            logp = dist.log_prob(x)

        delta  = raw_policy.double() * ds
        action = (h_act[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, :-1] = h_act[:, 1:]; h_act[:, -1] = action
        h_act32[:, :-1] = h_act32[:, 1:]; h_act32[:, -1] = action.float()
        h_lat[:, :-1] = h_lat[:, 1:]; h_lat[:, -1] = cur32

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[si] = obs_buf; all_raw[si] = raw_policy; all_logp[si] = logp; all_val[si] = val
            si += 1
        return action

    costs = sim.rollout(ctrl)['total_cost']

    if deterministic:
        return costs.tolist()

    # Align reward timing with official cost window using post-step simulator histories.
    S = si
    start = CONTROL_START_IDX
    end = start + S
    if sim._gpu:
        pred = sim.current_lataccel_history[:, start:end].float()
        target = dg['target_lataccel'][:, start:end].float()
        act = sim.action_history[:, start:end].float()
    else:
        pred = torch.from_numpy(sim.current_lataccel_history[:, start:end]).to(device='cuda', dtype=torch.float32)
        target = torch.from_numpy(sim.data['target_lataccel'][:, start:end]).to(device='cuda', dtype=torch.float32)
        act = torch.from_numpy(sim.action_history[:, start:end]).to(device='cuda', dtype=torch.float32)

    lat_r = (target - pred)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    # Keep finite-difference terms strictly within [CONTROL_START_IDX, COST_END_IDX).
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    rew = (-(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / max(REWARD_SCALE, 1e-8)).float()
    dones = torch.zeros((N, S), dtype=torch.float32, device='cuda')
    dones[:, -1] = 1.0

    return dict(
        obs=all_obs[:S].permute(1, 0, 2).reshape(-1, OBS_DIM),
        raw=all_raw[:S].T.reshape(-1),
        old_logp=all_logp[:S].T.reshape(-1),
        val_2d=all_val[:S].T,
        rew=rew, done=dones, costs=costs)


# ══════════════════════════════════════════════════════════════
#  BC Pretrain (actor-only NLL, CPU extraction — proven 83.3 path)
# ══════════════════════════════════════════════════════════════

def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)


def _build_obs_bc(target, current, state, fplan,
                  hist_act, hist_lat):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego ** 2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego ** 2, 1.0)
    _flat = getattr(fplan, 'lataccel', None)
    fp0 = _flat[0] if (_flat and len(_flat) > 0) else target
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0

    core = np.array([
        target / S_LAT, current / S_LAT, (target - current) / S_LAT,
        k_tgt / S_CURV, k_cur / S_CURV, (k_tgt - k_cur) / S_CURV,
        state.v_ego / S_VEGO, state.a_ego / S_AEGO,
        state.roll_lataccel / S_ROLL, hist_act[-1] / S_STEER,
        0.0,
        (fp0 - target) / DEL_T / S_LAT,
        (current - hist_lat[-1]) / DEL_T / S_LAT,
        (hist_act[-1] - hist_act[-2]) / DEL_T / S_STEER,
        fric, max(0.0, 1.0 - fric),
    ], dtype=np.float32)

    obs = np.concatenate([
        core,
        np.array(hist_act, np.float32) / S_STEER,
        np.array(hist_lat, np.float32) / S_LAT,
        _future_raw(fplan, 'lataccel', target) / S_LAT,
        _future_raw(fplan, 'roll_lataccel', state.roll_lataccel) / S_ROLL,
        _future_raw(fplan, 'v_ego', state.v_ego) / S_VEGO,
        _future_raw(fplan, 'a_ego', state.a_ego) / S_AEGO,
    ])
    return np.clip(obs, -5.0, 5.0)


def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df['roll'].values) * ACC_G
    v_ego   = df['vEgo'].values
    a_ego   = df['aEgo'].values
    tgt     = df['targetLateralAcceleration'].values
    steer   = -df['steerCommand'].values

    obs_list, raw_list = [], []
    h_act = [0.0] * HIST_LEN
    h_lat = [0.0] * HIST_LEN

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        target_la = tgt[step_idx]
        state = State(roll_lataccel=roll_la[step_idx],
                      v_ego=v_ego[step_idx], a_ego=a_ego[step_idx])
        fplan = FuturePlan(
            lataccel=tgt[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist())

        obs = _build_obs_bc(target_la, target_la, state, fplan, h_act, h_lat)
        raw_target = np.clip((steer[step_idx] - h_act[-1]) / DELTA_SCALE_MAX, -1.0, 1.0)
        obs_list.append(obs)
        raw_list.append(raw_target)
        h_act = h_act[1:] + [steer[step_idx]]
        h_lat = h_lat[1:] + [tgt[step_idx]]

    return (np.array(obs_list, np.float32), np.array(raw_list, np.float32))


def pretrain_bc(ac, all_csvs):
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(_bc_worker, [str(f) for f in all_csvs],
                          max_workers=10, chunksize=50, disable=False)
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
            a_p, b_p = ac.beta_params(obs_t[idx])
            x = ((raw_t[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        print(f"  BC epoch {ep}: loss={total/nb:.6f}  lr={opt.param_groups[0]['lr']:.1e}")
    print("BC pretrain done.\n")


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
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot
    @property
    def std(self): return np.sqrt(self.var + 1e-8)


class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(ac.actor.parameters(), lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._rms = RunningMeanStd()

    @staticmethod
    def _beta_sigma_raw(alpha, beta):
        # Beta std in [0,1] mapped to raw action space [-1,1]
        return 2.0 * torch.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

    def _gae(self, rew, val, done):
        if REWARD_RMS_NORM:
            with torch.no_grad():
                flat = rew.reshape(-1)
                self._rms.update(flat.mean().item(), flat.var().item(), flat.numel())
            rew = rew / max(self._rms.std, 1e-8)
        N, S = rew.shape
        adv = torch.empty_like(rew)
        g = torch.zeros(N, dtype=torch.float32, device='cuda')
        for t in range(S - 1, -1, -1):
            nv = val[:, t+1] if t < S-1 else g
            mask = 1.0 - done[:, t]
            g = (rew[:, t] + GAMMA * nv * mask - val[:, t]) + GAMMA * LAMDA * mask * g
            adv[:, t] = g
        return adv.reshape(-1), (adv + val).reshape(-1)

    def update(self, gd, critic_only=False, ds=DELTA_SCALE_MAX):
        obs = gd['obs']
        raw = gd['raw'].unsqueeze(-1)
        adv_t, ret_t = self._gae(gd['rew'], gd['val_2d'], gd['done'])
        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        ds = float(ds)
        sigma_pen = torch.tensor(0.0, device='cuda')

        # Weighted means over all optimizer minibatches for stable diagnostics.
        n_vf = 0
        n_actor = 0
        vf_sum = 0.0
        pi_sum = 0.0
        ent_sum = 0.0
        sigma_pen_sum = 0.0

        with torch.no_grad():
            if 'old_logp' in gd:
                old_lp = gd['old_logp']
            else:
                a_old, b_old = self.ac.beta_params(obs)
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t.squeeze(-1))

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device='cuda').split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                val = self.ac.critic(obs[idx]).squeeze(-1)
                vf_loss = F.mse_loss(val, ret_t[idx], reduction='mean')
                bs = int(idx.numel())
                vf_sum += vf_loss.detach().item() * bs
                n_vf += bs

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                else:
                    a_c, b_c = self.ac.beta_params(obs[idx])
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv,
                        ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv).mean()
                    ent = dist.entropy().mean()
                    sigma_raw_mb = self._beta_sigma_raw(a_c, b_c).mean()
                    sigma_eff_mb = sigma_raw_mb * ds
                    sigma_pen = F.relu(SIGMA_FLOOR - sigma_eff_mb)
                    loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent + SIGMA_FLOOR_COEF * sigma_pen
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.pi_opt.step(); self.vf_opt.step()
                    pi_sum += pi_loss.detach().item() * bs
                    ent_sum += ent.detach().item() * bs
                    sigma_pen_sum += sigma_pen.detach().item() * bs
                    n_actor += bs

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs[:1000])
            sigma_raw = self._beta_sigma_raw(a_d, b_d).mean().item()
            sigma_eff = sigma_raw * ds
        return dict(
            pi=(pi_sum / max(1, n_actor)) if not critic_only else 0.0,
            vf=(vf_sum / max(1, n_vf)),
            ent=(ent_sum / max(1, n_actor)) if not critic_only else 0.0,
            σ=sigma_eff, σraw=sigma_raw,
            σpen=(sigma_pen_sum / max(1, n_actor)) if not critic_only else 0.0,
            lr=self.pi_opt.param_groups[0]['lr'])


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════

def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
    return float(np.mean(costs)), float(np.std(costs))


def train():
    ac = ActorCritic().to(DEV)
    ppo = PPO(ac)
    mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
    ort_sess = make_ort_session(mdl_path)

    all_csv = sorted((ROOT / 'data').glob('*.csv'))
    tr_f = all_csv  # use all for training
    va_f = all_csv[:EVAL_N]  # eval on first 100 (sorted order)
    csv_cache = CSVCache([str(f) for f in all_csv])

    warmup_off = 0
    resumed_ds = None
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt['ac'])
        if RESUME_OPT and 'pi_opt' in ckpt:
            ppo.pi_opt.load_state_dict(ckpt['pi_opt'])
            ppo.vf_opt.load_state_dict(ckpt['vf_opt'])
            if 'ret_rms' in ckpt:
                r = ckpt['ret_rms']
                ppo._rms.mean, ppo._rms.var, ppo._rms.count = r['mean'], r['var'], r['count']
        elif RESUME_OPT:
            print("RESUME_OPT=1 but optimizer state missing in checkpoint; using fresh optimizer/RMS state")
        warmup_off = 0 if RESUME_WARMUP else CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
        if RESUME_OPT:
            print("RESUME_OPT=1: optimizer state, LR/eps, and RMS restored from checkpoint")
        else:
            print("RESUME_OPT=0: resumed weights only; optimizer and RMS use fresh state")
        if RESUME_DS:
            ds_ckpt = ckpt.get('delta_scale', None)
            if ds_ckpt is not None:
                resumed_ds = float(ds_ckpt)
                print(f"Resumed delta_scale={resumed_ds:.6f} from checkpoint")
            else:
                print("RESUME_DS=1 but checkpoint has no delta_scale; using schedule/env")
        if RESET_CRITIC:
            for layer in ac.critic[:-1]:
                if isinstance(layer, nn.Linear):
                    _ortho(layer)
            if isinstance(ac.critic[-1], nn.Linear):
                _ortho(ac.critic[-1], gain=1.0)
            ppo.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
            ppo._rms = RunningMeanStd()
            warmup_off = 0
            print("RESET_CRITIC=1: critic, vf_opt, and ret_rms reset; critic warmup re-enabled")
    else:
        pretrain_bc(ac, all_csv)

    ds_max_run = DELTA_SCALE_MAX
    ds_min_run = DELTA_SCALE_MIN
    if RESUME_DS and resumed_ds is not None:
        ds_max_run = resumed_ds
        # Keep decay monotonic downward even if env min > resumed max.
        ds_min_run = min(ds_min_run, ds_max_run)

    baseline_ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(0.0)) if DELTA_SCALE_DECAY else ds_max_run
    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=baseline_ds)
    best, best_ep = vm, 'init'
    print(f"Baseline: {vm:.1f} ± {vs:.1f}  (Δs={baseline_ds:.4f})")

    cur_ds = ds_max_run
    def save_best():
        torch.save({
            'ac': ac.state_dict(),
            'pi_opt': ppo.pi_opt.state_dict(),
            'vf_opt': ppo.vf_opt.state_dict(),
            'ret_rms': {'mean': ppo._rms.mean, 'var': ppo._rms.var, 'count': ppo._rms.count},
            'delta_scale': cur_ds,
        }, BEST_PT)

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
          f"  rew_scale={REWARD_SCALE:g}"
          f"  lr_decay={'on' if LR_DECAY else 'off'}"
          f"  resume_opt={'on' if RESUME_OPT else 'off'}"
          f"  reset_critic={'on' if RESET_CRITIC else 'off'}"
          f"  resume_warmup={'on' if RESUME_WARMUP else 'off'}"
          f"  σfloor_eff={SIGMA_FLOOR} coef={SIGMA_FLOOR_COEF}"
          f"  rew_rms_norm={'on' if REWARD_RMS_NORM else 'off'}"
          f"  adv_norm={'on' if ADV_NORM else 'off'}"
          f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {ds_max_run}→{ds_min_run}  K={K_EPOCHS}  dim={STATE_DIM}\n")

    for epoch in range(MAX_EP):
        if DELTA_SCALE_DECAY:
            ds = ds_min_run + 0.5 * (ds_max_run - ds_min_run) * (1 + np.cos(np.pi * epoch / MAX_EP))
        else:
            ds = ds_max_run
        cur_ds = ds
        if RESUME and RESUME_OPT and epoch == 0:
            # Keep checkpoint optimizer LR for first resumed update.
            pi_lr = ppo.pi_opt.param_groups[0]['lr']
            vf_lr = ppo.vf_opt.param_groups[0]['lr']
        elif LR_DECAY:
            pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
            vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
        else:
            pi_lr = PI_LR
            vf_lr = VF_LR
        for pg in ppo.pi_opt.param_groups: pg['lr'] = pi_lr
        for pg in ppo.vf_opt.param_groups: pg['lr'] = vf_lr
        t0 = time.time()
        batch = random.sample(tr_f, min(CSVS_EPOCH, len(tr_f)))
        res = rollout(batch, ac, mdl_path, ort_sess, csv_cache, deterministic=False, ds=ds)

        t1 = time.time()
        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co, ds=ds)
        tu = time.time() - t1

        phase = "  [critic warmup]" if co else ""
        line = (f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}  σraw={info['σraw']:.4f}"
                f"  σpen={info['σpen']:.4f}  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
                f"  Δs={ds:.4f}  lr={info['lr']:.1e}  ⏱{t1-t0:.0f}+{tu:.0f}s{phase}")

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
    torch.save({'ac': ac.state_dict()}, EXP_DIR / 'final_model.pt')


if __name__ == '__main__':
    train()
