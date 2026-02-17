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
from tinyphysics_batched import BatchedSimulator, BatchedPhysicsModel, CSVCache, make_ort_session

torch.manual_seed(42); np.random.seed(42)
torch.set_float32_matmul_precision('high')
DEV = torch.device('cuda')

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN   = 256, 256
A_LAYERS, C_LAYERS  = 4, 4
DELTA_SCALE         = 0.25
MAX_DELTA           = 0.5

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── PPO ───────────────────────────────────────────────────────
PI_LR      = float(os.getenv('PI_LR', '3e-4'))
VF_LR      = float(os.getenv('VF_LR', '3e-4'))
GAMMA       = 0.95
LAMDA       = 0.9
K_EPOCHS    = 4
EPS_CLIP    = 0.2
VF_COEF     = 1.0
ENT_COEF    = float(os.getenv('ENT_COEF', '0.001'))
ACT_SMOOTH  = float(os.getenv('ACT_SMOOTH', '10.0'))
MINI_BS     = int(os.getenv('MINI_BS', '100000'))
CRITIC_WARMUP = 4

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS   = int(os.getenv('BC_EPOCHS', '20'))
BC_LR       = float(os.getenv('BC_LR', '0.01'))
BC_BS       = int(os.getenv('BC_BS', '8192'))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv('CSVS', '500'))
MAX_EP     = int(os.getenv('EPOCHS', '200'))
EVAL_EVERY = 5
EVAL_N     = 100
RESUME     = os.getenv('RESUME', '0') == '1'
DEBUG      = int(os.getenv('DEBUG', '0'))

EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
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

def rollout(csv_files, ac, mdl_path, ort_session, csv_cache,
            deterministic=False, sim_model=None):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session,
                           cached_data=data, cached_rng=rng,
                           sim_model=sim_model)
    N, T = sim.N, sim.T
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX

    # Pre-cast to float32 once — eliminates per-step .float() casts
    dg_f = {}
    for k in ('target_lataccel', 'roll_lataccel', 'v_ego', 'a_ego'):
        dg_f[k] = dg[k].float()

    h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device='cuda')
    zeros_N = torch.zeros(N, dtype=torch.float64, device='cuda')

    if not deterministic:
        # (N, S, ...) layout so obs_flat = view, no 700MB copy
        all_obs = torch.empty((N, max_steps, OBS_DIM), dtype=torch.float32, device='cuda')
        all_raw = torch.empty((N, max_steps), dtype=torch.float32, device='cuda')
        tgt_hist = torch.empty((N, max_steps), dtype=torch.float64, device='cuda')
        cur_hist = torch.empty((N, max_steps), dtype=torch.float64, device='cuda')
        act_hist = torch.empty((N, max_steps), dtype=torch.float64, device='cuda')
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si
        target  = dg['target_lataccel'][:, step_idx]
        current = sim_ref.current_lataccel

        cur32 = current.float()
        target_f = dg_f['target_lataccel'][:, step_idx]
        error = target_f - cur32
        h_error[:, :-1] = h_error[:, 1:]; h_error[:, -1] = error
        ei = h_error.mean(dim=1) * DEL_T

        if step_idx < CONTROL_START_IDX:
            h_act[:, :-1] = h_act[:, 1:]; h_act[:, -1] = 0.0
            h_act32[:, :-1] = h_act32[:, 1:]; h_act32[:, -1] = 0.0
            h_lat[:, :-1] = h_lat[:, 1:]; h_lat[:, -1] = cur32
            return zeros_N

        fill_obs(obs_buf, target_f, cur32,
                 dg_f['roll_lataccel'][:, step_idx],
                 dg_f['v_ego'][:, step_idx],
                 dg_f['a_ego'][:, step_idx],
                 h_act32, h_lat, ei, dg_f, step_idx, T)

        a_p, b_p = ac.beta_params(obs_buf)

        raw = 2.0 * a_p / (a_p + b_p) - 1.0 if deterministic \
              else 2.0 * torch.distributions.Beta(a_p, b_p).sample() - 1.0

        delta  = (raw.double() * DELTA_SCALE).clamp(-MAX_DELTA, MAX_DELTA)
        action = (h_act[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, :-1] = h_act[:, 1:]; h_act[:, -1] = action
        h_act32[:, :-1] = h_act32[:, 1:]; h_act32[:, -1] = action.float()
        h_lat[:, :-1] = h_lat[:, 1:]; h_lat[:, -1] = cur32

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[:, si] = obs_buf; all_raw[:, si] = raw
            tgt_hist[:, si] = target; cur_hist[:, si] = current; act_hist[:, si] = action
            si += 1
        return action

    with torch.inference_mode():
        costs = sim.rollout(ctrl)['total_cost']

    if deterministic:
        return costs.tolist()

    S = si
    obs_flat = all_obs.reshape(-1, OBS_DIM)
    with torch.inference_mode():
        val_2d = ac.critic(obs_flat).squeeze(-1).reshape(N, S)

    lat_r = (tgt_hist - cur_hist)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(cur_hist, dim=1, prepend=cur_hist[:, :1]) / DEL_T
    act_d = torch.diff(act_hist, dim=1, prepend=act_hist[:, :1]) / DEL_T
    rew = (-(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / 500.0).float()
    dones = torch.zeros((N, S), dtype=torch.float32, device='cuda')
    dones[:, -1] = 1.0

    return dict(
        obs=obs_flat,
        raw=all_raw.reshape(-1),
        val_2d=val_2d,
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
        raw_target = np.clip((steer[step_idx] - h_act[-1]) / DELTA_SCALE, -1.0, 1.0)
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

    def _gae(self, rew, val, done):
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

    def update(self, gd, critic_only=False):
        obs = gd['obs']
        raw = gd['raw'].unsqueeze(-1)
        adv_t, ret_t = self._gae(gd['rew'], gd['val_2d'], gd['done'])
        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)

        with torch.no_grad():
            a_old, b_old = self.ac.beta_params(obs)
            old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t.squeeze(-1))
        old_val = gd['val_2d'].reshape(-1)

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device='cuda').split(MINI_BS):
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                val = self.ac.critic(obs[idx]).squeeze(-1)
                vc = old_val[idx] + (val - old_val[idx]).clamp(-10, 10)
                vf_loss = torch.max(
                    F.huber_loss(val, ret_t[idx], delta=10.0, reduction='none'),
                    F.huber_loss(vc,  ret_t[idx], delta=10.0, reduction='none')).mean()

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
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
                    loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                    self.pi_opt.step(); self.vf_opt.step()

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs[:1000])
            sigma = (2.0 * torch.sqrt(a_d*b_d / ((a_d+b_d)**2 * (a_d+b_d+1)))).mean().item()
        return dict(
            pi=pi_loss.item() if not critic_only else 0.0,
            vf=vf_loss.item(),
            ent=ent.item() if not critic_only else 0.0,
            σ=sigma, lr=self.pi_opt.param_groups[0]['lr'])


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════

class TrainingContext:
    def __init__(self, ac, ppo, mdl_path, ort_sess, tr_f, va_f, csv_cache, warmup_off, sim_model):
        self.ac, self.ppo = ac, ppo
        self.mdl_path, self.ort_sess = mdl_path, ort_sess
        self.tr_f, self.va_f = tr_f, va_f
        self.csv_cache = csv_cache
        self.warmup_off = warmup_off
        self.sim_model = sim_model
        self.best, self.best_ep = float('inf'), 'init'

    def save_best(self):
        ac_sd = {k.replace('._orig_mod', ''): v for k, v in self.ac.state_dict().items()}
        torch.save({
            'ac': ac_sd,
            'pi_opt': self.ppo.pi_opt.state_dict(),
            'vf_opt': self.ppo.vf_opt.state_dict(),
            'ret_rms': {'mean': self.ppo._rms.mean, 'var': self.ppo._rms.var,
                        'count': self.ppo._rms.count},
        }, BEST_PT)


def evaluate(ac, files, mdl_path, ort_session, csv_cache, sim_model=None):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache,
                    deterministic=True, sim_model=sim_model)
    return float(np.mean(costs)), float(np.std(costs))


def train_one_epoch(epoch, ctx):
    t0 = time.time()
    batch = random.sample(ctx.tr_f, min(CSVS_EPOCH, len(ctx.tr_f)))
    res = rollout(batch, ctx.ac, ctx.mdl_path, ctx.ort_sess, ctx.csv_cache,
                  sim_model=ctx.sim_model)
    t1 = time.time()

    co = epoch < (CRITIC_WARMUP - ctx.warmup_off)
    info = ctx.ppo.update(res, critic_only=co)
    tu = time.time() - t1

    phase = "  [critic warmup]" if co else ""
    line = (f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  lr={info['lr']:.1e}  ⏱{t1-t0:.0f}+{tu:.0f}s{phase}")

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(ctx.ac, ctx.va_f, ctx.mdl_path, ctx.ort_sess,
                          ctx.csv_cache, sim_model=ctx.sim_model)
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            ctx.save_best()
            mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"
    print(line)


def train():
    ac = ActorCritic().to(DEV)
    mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
    ort_sess = make_ort_session(mdl_path)

    all_csv = sorted((ROOT / 'data').glob('*.csv'))
    va_f = all_csv[:EVAL_N]
    tr_f = all_csv[EVAL_N:]
    random.seed(42); random.shuffle(tr_f)
    csv_cache = CSVCache(sorted(set(str(f) for f in tr_f + va_f)))

    warmup_off = 0
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt['ac'])
        if 'pi_opt' in ckpt:
            warmup_off = CRITIC_WARMUP
        print(f"Resumed from {BEST_PT.name}")
    else:
        all_csvs = sorted((ROOT / 'data').glob('*.csv'))
        pretrain_bc(ac, all_csvs)

    ac.actor = torch.compile(ac.actor)
    ac.critic = torch.compile(ac.critic)
    ppo = PPO(ac)
    if RESUME and BEST_PT.exists() and 'pi_opt' in ckpt:
        ppo.pi_opt.load_state_dict(ckpt['pi_opt'])
        ppo.vf_opt.load_state_dict(ckpt['vf_opt'])
        for pg in ppo.pi_opt.param_groups: pg['lr'] = PI_LR; pg['eps'] = 1e-5
        for pg in ppo.vf_opt.param_groups: pg['lr'] = VF_LR; pg['eps'] = 1e-5
        if 'ret_rms' in ckpt:
            r = ckpt['ret_rms']
            ppo._rms.mean, ppo._rms.var, ppo._rms.count = r['mean'], r['var'], r['count']

    sim_model = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)

    ctx = TrainingContext(ac, ppo, mdl_path, ort_sess, tr_f, va_f, csv_cache, warmup_off, sim_model)

    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, sim_model=sim_model)
    ctx.best, ctx.best_ep = vm, 'init'
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")
    ctx.save_best()

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
          f"  Δscale={DELTA_SCALE}  K={K_EPOCHS}  dim={STATE_DIM}\n")

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best: {ctx.best:.1f} (epoch {ctx.best_ep})")
    ac_sd = {k.replace('._orig_mod', ''): v for k, v in ac.state_dict().items()}
    torch.save({'ac': ac_sd}, EXP_DIR / 'final_model.pt')
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
