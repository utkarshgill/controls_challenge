"""
exp050 — Rich-Obs PPO from Scratch
====================================
207-dim obs: error, ei, error_diff, v, a_ego, roll, prev_act,
            κ×51, Δκ×49, future_v×50, future_a×50.
Delta actions. ReLU. Huber VF loss. Val = official first 100 sorted files.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, os, random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, BaseController,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
)
from tqdm.contrib.concurrent import process_map

# ── Config ────────────────────────────────────────────────────

HIST_LEN    = 20         # last 20 actions + last 20 errors = 40 dims
STATE_DIM   = 248       # 7 + 51κ + 50Δκ + 50 future_a_ego + 50 future_err + 20 act_hist + 20 err_hist
HIDDEN      = 256
A_LAYERS    = 4
C_LAYERS    = 4
FUTURE_K    = 50

DELTA_SCALE = 0.3
MAX_DELTA   = 0.3

PI_LR       = 3e-4
VF_LR       = 1e-4
GAMMA       = 0.90
LAMDA       = 0.95
K_EPOCHS    = 5
EPS_CLIP    = 0.2
VF_COEF     = 1.0
ENT_COEF    = 0.001
MINI_BS     = 5000

CSVS_EPOCH  = int(os.getenv('CSVS',    '100'))
MAX_EP      = int(os.getenv('EPOCHS',   '200'))
EVAL_EVERY  = 5
EVAL_N      = 100
WORKERS     = int(os.getenv('WORKERS',  '8'))

WARMUP_N    = CONTROL_START_IDX - CONTEXT_LENGTH   # 80
SCORED_N    = COST_END_IDX - CONTROL_START_IDX     # 400

# Scaling (data-driven from 2k file analysis, p95-based):
# [error/0.75, ei/114, ediff/0.04, v/34, a_ego/0.65, roll/0.05, prev_act/2,
#  κ×667 (×51), Δκ×10000 (×49), future_a/0.65 (×50)]
OBS_SCALE   = np.array(
    [1/0.75, 1/114, 1/0.04, 1/34, 1/0.65, 1/0.05, 1/2]
    + [667.0]*51 + [10000.0]*50
    + [1/0.65]*50
    + [1/2.0]*50
    + [1/2]*HIST_LEN + [1/0.75]*HIST_LEN,
    dtype=np.float32)
OBS_SCALE_T = torch.tensor(OBS_SCALE)

EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
BEST_PT = EXP_DIR / 'best_model.pt'


# ── Observation (207 dims) ───────────────────────────────────

def _kappa(lat, roll, v):
    return np.clip((lat - roll) / max(v * v, 25.0), -1.0, 1.0)

def _future_kappa(fplan, target, state):
    n = len(fplan.lataccel) if fplan else 0
    if n >= FUTURE_K:
        fl = np.asarray(fplan.lataccel[:FUTURE_K], np.float32)
        fr = np.asarray(fplan.roll_lataccel[:FUTURE_K], np.float32)
        fv = np.asarray(fplan.v_ego[:FUTURE_K], np.float32)
    elif n > 0:
        fl = np.array(fplan.lataccel, np.float32)
        fr = np.array(fplan.roll_lataccel, np.float32)
        fv = np.array(fplan.v_ego, np.float32)
        p = FUTURE_K - n
        fl, fr, fv = [np.pad(x, (0, p), 'edge') for x in (fl, fr, fv)]
    else:
        return np.full(FUTURE_K, _kappa(target, state.roll_lataccel, state.v_ego),
                       dtype=np.float32)
    return np.clip((fl - fr) / np.maximum(fv**2, 25.0), -1.0, 1.0)

def _future_profile(fplan, attr, fallback, k=FUTURE_K):
    """Extract a future profile, padding with edge if short."""
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)

def build_obs(target, current, state, fplan, prev_act, error_integral, prev_error,
              act_hist, err_hist):
    obs = np.empty(STATE_DIM, np.float32)
    e = target - current
    # Feedback (7 dims)
    obs[0]  = e                         # error
    obs[1]  = error_integral            # integral
    obs[2]  = e - prev_error            # error_diff (D-term)
    obs[3]  = state.v_ego
    obs[4]  = state.a_ego
    obs[5]  = state.roll_lataccel
    obs[6]  = prev_act
    # Feedforward: κ now + future (51 dims)
    obs[7]  = _kappa(target, state.roll_lataccel, state.v_ego)
    fk = _future_kappa(fplan, target, state)
    obs[8:58] = fk
    # Feedforward: Δκ (50 dims) — prepend κ_now so first diff = fk[0] - κ_now
    obs[58:108] = np.diff(np.concatenate([[obs[7]], fk]))
    # Future a_ego profile (50 dims) — friction circle / braking anticipation
    obs[108:158] = _future_profile(fplan, 'a_ego', state.a_ego)
    # Future error profile (50 dims) — what gap to close at each future step
    obs[158:208] = _future_profile(fplan, 'lataccel', target) - current
    # History buffer (40 dims) — recent actions & errors for ONNX context
    obs[208:208+HIST_LEN] = act_hist
    obs[208+HIST_LEN:208+2*HIST_LEN] = err_hist
    # Clip scaled obs to prevent outlier blowup
    np.clip(obs * OBS_SCALE, -5.0, 5.0, out=obs)
    return obs


# ── Network (ReLU, zero-init last layer) ─────────────────────

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 1))
        self.actor = nn.Sequential(*a)
        self.log_std = nn.Parameter(torch.full((1,), -1.2))  # σ₀≈0.3

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    @torch.inference_mode()
    def act(self, obs_np, deterministic=False):
        s = torch.from_numpy(obs_np).unsqueeze(0)  # already scaled+clipped in build_obs
        mu  = self.actor(s).item()
        val = self.critic(s).item()
        if deterministic:
            return mu, val
        std = self.log_std.exp().item()
        return mu + std * np.random.randn(), val


# ── Controller ───────────────────────────────────────────────

class DeltaController(BaseController):
    def __init__(self, ac, deterministic=False):
        self.ac, self.det = ac, deterministic
        self.n, self.prev_act = 0, 0.0
        self._ei, self._pe = 0.0, 0.0
        self._act_hist = [0.0] * HIST_LEN
        self._err_hist = [0.0] * HIST_LEN
        self.traj = []

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        e = target_lataccel - current_lataccel
        self._ei += e

        if self.n <= WARMUP_N:
            self._pe = e
            self._err_hist.append(e); self._err_hist.pop(0)
            # act_hist stays 0.0 during warmup (prev_act=0.0)
            return 0.0  # sim overrides with CSV steer; prev_act stays 0.0

        obs = build_obs(target_lataccel, current_lataccel, state,
                        future_plan, self.prev_act, self._ei, self._pe,
                        self._act_hist, self._err_hist)
        self._pe = e
        raw, val = self.ac.act(obs, self.det)

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self.prev_act + delta, *STEER_RANGE))
        self.prev_act = action

        # Update history buffers
        self._act_hist.append(action); self._act_hist.pop(0)
        self._err_hist.append(e); self._err_hist.pop(0)

        if self.n <= WARMUP_N + SCORED_N:
            self.traj.append(dict(obs=obs, raw=raw, val=val,
                                  tgt=target_lataccel, cur=current_lataccel))
        return action


# ── Rewards ──────────────────────────────────────────────────

def compute_rewards(traj):
    tgt = np.array([t['tgt'] for t in traj], np.float32)
    cur = np.array([t['cur'] for t in traj], np.float32)
    lat  = (tgt - cur)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
    jerk = np.diff(cur, prepend=cur[0]) / DEL_T
    return (-(lat + jerk**2 * 100) / 500.0).astype(np.float32)


# ── PPO ──────────────────────────────────────────────────────

class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(
            list(ac.actor.parameters()) + [ac.log_std], lr=PI_LR)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR)
        self.pi_sched = optim.lr_scheduler.CosineAnnealingLR(self.pi_opt, T_max=MAX_EP)
        self.vf_sched = optim.lr_scheduler.CosineAnnealingLR(self.vf_opt, T_max=MAX_EP)

    @staticmethod
    def gae(all_r, all_v, all_d):
        advs, rets = [], []
        for r, v, d in zip(all_r, all_v, all_d):
            T = len(r); adv = np.zeros(T, np.float32); g = 0.0
            for t in range(T-1, -1, -1):
                nv = 0.0 if t == T-1 else v[t+1]
                g = (r[t] + GAMMA*nv*(1-d[t]) - v[t]) + GAMMA*LAMDA*(1-d[t])*g
                adv[t] = g
            advs.append(adv); rets.append(adv + v)
        a = np.concatenate(advs); r = np.concatenate(rets)
        return (a - a.mean()) / (a.std() + 1e-8), r

    def update(self, all_obs, all_raw, all_rew, all_val, all_done):
        obs_t = torch.FloatTensor(np.concatenate(all_obs))  # already scaled+clipped
        raw_t = torch.FloatTensor(np.concatenate(all_raw)).unsqueeze(-1)
        adv, ret = self.gae(all_rew, all_val, all_done)
        adv_t, ret_t = torch.FloatTensor(adv), torch.FloatTensor(ret)

        std = self.ac.log_std.exp()
        with torch.no_grad():
            mu = self.ac.actor(obs_t)
            old_lp = torch.distributions.Normal(mu, std).log_prob(raw_t).sum(-1)

        N = len(obs_t)
        for _ in range(K_EPOCHS):
            for idx in torch.randperm(N).split(MINI_BS):
                mu  = self.ac.actor(obs_t[idx])
                val = self.ac.critic(obs_t[idx])
                std = self.ac.log_std.exp()
                dist = torch.distributions.Normal(mu, std)
                lp  = dist.log_prob(raw_t[idx]).sum(-1)

                ratio = (lp - old_lp[idx]).exp()
                pi_loss = -torch.min(
                    ratio * adv_t[idx],
                    ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_t[idx]).mean()
                vf_loss = F.huber_loss(val.squeeze(-1), ret_t[idx], delta=10.0)
                ent = dist.entropy().sum(-1).mean()

                loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent
                self.pi_opt.zero_grad(); self.vf_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.ac.actor.parameters()) + [self.ac.log_std], 0.5)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                self.pi_opt.step(); self.vf_opt.step()

        self.pi_sched.step(); self.vf_sched.step()
        return dict(pi=pi_loss.item(), vf=vf_loss.item(),
                    ent=ent.item(), σ=self.ac.log_std.exp().item(),
                    lr=self.pi_sched.get_last_lr()[0])


# ── Workers ──────────────────────────────────────────────────

def _train_worker(args):
    torch.set_num_threads(1)
    csv, mdl, ckpt = args
    ac = ActorCritic()
    ac.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu'))
    ac.eval()
    ctrl = DeltaController(ac, deterministic=False)
    sim = TinyPhysicsSimulator(
        TinyPhysicsModel(mdl, debug=False), str(csv), controller=ctrl, debug=False)
    cost = sim.rollout()
    T = len(ctrl.traj)
    return (np.array([t['obs'] for t in ctrl.traj], np.float32),
            np.array([t['raw'] for t in ctrl.traj], np.float32),
            compute_rewards(ctrl.traj),
            np.array([t['val'] for t in ctrl.traj], np.float32),
            np.concatenate([np.zeros(T-1, np.float32), [1.0]]),
            cost['total_cost'])

def _eval_worker(args):
    torch.set_num_threads(1)
    csv, mdl, ckpt = args
    ac = ActorCritic()
    ac.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu'))
    ac.eval()
    ctrl = DeltaController(ac, deterministic=True)
    sim = TinyPhysicsSimulator(
        TinyPhysicsModel(mdl, debug=False), str(csv), controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


# ── Training ─────────────────────────────────────────────────

class Ctx:
    def __init__(self):
        self.ac  = ActorCritic()
        self.ppo = PPO(self.ac)
        self.mdl = ROOT / 'models' / 'tinyphysics.onnx'
        all_f = sorted((ROOT / 'data').glob('*.csv'))
        # Official eval uses first 100 sorted files — match exactly
        self.va_f = all_f[:EVAL_N]
        rest = all_f[EVAL_N:]
        random.seed(42)
        random.shuffle(rest)
        self.tr_f = rest
        self.best = float('inf')
        self.best_ep = -1


def evaluate(ac, mdl, files, n=EVAL_N):
    torch.save(ac.state_dict(), TMP)
    args = [(str(f), str(mdl), str(TMP)) for f in files[:n]]
    costs = process_map(_eval_worker, args, max_workers=WORKERS,
                        chunksize=5, disable=True)
    return float(np.mean(costs)), float(np.std(costs))


def train_one_epoch(epoch, ctx):
    t0 = time.time()
    torch.save(ctx.ac.state_dict(), TMP)
    batch = random.sample(ctx.tr_f, min(CSVS_EPOCH, len(ctx.tr_f)))
    res = process_map(_train_worker,
                      [(str(f), str(ctx.mdl), str(TMP)) for f in batch],
                      max_workers=WORKERS, chunksize=10, disable=True)
    tc = time.time() - t0

    t1 = time.time()
    info = ctx.ppo.update(
        [r[0] for r in res], [r[1] for r in res],
        [r[2] for r in res], [r[3] for r in res],
        [r[4] for r in res])
    tu = time.time() - t1

    costs = [r[5] for r in res]
    line = (f"E{epoch:3d}  train={np.mean(costs):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  lr={info['lr']:.1e}  ⏱{tc:.0f}+{tu:.0f}s")

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(ctx.ac, ctx.mdl, ctx.va_f)
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            torch.save(ctx.ac.state_dict(), BEST_PT)
            mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.0f}{mk}"

    print(line)


def train():
    ctx = Ctx()

    vm, vs = evaluate(ctx.ac, ctx.mdl, ctx.va_f)
    ctx.best, ctx.best_ep = vm, 'init'
    print(f"Random init baseline: {vm:.1f} ± {vs:.1f}")
    torch.save(ctx.ac.state_dict(), BEST_PT)

    print(f"\nPPO from scratch  csvs={CSVS_EPOCH}  epochs={MAX_EP}  workers={WORKERS}")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}"
          f"  σ₀={ctx.ac.log_std.exp().item():.3f}"
          f"  layers={A_LAYERS}+{C_LAYERS}  K={K_EPOCHS}  dim={STATE_DIM}\n")

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best val: {ctx.best:.1f} (epoch {ctx.best_ep})")
    torch.save(ctx.ac.state_dict(), EXP_DIR / 'final_model.pt')
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
