"""
exp049 — Clean PPO from Scratch
================================
beautiful_lander recipe applied to controls challenge.
No BC. Delta actions. ReLU. Higher LRs. Global adv norm.
107 dims: error, ei, pe, v, a_ego, roll, prev_act, κ×51, Δκ×49. γ=0.95, K=6.
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

# ── Config (beautiful_lander recipe) ─────────────────────────

STATE_DIM   = 107       # error, ei, pe, v, a_ego, roll, prev_act, κ_now, κ_future×50, Δκ×49
HIDDEN      = 128
A_LAYERS    = 5
C_LAYERS    = 3
FUTURE_K    = 50

DELTA_SCALE = 0.1       # raw × 0.1 = physical delta
MAX_DELTA   = 0.3       # safety clip

PI_LR       = 3e-4
VF_LR       = 3e-4
GAMMA       = 0.95      # tracking problem: short credit horizon
LAMDA       = 0.95
K_EPOCHS    = 6         # 600k transitions → ~720 grad steps (not 2400)
EPS_CLIP    = 0.2
VF_COEF     = 0.5
ENT_COEF    = 0.001
MINI_BS     = 5000

CSVS_EPOCH  = int(os.getenv('CSVS',    '300'))
MAX_EP      = int(os.getenv('EPOCHS',   '200'))
EVAL_EVERY  = 5
EVAL_N      = 100
WORKERS     = int(os.getenv('WORKERS',  '8'))
N_TRAIN     = 18_000
N_VAL       = 1_000
N_TEST      = 1_000

WARMUP_N    = CONTROL_START_IDX - CONTEXT_LENGTH   # 80
SCORED_N    = COST_END_IDX - CONTROL_START_IDX     # 400

# [error/3, ei/10, pe/3, v/30, a_ego/5, roll/3, prev_act/2, κ×100 (×51), Δκ×500 (×49)]
OBS_SCALE   = np.array([1/3, 1/10, 1/3, 1/30, 1/5, 1/3, 1/2] + [100.0]*51 + [500.0]*49, dtype=np.float32)
OBS_SCALE_T = torch.tensor(OBS_SCALE)

EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
BEST_PT = EXP_DIR / 'best_model.pt'


# ── Observation (54 dims) ────────────────────────────────────

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

def build_obs(target, current, state, fplan, prev_act, error_integral, prev_error):
    obs = np.empty(STATE_DIM, np.float32)
    # Feedback (7 dims)
    obs[0]  = target - current
    obs[1]  = error_integral
    obs[2]  = prev_error
    obs[3]  = state.v_ego
    obs[4]  = state.a_ego
    obs[5]  = state.roll_lataccel
    obs[6]  = prev_act
    # Feedforward: κ now + future (51 dims)
    obs[7]  = _kappa(target, state.roll_lataccel, state.v_ego)
    fk = _future_kappa(fplan, target, state)
    obs[8:58] = fk
    # Feedforward: Δκ (49 dims) — rate of curvature change
    obs[58:107] = np.diff(fk)
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
        self.log_std = nn.Parameter(torch.full((1,), -1.2))  # σ₀≈0.3, learned

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        # Zero-init actor last layer → initial Δ ≈ 0 → smooth start
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    @torch.inference_mode()
    def act(self, obs_np, deterministic=False):
        s = torch.from_numpy(obs_np).unsqueeze(0) * OBS_SCALE_T
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
        self.traj = []

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        e = target_lataccel - current_lataccel
        self._ei += e

        if self.n <= WARMUP_N:
            de = e - self._pe; self._pe = e
            self.prev_act = float(np.clip(
                0.195*e + 0.1*self._ei - 0.053*de, *STEER_RANGE))
            return 0.0

        obs = build_obs(target_lataccel, current_lataccel, state,
                        future_plan, self.prev_act, self._ei, self._pe)
        self._pe = e
        raw, val = self.ac.act(obs, self.det)

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self.prev_act + delta, *STEER_RANGE))
        self.prev_act = action

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


# ── PPO (global advantage norm, beautiful_lander recipe) ─────

class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(
            list(ac.actor.parameters()) + [ac.log_std], lr=PI_LR)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR)

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
        # Global normalization (not per-trajectory)
        a = np.concatenate(advs); r = np.concatenate(rets)
        return (a - a.mean()) / (a.std() + 1e-8), r

    def update(self, all_obs, all_raw, all_rew, all_val, all_done):
        obs_t = torch.FloatTensor(np.concatenate(all_obs)) * OBS_SCALE_T
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

        return dict(pi=pi_loss.item(), vf=vf_loss.item(),
                    ent=ent.item(), σ=self.ac.log_std.exp().item())


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


# ── Training (no BC, pure PPO from scratch) ──────────────────

class Ctx:
    def __init__(self):
        self.ac  = ActorCritic()
        self.ppo = PPO(self.ac)
        self.mdl = ROOT / 'models' / 'tinyphysics.onnx'
        all_f = sorted((ROOT / 'data').glob('*.csv'))
        # Official eval uses first 100 sorted files — use same for val
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
    # ── Collect ──
    t0 = time.time()
    torch.save(ctx.ac.state_dict(), TMP)
    batch = random.sample(ctx.tr_f, min(CSVS_EPOCH, len(ctx.tr_f)))
    res = process_map(_train_worker,
                      [(str(f), str(ctx.mdl), str(TMP)) for f in batch],
                      max_workers=WORKERS, chunksize=10, disable=True)
    tc = time.time() - t0

    # ── Update ──
    t1 = time.time()
    info = ctx.ppo.update(
        [r[0] for r in res], [r[1] for r in res],
        [r[2] for r in res], [r[3] for r in res],
        [r[4] for r in res])
    tu = time.time() - t1

    # ── Log ──
    costs = [r[5] for r in res]
    line = (f"E{epoch:3d}  train={np.mean(costs):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  ⏱{tc:.0f}+{tu:.0f}s")

    # ── Eval ──
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

    # ── Resume from best checkpoint if it exists ──
    if BEST_PT.exists():
        print(f"Resuming from {BEST_PT}")
        ctx.ac.load_state_dict(
            torch.load(BEST_PT, weights_only=False, map_location='cpu'))
        ctx.ppo = PPO(ctx.ac)  # fresh optimizer with new LR
        vm, vs = evaluate(ctx.ac, ctx.mdl, ctx.va_f)
        ctx.best, ctx.best_ep = vm, 'resume'
        print(f"Resumed baseline: {vm:.1f} ± {vs:.1f}")
    else:
        vm, vs = evaluate(ctx.ac, ctx.mdl, ctx.va_f)
        ctx.best, ctx.best_ep = vm, 'init'
        print(f"Random init baseline: {vm:.1f} ± {vs:.1f}")
        torch.save(ctx.ac.state_dict(), BEST_PT)

    print(f"\nPPO from scratch  csvs={CSVS_EPOCH}  epochs={MAX_EP}  workers={WORKERS}")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}"
          f"  σ₀={ctx.ac.log_std.exp().item():.3f}"
          f"  layers={A_LAYERS}+{C_LAYERS}  K={K_EPOCHS}\n")

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best val: {ctx.best:.1f} (epoch {ctx.best_ep})")
    torch.save(ctx.ac.state_dict(), EXP_DIR / 'final_model.pt')
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
