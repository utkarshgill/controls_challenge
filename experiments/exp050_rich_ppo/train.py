# exp050 — 256-dim Beta PPO with delta actions, NLL BC pretrain, Huber VF loss

import numpy as np, pandas as pd, sys, os, time, random, json, subprocess, tempfile, multiprocessing
import io, pickle, socket, struct, threading
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, BaseController,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, FUTURE_PLAN_STEPS,
    STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, ACC_G, State, FuturePlan)
from tinyphysics_batched import BatchedSimulator, CSVCache, pool_init, get_pool_cache, run_parallel_chunked

torch.manual_seed(42)
np.random.seed(42)

# architecture
HIST_LEN, FUTURE_K   = 20, 50
STATE_DIM, HIDDEN     = 256, 256        # 16 core + 40 hist + 200 future
A_LAYERS, C_LAYERS    = 4, 4
DELTA_SCALE, MAX_DELTA = 0.25, 0.5

# PPO
PI_LR, VF_LR     = 3e-4, 3e-4
GAMMA, LAMDA      = 0.95, 0.9
K_EPOCHS, EPS_CLIP = 4, 0.2
VF_COEF, ENT_COEF = 1.0, 0.003
MINI_BS           = int(os.getenv('MINI_BS', '100000'))
CRITIC_WARMUP     = 4

# BC
BC_EPOCHS    = int(os.getenv('BC_EPOCHS', '20'))
BC_LR        = float(os.getenv('BC_LR', '0.01'))
BC_BS        = int(os.getenv('BC_BS', '8192'))
BC_GRAD_CLIP = float(os.getenv('BC_GRAD_CLIP', '2.0'))

# runtime
CSVS_EPOCH = int(os.getenv('CSVS',    '500'))
MAX_EP     = int(os.getenv('EPOCHS',   '200'))
EVAL_EVERY, EVAL_N = 5, 100
WORKERS    = int(os.getenv('WORKERS',  '10'))
BC_WORKERS = int(os.getenv('BC_WORKERS', '10'))
RESUME     = os.getenv('RESUME', '0') == '1'
DECAY_LR   = os.getenv('DECAY_LR', '0') == '1'
BATCHED    = os.getenv('BATCHED', '1') == '1'
USE_CUDA   = os.getenv('CUDA', '0') == '1'
DEV        = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# observation scaling
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
BEST_PT = EXP_DIR / 'best_model.pt'

# Multi-remote config
# REMOTE_HOSTS=ip1,ip2  REMOTE_PORTS=5555,5555  FRAC=local:r1:r2 (e.g. 3.5:3.5:3)
_remote_hosts_str = os.getenv('REMOTE_HOSTS', '')
_remote_ports_str = os.getenv('REMOTE_PORTS', '')
REMOTE_DIR     = os.getenv('REMOTE_DIR', '~/Desktop/stuff/controls_challenge')
REMOTE_PY      = os.getenv('REMOTE_PY', '~/Desktop/stuff/controls_challenge/.venv/bin/python')
SSH_KEY        = os.path.expanduser('~/.ssh/id_ed25519')

REMOTE_HOSTS = [h.strip() for h in _remote_hosts_str.split(',') if h.strip()] if _remote_hosts_str else []
REMOTE_PORTS = [int(p.strip()) for p in _remote_ports_str.split(',') if p.strip()] if _remote_ports_str else []
if REMOTE_HOSTS and len(REMOTE_PORTS) < len(REMOTE_HOSTS):
    REMOTE_PORTS += [5555] * (len(REMOTE_HOSTS) - len(REMOTE_PORTS))

USE_REMOTE = len(REMOTE_HOSTS) > 0

# FRAC = "local_weight:remote1_weight:remote2_weight:..."  (default: equal split)
_frac_str = os.getenv('FRAC', '')
if _frac_str:
    _frac_parts = [float(x) for x in _frac_str.split(':')]
else:
    _frac_parts = [1.0] + [1.0] * len(REMOTE_HOSTS)
_frac_total = sum(_frac_parts)
FRAC_LOCAL = _frac_parts[0] / _frac_total
FRAC_REMOTES = [_frac_parts[i+1] / _frac_total for i in range(len(REMOTE_HOSTS))]


def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        a = np.array(vals, np.float32)
        return np.pad(a, (0, k - len(a)), 'edge')
    return np.full(k, fallback, dtype=np.float32)

def _curv(lat, roll, v): return (lat - roll) / max(v * v, 1.0)
def build_obs(target, current, state, fplan,
              hist_act, hist_lat, hist_v, hist_a, hist_roll,
              error_integral=0.0):
    error = target - current
    k_tgt = _curv(target, state.roll_lataccel, state.v_ego)
    k_cur = _curv(current, state.roll_lataccel, state.v_ego)
    _flat = getattr(fplan, 'lataccel', None)
    fplan_lat0 = _flat[0] if (_flat and len(_flat) > 0) else target
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0

    core = np.array([
        target / S_LAT,
        current / S_LAT,
        error / S_LAT,
        k_tgt / S_CURV,
        k_cur / S_CURV,
        (k_tgt - k_cur) / S_CURV,
        state.v_ego / S_VEGO,
        state.a_ego / S_AEGO,
        state.roll_lataccel / S_ROLL,
        hist_act[-1] / S_STEER,
        error_integral / S_LAT,
        (fplan_lat0 - target) / DEL_T / S_LAT,
        (current - hist_lat[-1]) / DEL_T / S_LAT,
        (hist_act[-1] - hist_act[-2]) / DEL_T / S_STEER,
        fric,
        max(0.0, 1.0 - fric),
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

def _ortho_init(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)

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

        for layer in self.actor[:-1]:
            _ortho_init(layer)
        _ortho_init(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]:
            _ortho_init(layer)
        _ortho_init(self.critic[-1], gain=1.0)

    def beta_params(self, obs_t):
        out = self.actor(obs_t)
        alpha = F.softplus(out[..., 0]) + 1.0
        beta  = F.softplus(out[..., 1]) + 1.0
        return alpha, beta

    @torch.inference_mode()
    def act(self, obs_np, deterministic=False):
        s   = torch.from_numpy(obs_np).unsqueeze(0)
        a, b = self.beta_params(s)
        val  = self.critic(s).item()
        if deterministic:
            raw = (2.0 * a / (a + b) - 1.0).item()       # Beta mean → [-1,1]
        else:
            x   = torch.distributions.Beta(a, b).sample()
            raw = (2.0 * x - 1.0).item()
        return raw, val


class DeltaController(BaseController):
    def __init__(self, ac, deterministic=False):
        self.ac, self.det = ac, deterministic
        self.n = 0
        self._h_act   = [0.0] * HIST_LEN
        self._h_lat   = [0.0] * HIST_LEN
        self._h_v     = [0.0] * HIST_LEN
        self._h_a     = [0.0] * HIST_LEN
        self._h_roll  = [0.0] * HIST_LEN
        self._h_error = [0.0] * HIST_LEN
        self.traj = []

    def _push(self, action, current, state):
        self._h_act = self._h_act[1:] + [action]
        self._h_lat = self._h_lat[1:] + [current]
        self._h_v = self._h_v[1:] + [state.v_ego]
        self._h_a = self._h_a[1:] + [state.a_ego]
        self._h_roll = self._h_roll[1:] + [state.roll_lataccel]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        step_idx = CONTEXT_LENGTH + self.n - 1

        error = target_lataccel - current_lataccel
        self._h_error = self._h_error[1:] + [error]
        error_integral = float(np.mean(self._h_error)) * DEL_T

        if step_idx < CONTROL_START_IDX:
            self._push(0.0, current_lataccel, state)
            return 0.0   # simulator overrides anyway

        obs = build_obs(target_lataccel, current_lataccel, state, future_plan,
                        self._h_act, self._h_lat, self._h_v, self._h_a, self._h_roll,
                        error_integral=error_integral)
        raw, val = self.ac.act(obs, self.det)

        delta  = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self._h_act[-1] + delta, *STEER_RANGE))
        self._push(action, current_lataccel, state)

        if step_idx < COST_END_IDX:
            self.traj.append(dict(obs=obs, raw=raw, val=val,
                                  tgt=target_lataccel, cur=current_lataccel))
        return action


def compute_rewards(traj):
    tgt = np.array([t['tgt'] for t in traj], np.float32)
    cur = np.array([t['cur'] for t in traj], np.float32)
    lat  = (tgt - cur)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER
    jerk = np.diff(cur, prepend=cur[0]) / DEL_T
    return (-(lat + jerk**2 * 100) / 500.0).astype(np.float32)


class RunningMeanStd:
    """Welford's online algorithm for running variance (used for return normalization)."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
    def update(self, x):
        x = np.asarray(x, np.float64).ravel()
        batch_mean, batch_var, batch_count = x.mean(), x.var(), len(x)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot
    @property
    def std(self): return np.sqrt(self.var + 1e-8)


class PPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.Adam(ac.actor.parameters(), lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(ac.critic.parameters(), lr=VF_LR, eps=1e-5)
        self._ret_rms = RunningMeanStd()  # for return normalization

    def set_lr(self, epoch, total):
        """Cosine annealing: lr = base * 0.5 * (1 + cos(pi * epoch / total))"""
        frac = 0.5 * (1.0 + np.cos(np.pi * epoch / total))
        for pg in self.pi_opt.param_groups: pg['lr'] = PI_LR * frac
        for pg in self.vf_opt.param_groups: pg['lr'] = VF_LR * frac
        return PI_LR * frac

    def gae(self, all_r, all_v, all_d):
        # Normalize rewards by running return std
        self._ret_rms.update(np.concatenate(all_r))
        rstd = self._ret_rms.std
        all_r = [r / rstd for r in all_r]

        advs, rets = [], []
        for r, v, d in zip(all_r, all_v, all_d):
            T = len(r)
            adv = np.zeros(T, np.float32)
            g = 0.0
            for t in range(T-1, -1, -1):
                nv = 0.0 if t == T-1 else v[t+1]
                g = (r[t] + GAMMA*nv*(1-d[t]) - v[t]) + GAMMA*LAMDA*(1-d[t])*g
                adv[t] = g
            advs.append(adv)
            rets.append(adv + v)
        a = np.concatenate(advs)
        r = np.concatenate(rets)
        return a, r  # advantage normalization moved to minibatch level

    def gae_gpu(self, rew_2d, val_2d, done_2d):
        """Vectorized GAE on GPU. rew/val/done are (N, S) CUDA tensors."""
        self._ret_rms.update(rew_2d.detach().cpu().numpy().ravel())
        rstd = max(self._ret_rms.std, 1e-8)
        rew_2d = rew_2d / rstd

        N, S = rew_2d.shape
        adv = torch.empty((N, S), dtype=torch.float32, device=rew_2d.device)
        g = torch.zeros(N, dtype=torch.float32, device=rew_2d.device)
        zero = torch.zeros(N, dtype=torch.float32, device=rew_2d.device)
        for t in range(S - 1, -1, -1):
            nv = val_2d[:, t + 1] if t < S - 1 else zero
            mask = 1.0 - done_2d[:, t]
            delta = rew_2d[:, t] + GAMMA * nv * mask - val_2d[:, t]
            g = delta + GAMMA * LAMDA * mask * g
            adv[:, t] = g
        ret = adv + val_2d
        return adv.reshape(-1), ret.reshape(-1)

    @staticmethod
    def _beta_sigma(a, b):  # std of 2*Beta(a,b)-1 in [-1,1]
        return 2.0 * torch.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))

    def update(self, all_obs, all_raw=None, all_rew=None, all_val=None,
               all_done=None, critic_only=False):
        if isinstance(all_obs, dict):
            # GPU fast path — pre-flattened tensors from _batched_rollout_gpu
            gd = all_obs
            obs_t = gd['obs']                          # (N*S, OBS) already on GPU
            raw_t = gd['raw'].unsqueeze(-1)            # (N*S, 1)
            adv_t, ret_t = self.gae_gpu(gd['rew'], gd['val_2d'], gd['done'])
        elif isinstance(all_obs[0], torch.Tensor):
            obs_t = torch.cat(all_obs, dim=0).to(DEV)
            raw_t = torch.cat(all_raw, dim=0).unsqueeze(-1).to(DEV)
            rew_np = [r.cpu().numpy() for r in all_rew]
            val_np = [v.cpu().numpy() for v in all_val]
            don_np = [d.cpu().numpy() for d in all_done]
            adv, ret = self.gae(rew_np, val_np, don_np)
            adv_t = torch.FloatTensor(adv).to(DEV)
            ret_t = torch.FloatTensor(ret).to(DEV)
        else:
            obs_t = torch.FloatTensor(np.concatenate(all_obs)).to(DEV)
            raw_t = torch.FloatTensor(np.concatenate(all_raw)).unsqueeze(-1).to(DEV)
            rew_np, val_np, don_np = all_rew, all_val, all_done
            adv, ret = self.gae(rew_np, val_np, don_np)
            adv_t = torch.FloatTensor(adv).to(DEV)
            ret_t = torch.FloatTensor(ret).to(DEV)

        x_t = ((raw_t + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)  # raw → Beta support (0,1)

        with torch.no_grad():
            a_old, b_old = self.ac.beta_params(obs_t)
            old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t.squeeze(-1))
            old_val = self.ac.critic(obs_t).squeeze(-1)

        N = len(obs_t)
        for _ in range(K_EPOCHS):
            for idx in torch.randperm(N).split(MINI_BS):
                # Per-minibatch advantage normalization
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                val = self.ac.critic(obs_t[idx]).squeeze(-1)
                v_clipped = old_val[idx] + (val - old_val[idx]).clamp(-10.0, 10.0)
                vf_loss = torch.max(
                    F.huber_loss(val, ret_t[idx], delta=10.0, reduction='none'),
                    F.huber_loss(v_clipped, ret_t[idx], delta=10.0, reduction='none'),
                ).mean()

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                    self.vf_opt.step()
                else:
                    a_cur, b_cur = self.ac.beta_params(obs_t[idx])
                    dist = torch.distributions.Beta(a_cur, b_cur)
                    lp   = dist.log_prob(x_t[idx].squeeze(-1))  # on (0,1)

                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio * mb_adv,
                        ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv).mean()
                    ent = dist.entropy().mean()   # +log2 is constant, drops out

                    loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                    self.pi_opt.step()
                    self.vf_opt.step()

        pi_val = pi_loss.item() if not critic_only else 0.0
        ent_val = ent.item() if not critic_only else 0.0
        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs_t[:1000])
            sigma = self._beta_sigma(a_d, b_d).mean().item()
        cur_lr = self.pi_opt.param_groups[0]['lr']
        return dict(pi=pi_val, vf=vf_loss.item(),
                    ent=ent_val, σ=sigma,
                    lr=cur_lr)


def build_obs_batch(target, current, roll_la, v_ego, a_ego,
                    h_act, h_lat, fplan_data, step_idx,
                    error_integral=None):
    N = target.shape[0]
    T = fplan_data['target_lataccel'].shape[1]

    if error_integral is None:
        error_integral = np.zeros(N, np.float32)

    error = target - current
    v2 = np.maximum(v_ego * v_ego, 1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    h_act32 = h_act.astype(np.float32) if h_act.dtype != np.float32 else h_act
    fplan_lat0 = fplan_data['target_lataccel'][:, min(step_idx + 1, T - 1)]
    fric = np.sqrt(current**2 + a_ego**2) / 7.0

    core = np.column_stack([
        target / S_LAT,
        current / S_LAT,
        error / S_LAT,
        k_tgt / S_CURV,
        k_cur / S_CURV,
        (k_tgt - k_cur) / S_CURV,
        v_ego / S_VEGO,
        a_ego / S_AEGO,
        roll_la / S_ROLL,
        h_act32[:, -1] / S_STEER,
        error_integral / S_LAT,
        (fplan_lat0 - target) / DEL_T / S_LAT,
        (current - h_lat[:, -1]) / DEL_T / S_LAT,
        (h_act32[:, -1] - h_act32[:, -2]) / DEL_T / S_STEER,
        fric,
        np.maximum(0.0, 1.0 - fric),
    ]).astype(np.float32)

    end = min(step_idx + FUTURE_PLAN_STEPS, T)
    fallback = {'target_lataccel': target, 'roll_lataccel': roll_la,
                'v_ego': v_ego, 'a_ego': a_ego}
    futures = []
    for attr, scale in [('target_lataccel', S_LAT), ('roll_lataccel', S_ROLL),
                        ('v_ego', S_VEGO), ('a_ego', S_AEGO)]:
        slc = fplan_data[attr][:, step_idx+1:end]
        if slc.shape[1] == 0:
            padded = np.repeat(fallback[attr][:, None], FUTURE_K, axis=1)
        elif slc.shape[1] < FUTURE_K:
            padded = np.concatenate([slc,
                np.repeat(slc[:, -1:], FUTURE_K - slc.shape[1], axis=1)], 1)
        else:
            padded = slc
        futures.append(padded.astype(np.float32) / scale)

    obs = np.concatenate([core, h_act32 / S_STEER, h_lat / S_LAT] + futures, axis=1)
    return np.clip(obs, -5.0, 5.0)


def batched_rollout(csv_files, ac, mdl_path, deterministic=False,
                    ort_session=None, csv_cache=None):
    # Build simulator: use cached data if available
    if csv_cache is not None:
        data, rng_rows = csv_cache.slice(csv_files)
        sim = BatchedSimulator(str(mdl_path), ort_session=ort_session,
                               cached_data=data, cached_rng=rng_rows)
    else:
        sim = BatchedSimulator(str(mdl_path), csv_files, ort_session=ort_session)
    N = sim.N
    T = sim.T

    if USE_CUDA:
        return _batched_rollout_gpu(sim, ac, N, T, deterministic)
    else:
        return _batched_rollout_cpu(sim, ac, N, T, deterministic)


def _batched_rollout_gpu(sim, ac, N, T, deterministic):
    """All-GPU rollout: controller builds obs on GPU, no CPU<->GPU per step."""
    OBS_DIM = 16 + HIST_LEN + HIST_LEN + FUTURE_K * 4
    max_steps = COST_END_IDX - CONTROL_START_IDX
    dg = sim.data_gpu  # GPU tensors: {roll_lataccel, v_ego, a_ego, target_lataccel, steer_command}

    # GPU ring buffers
    h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')

    # GPU obs buffer
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device='cuda')

    # GPU training data collection
    if not deterministic:
        all_obs = torch.empty((max_steps, N, OBS_DIM), dtype=torch.float32, device='cuda')
        all_raw = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        all_val = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        tgt_hist = torch.empty((max_steps, N), dtype=torch.float64, device='cuda')
        cur_hist = torch.empty((max_steps, N), dtype=torch.float64, device='cuda')
    step_ctr = 0

    def controller_fn(step_idx, sim_ref):
        nonlocal step_ctr

        target     = dg['target_lataccel'][:, step_idx]
        current_la = sim_ref.current_lataccel
        roll_la    = dg['roll_lataccel'][:, step_idx]
        v_ego      = dg['v_ego'][:, step_idx]
        a_ego      = dg['a_ego'][:, step_idx]

        cla32 = current_la.float()
        error = (target - current_la).float()

        h_error[:, :-1] = h_error[:, 1:]
        h_error[:, -1] = error
        error_integral = h_error.mean(dim=1) * DEL_T

        if step_idx < CONTROL_START_IDX:
            h_act[:, :-1] = h_act[:, 1:]
            h_act[:, -1] = 0.0
            h_act32[:, :-1] = h_act32[:, 1:]
            h_act32[:, -1] = 0.0
            h_lat[:, :-1] = h_lat[:, 1:]
            h_lat[:, -1] = cla32
            return torch.zeros(N, dtype=torch.float64, device='cuda')

        # === Obs building on GPU ===
        v2 = torch.clamp(v_ego * v_ego, min=1.0)
        k_tgt = (target - roll_la) / v2
        k_cur = (current_la - roll_la) / v2
        fplan_lat0 = dg['target_lataccel'][:, min(step_idx + 1, T - 1)]
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
            slc = dg[attr][:, step_idx+1:end]
            w = slc.shape[1]
            if w == 0:
                fb = dg[attr][:, step_idx]
                obs_buf[:, c:c+FUTURE_K] = (fb / scale).float().unsqueeze(1)
            elif w < FUTURE_K:
                obs_buf[:, c:c+w] = slc.float() / scale
                obs_buf[:, c+w:c+FUTURE_K] = (slc[:, -1:].float() / scale)
            else:
                obs_buf[:, c:c+FUTURE_K] = slc[:, :FUTURE_K].float() / scale
            c += FUTURE_K

        obs_buf.clamp_(-5.0, 5.0)

        with torch.inference_mode():
            a_p, b_p = ac.beta_params(obs_buf)
            val = ac.critic(obs_buf).squeeze(-1)

        if deterministic:
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
        else:
            x = torch.distributions.Beta(a_p, b_p).sample()
            raw = 2.0 * x - 1.0

        delta  = (raw.double() * DELTA_SCALE).clamp(-MAX_DELTA, MAX_DELTA)
        action = (h_act[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # Shift and write
        h_act[:, :-1] = h_act[:, 1:]
        h_act[:, -1] = action
        h_act32[:, :-1] = h_act32[:, 1:]
        h_act32[:, -1] = action.float()
        h_lat[:, :-1] = h_lat[:, 1:]
        h_lat[:, -1] = cla32

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[step_ctr] = obs_buf
            all_raw[step_ctr] = raw
            all_val[step_ctr] = val
            tgt_hist[step_ctr] = target
            cur_hist[step_ctr] = current_la
            step_ctr += 1

        return action

    cost_dict = sim.rollout(controller_fn)
    total_costs = cost_dict['total_cost']  # numpy (N,)

    if deterministic:
        return total_costs.tolist()

    # Training data is already on GPU — compute rewards on GPU
    S = step_ctr
    tgt_arr = tgt_hist[:S].T                                   # (N, S)
    cur_arr = cur_hist[:S].T
    lat_r = (tgt_arr - cur_arr)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk_r = torch.diff(cur_arr, dim=1, prepend=cur_arr[:, :1]) / DEL_T
    rew = (-(lat_r + jerk_r**2 * 100) / 500.0).float()        # (N, S)
    dones = torch.zeros((N, S), dtype=torch.float32, device='cuda')
    dones[:, -1] = 1.0

    # Return pre-flattened GPU tensors — no per-episode split/recat needed.
    obs_flat = all_obs[:S].permute(1, 0, 2).reshape(-1, OBS_DIM)  # (N*S, OBS)
    raw_flat = all_raw[:S].T.reshape(-1)                           # (N*S,)
    val_2d   = all_val[:S].T                                       # (N, S)
    return dict(obs=obs_flat, raw=raw_flat, val_2d=val_2d,
                rew=rew, done=dones, costs=total_costs)


def _batched_rollout_cpu(sim, ac, N, T, deterministic):
    """Original CPU path (unchanged logic, numpy-based)."""
    _dev = next(ac.parameters()).device
    h_act   = np.zeros((N, HIST_LEN), np.float64)
    h_act32 = np.zeros((N, HIST_LEN), np.float32)
    h_lat   = np.zeros((N, HIST_LEN), np.float32)
    h_error = np.zeros((N, HIST_LEN), np.float32)
    OBS_DIM = 16 + HIST_LEN + HIST_LEN + FUTURE_K * 4
    obs_buf = np.empty((N, OBS_DIM), np.float32)
    max_steps = COST_END_IDX - CONTROL_START_IDX
    all_obs = np.empty((max_steps, N, OBS_DIM), np.float32)
    all_raw = np.empty((max_steps, N), np.float32)
    all_val = np.empty((max_steps, N), np.float32)
    tgt_hist = np.empty((max_steps, N), np.float64)
    cur_hist = np.empty((max_steps, N), np.float64)
    step_ctr = 0
    fdata = sim.data

    def controller_fn(step_idx, target, current_la, state_dict, future_plan):
        nonlocal h_act, h_act32, h_lat, h_error, step_ctr
        roll_la = state_dict['roll_lataccel']
        v_ego   = state_dict['v_ego']
        a_ego   = state_dict['a_ego']
        cla32  = current_la.astype(np.float32)
        error  = (target - current_la).astype(np.float32)
        h_error[:, :-1] = h_error[:, 1:]
        h_error[:, -1] = error
        error_integral = h_error.mean(axis=1) * DEL_T
        if step_idx < CONTROL_START_IDX:
            h_act[:, :-1] = h_act[:, 1:]
            h_act[:, -1] = 0.0
            h_act32[:, :-1] = h_act32[:, 1:]
            h_act32[:, -1] = 0.0
            h_lat[:, :-1] = h_lat[:, 1:]
            h_lat[:, -1] = cla32
            return np.zeros(N)
        v2 = np.maximum(v_ego * v_ego, 1.0)
        k_tgt = (target - roll_la) / v2
        k_cur = (current_la - roll_la) / v2
        fplan_lat0 = fdata['target_lataccel'][:, min(step_idx + 1, T - 1)]
        fric = np.sqrt(current_la**2 + a_ego**2) / 7.0
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
        obs_buf[:, c] = np.maximum(0.0, 1.0 - fric);       c += 1
        obs_buf[:, c:c+HIST_LEN] = h_act32 / S_STEER;     c += HIST_LEN
        obs_buf[:, c:c+HIST_LEN] = h_lat / S_LAT;          c += HIST_LEN
        end = min(step_idx + FUTURE_PLAN_STEPS, T)
        for attr, scale in [('target_lataccel', S_LAT), ('roll_lataccel', S_ROLL),
                            ('v_ego', S_VEGO), ('a_ego', S_AEGO)]:
            slc = fdata[attr][:, step_idx+1:end]
            w = slc.shape[1]
            if w == 0:
                fb = {'target_lataccel': target, 'roll_lataccel': roll_la,
                      'v_ego': v_ego, 'a_ego': a_ego}[attr]
                obs_buf[:, c:c+FUTURE_K] = (fb / scale).astype(np.float32)[:, None]
            elif w < FUTURE_K:
                obs_buf[:, c:c+w] = slc.astype(np.float32) / scale
                obs_buf[:, c+w:c+FUTURE_K] = (slc[:, -1:].astype(np.float32) / scale)
            else:
                obs_buf[:, c:c+FUTURE_K] = slc[:, :FUTURE_K].astype(np.float32) / scale
            c += FUTURE_K
        np.clip(obs_buf, -5.0, 5.0, out=obs_buf)
        obs_t = torch.from_numpy(obs_buf).to(_dev)
        with torch.inference_mode():
            a_p, b_p = ac.beta_params(obs_t)
            val = ac.critic(obs_t).squeeze(-1).cpu().numpy()
        if deterministic:
            raw = (2.0 * a_p / (a_p + b_p) - 1.0).cpu().numpy()
        else:
            x   = torch.distributions.Beta(a_p, b_p).sample()
            raw = (2.0 * x - 1.0).cpu().numpy()
        delta  = np.clip(raw.astype(np.float64) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        action = np.clip(h_act[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, :-1] = h_act[:, 1:]
        h_act[:, -1] = action
        h_act32[:, :-1] = h_act32[:, 1:]
        h_act32[:, -1] = action
        h_lat[:, :-1] = h_lat[:, 1:]
        h_lat[:, -1] = cla32
        if step_idx < COST_END_IDX:
            all_obs[step_ctr] = obs_buf
            all_raw[step_ctr] = raw
            all_val[step_ctr] = val
            tgt_hist[step_ctr] = target
            cur_hist[step_ctr] = current_la
            step_ctr += 1
        return action

    cost_dict = sim.rollout(controller_fn)
    total_costs = cost_dict['total_cost']
    if deterministic:
        return total_costs.tolist()
    S = step_ctr
    obs_arr = np.ascontiguousarray(all_obs[:S].transpose(1, 0, 2))
    raw_arr = np.ascontiguousarray(all_raw[:S].T)
    val_arr = np.ascontiguousarray(all_val[:S].T)
    tgt_arr = tgt_hist[:S].T
    cur_arr = cur_hist[:S].T
    lat_r = (tgt_arr - cur_arr)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk_r = np.diff(cur_arr, axis=1, prepend=cur_arr[:, :1]) / DEL_T
    rew = (-(lat_r + jerk_r**2 * 100) / 500.0).astype(np.float32)
    dones = np.zeros((N, S), np.float32)
    dones[:, -1] = 1.0
    results = []
    for i in range(N):
        results.append((obs_arr[i], raw_arr[i], rew[i], val_arr[i], dones[i], float(total_costs[i])))
    return results


# ── persistent remote TCP clients (multi-remote) ────────────────────

_remote_socks = {}  # idx -> socket

def _tcp_send(sock, data: bytes):
    sock.sendall(struct.pack('>I', len(data)))
    sock.sendall(data)

def _tcp_recv(sock) -> bytes:
    hdr = _tcp_recvall(sock, 4)
    if not hdr:
        return b''
    length = struct.unpack('>I', hdr)[0]
    return _tcp_recvall(sock, length)

def _tcp_recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def _get_remote_sock(idx):
    if idx not in _remote_socks:
        host, port = REMOTE_HOSTS[idx], REMOTE_PORTS[idx]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        _remote_socks[idx] = s
        print(f"  [remote-{idx}] connected to {host}:{port}")
    return _remote_socks[idx]

def _close_remote_sock(idx):
    s = _remote_socks.pop(idx, None)
    if s is not None:
        try: s.close()
        except Exception: pass

def _remote_request(idx, ckpt_path, csv_list, mode='train', _retries=1):
    """Send ckpt + CSV list to remote server idx, receive NPZ results."""
    host = REMOTE_HOSTS[idx]
    try:
        sock = _get_remote_sock(idx)
        with open(ckpt_path, 'rb') as f:
            ckpt_bytes = f.read()
        payload = pickle.dumps({'mode': mode, 'ckpt': ckpt_bytes, 'csvs': csv_list})
        _tcp_send(sock, payload)
        resp = _tcp_recv(sock)
        if not resp:
            print(f"  [remote-{idx}] empty response")
            _close_remote_sock(idx)
            return []

        data = np.load(io.BytesIO(resp), allow_pickle=False)
        if mode == 'eval':
            return data['costs'].tolist()

        obs_all  = data['obs']
        raw_all  = data['raw']
        rew_all  = data['rew']
        val_all  = data['val']
        done_all = data['done']
        costs    = data['costs']
        ep_lens  = data['ep_lens']

        results = []
        offset = 0
        for i, L in enumerate(ep_lens):
            results.append((
                obs_all[offset:offset+L],
                raw_all[offset:offset+L],
                rew_all[offset:offset+L],
                val_all[offset:offset+L],
                done_all[offset:offset+L],
                float(costs[i]),
            ))
            offset += L
        return results
    except Exception as e:
        _close_remote_sock(idx)
        if _retries > 0:
            print(f"  [remote-{idx}] {e}, reconnecting...")
            time.sleep(0.5)
            return _remote_request(idx, ckpt_path, csv_list, mode, _retries=_retries-1)
        print(f"  [remote-{idx}] request failed: {e}")
        return []


def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    data = pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values,
    })
    steer = data['steer_command'].values
    tgt   = data['target_lataccel'].values
    obs_list, raw_list = [], []
    h_act  = [0.0] * HIST_LEN
    h_lat  = [0.0] * HIST_LEN
    h_v    = [0.0] * HIST_LEN
    h_a    = [0.0] * HIST_LEN
    h_roll = [0.0] * HIST_LEN

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        target_la = tgt[step_idx]
        current_la = tgt[step_idx]  # BC assumes perfect tracking

        state = State(
            roll_lataccel=data['roll_lataccel'].values[step_idx],
            v_ego=data['v_ego'].values[step_idx],
            a_ego=data['a_ego'].values[step_idx])
        fplan = FuturePlan(
            lataccel=tgt[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=data['roll_lataccel'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            v_ego=data['v_ego'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist(),
            a_ego=data['a_ego'].values[step_idx+1:step_idx+FUTURE_PLAN_STEPS].tolist())

        obs = build_obs(target_la, current_la, state, fplan,
                        h_act, h_lat, h_v, h_a, h_roll)

        prev_act = h_act[-1]
        delta = steer[step_idx] - prev_act
        raw_target = np.clip(delta / DELTA_SCALE, -1.0, 1.0)  # Beta support
        obs_list.append(obs)
        raw_list.append(raw_target)
        h_act = h_act[1:] + [steer[step_idx]]
        h_lat = h_lat[1:] + [tgt[step_idx]]
        h_v = h_v[1:] + [data['v_ego'].values[step_idx]]
        h_a = h_a[1:] + [data['a_ego'].values[step_idx]]
        h_roll = h_roll[1:] + [data['roll_lataccel'].values[step_idx]]

    return (np.array(obs_list, np.float32),
            np.array(raw_list, np.float32))


def pretrain_bc(ac, csv_files, epochs=BC_EPOCHS, lr=BC_LR, batch_size=BC_BS):
    print(f"BC pretrain: extracting from {len(csv_files)} CSVs ...")
    results = process_map(_bc_worker, [str(f) for f in csv_files],
                          max_workers=BC_WORKERS, chunksize=50, disable=False)
    all_obs = np.concatenate([r[0] for r in results])
    all_raw = np.concatenate([r[1] for r in results])
    N = len(all_obs)
    print(f"BC pretrain: {N} samples, {epochs} epochs")

    obs_t = torch.FloatTensor(all_obs).to(DEV)
    raw_t = torch.FloatTensor(all_raw).to(DEV)
    opt = optim.AdamW(ac.actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for idx in perm.split(batch_size):
            a_p, b_p = ac.beta_params(obs_t[idx])
            x_target = ((raw_t[idx] + 1.0) / 2.0).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x_target).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        print(f"  BC epoch {ep}: loss={total_loss/n_batches:.6f}  lr={opt.param_groups[0]['lr']:.1e}")

    print("BC pretrain done.\n")


_W = {}   # per-process globals

def _pool_init(mdl_path):
    pool_init(mdl_path)
    torch.set_num_threads(1)
    _W['mdl'] = TinyPhysicsModel(str(mdl_path), debug=False)  # for BATCHED=0
    _W['ckpt_mtime'] = None
    _W['ac'] = None

def _load_ckpt(ckpt):
    mtime = os.path.getmtime(ckpt)
    if _W.get('ckpt_mtime') != mtime:
        data = torch.load(ckpt, weights_only=False, map_location='cpu')
        ac = ActorCritic()
        ac.load_state_dict(data['ac'])
        ac.eval()
        _W['ac'] = ac
        _W['ckpt_mtime'] = mtime

def _batched_train_worker(args):
    csv_chunk, ckpt = args
    _load_ckpt(ckpt)
    c = get_pool_cache()
    return batched_rollout(csv_chunk, _W['ac'], c['model_path'],
                           deterministic=False, ort_session=c['ort_session'])

def _batched_eval_worker(args):
    csv_chunk, ckpt = args
    _load_ckpt(ckpt)
    c = get_pool_cache()
    return batched_rollout(csv_chunk, _W['ac'], c['model_path'],
                           deterministic=True, ort_session=c['ort_session'])

def _seq_train_worker(args):
    csv, ckpt = args
    _load_ckpt(ckpt)
    ctrl = DeltaController(_W['ac'], deterministic=False)
    sim = TinyPhysicsSimulator(_W['mdl'], str(csv), controller=ctrl, debug=False)
    cost = sim.rollout()
    T = len(ctrl.traj)
    return (np.array([t['obs'] for t in ctrl.traj], np.float32),
            np.array([t['raw'] for t in ctrl.traj], np.float32),
            compute_rewards(ctrl.traj),
            np.array([t['val'] for t in ctrl.traj], np.float32),
            np.concatenate([np.zeros(T-1, np.float32), [1.0]]),
            cost['total_cost'])

def _seq_eval_worker(args):
    csv, ckpt = args
    _load_ckpt(ckpt)
    ctrl = DeltaController(_W['ac'], deterministic=True)
    sim = TinyPhysicsSimulator(_W['mdl'], str(csv), controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


class Ctx:
    def __init__(self):
        self.ac  = ActorCritic()
        self.ppo = PPO(self.ac)
        self.mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
        all_f = sorted((ROOT / 'data').glob('*.csv'))
        self.va_f = all_f[:EVAL_N]
        rest = all_f[EVAL_N:]
        random.seed(42)
        random.shuffle(rest)
        self.tr_f = rest
        self.best = float('inf')
        self.best_ep = -1
        if USE_CUDA:
            from tinyphysics_batched import make_ort_session
            self.ort_session = make_ort_session(self.mdl_path)
            self.pool = None
            # Pre-load all CSVs once (training + eval)
            all_csv = list(set([str(f) for f in self.tr_f + self.va_f]))
            self.csv_cache = CSVCache(all_csv)
        else:
            self.ort_session = None
            self.csv_cache = None
            self.pool = multiprocessing.Pool(
                WORKERS, initializer=_pool_init, initargs=(self.mdl_path,))
        self.ac.to(DEV)

    def save_ckpt(self, path=TMP):
        torch.save({'ac': self.ac.state_dict()}, path)

    def save_best(self):
        rms = self.ppo._ret_rms
        torch.save({
            'ac': self.ac.state_dict(),
            'pi_opt': self.ppo.pi_opt.state_dict(),
            'vf_opt': self.ppo.vf_opt.state_dict(),
            'ret_rms': {'mean': rms.mean, 'var': rms.var, 'count': rms.count},
        }, BEST_PT)


def _split_files(all_files):
    """Split files across local + N remotes according to FRAC weights."""
    N = len(all_files)
    remote_slices = []
    offset = 0
    for frac in FRAC_REMOTES:
        n_r = int(N * frac)
        remote_slices.append(all_files[offset:offset+n_r])
        offset += n_r
    local_files = all_files[offset:]
    return local_files, remote_slices


def evaluate(ctx, files, n=EVAL_N):
    ctx.save_ckpt(TMP)
    all_files = files[:n]

    local_files, remote_slices = _split_files(all_files) if USE_REMOTE else (all_files, [])

    # --- launch remote threads in parallel ---
    remote_threads = []
    remote_results = [[] for _ in range(len(remote_slices))]
    for ri, r_files in enumerate(remote_slices):
        if not r_files:
            continue
        r_csvs = [str(Path(f).relative_to(ROOT)) for f in r_files]
        def _do(idx=ri, csvs=r_csvs):
            remote_results[idx] = _remote_request(idx, str(TMP), csvs, mode='eval')
        t = threading.Thread(target=_do)
        t.start()
        remote_threads.append(t)

    if USE_CUDA:
        costs = batched_rollout(local_files, ctx.ac, ctx.mdl_path,
                                deterministic=True, ort_session=ctx.ort_session,
                                csv_cache=ctx.csv_cache)
    elif BATCHED:
        costs = run_parallel_chunked(ctx.pool, local_files,
                                     _batched_eval_worker, WORKERS,
                                     extra_args=(str(TMP),))
    else:
        args = [(f, str(TMP)) for f in local_files]
        costs = list(ctx.pool.map(_seq_eval_worker, args))

    for t in remote_threads:
        t.join()
    for rc in remote_results:
        if rc:
            costs = costs + rc

    if not costs:
        return float('inf'), 0.0
    return float(np.mean(costs)), float(np.std(costs))


def train_one_epoch(epoch, ctx, warmup_off=0):
    if DECAY_LR:
        ctx.ppo.set_lr(epoch, MAX_EP)
    t0 = time.time()
    if USE_REMOTE:
        ctx.save_ckpt(TMP)
    batch = random.sample(ctx.tr_f, min(CSVS_EPOCH, len(ctx.tr_f)))

    local_batch, remote_slices = _split_files(batch) if USE_REMOTE else (batch, [])

    # --- launch remote threads in parallel ---
    remote_threads = []
    remote_results = [[] for _ in range(len(remote_slices))]
    remote_times = [0.0] * len(remote_slices)
    for ri, r_files in enumerate(remote_slices):
        if not r_files:
            continue
        r_csvs = [str(Path(f).relative_to(ROOT)) for f in r_files]
        def _do(idx=ri, csvs=r_csvs):
            rt0 = time.time()
            remote_results[idx] = _remote_request(idx, str(TMP), csvs, mode='train')
            remote_times[idx] = time.time() - rt0
        t = threading.Thread(target=_do)
        t.start()
        remote_threads.append(t)

    tl0 = time.time()
    if USE_CUDA:
        res = batched_rollout(local_batch, ctx.ac, ctx.mdl_path,
                              deterministic=False, ort_session=ctx.ort_session,
                              csv_cache=ctx.csv_cache)
    elif BATCHED:
        res = run_parallel_chunked(ctx.pool, local_batch,
                                   _batched_train_worker, WORKERS,
                                   extra_args=(str(TMP),))
    else:
        args = [(f, str(TMP)) for f in local_batch]
        res = list(ctx.pool.map(_seq_train_worker, args))
    local_time = time.time() - tl0

    # --- collect remote results and merge ---
    for t in remote_threads:
        t.join()
    for ri, rr in enumerate(remote_results):
        if rr:
            res = res + rr
    if USE_REMOTE:
        parts = [f"local={len(local_batch)}csvs {local_time:.0f}s"]
        for ri, r_files in enumerate(remote_slices):
            parts.append(f"r{ri}={len(r_files)}csvs {remote_times[ri]:.0f}s")
        print(f"  [split] {' | '.join(parts)}")
    tc = time.time() - t0

    if not res:
        print(f"E{epoch:3d}  NO DATA — skipping update")
        return

    t1 = time.time()
    co = epoch < (CRITIC_WARMUP - warmup_off)
    if isinstance(res, dict):
        # GPU fast path — pre-flattened tensors
        info = ctx.ppo.update(res, critic_only=co)
        costs = res['costs'].tolist()
    else:
        # CPU path — list of per-episode tuples
        info = ctx.ppo.update(
            [r[0] for r in res], [r[1] for r in res],
            [r[2] for r in res], [r[3] for r in res],
            [r[4] for r in res], critic_only=co)
        costs = [r[5] for r in res]
    tu = time.time() - t1
    phase = "  [critic warmup]" if co else ""
    line = (f"E{epoch:3d}  train={np.mean(costs):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  lr={info['lr']:.1e}  ⏱{tc:.0f}+{tu:.0f}s{phase}")

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(ctx, ctx.va_f)
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            ctx.save_best()
            mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"

    print(line)


def train():
    ctx = Ctx()

    resumed = False
    if RESUME and (EXP_DIR / 'best_model.pt').exists():
        ckpt = torch.load(EXP_DIR / 'best_model.pt', weights_only=False, map_location=DEV)
        ctx.ac.load_state_dict(ckpt['ac'])
        if 'pi_opt' in ckpt:
            ctx.ppo.pi_opt.load_state_dict(ckpt['pi_opt'])
            ctx.ppo.vf_opt.load_state_dict(ckpt['vf_opt'])
            # Re-apply eps=1e-5 (old checkpoints saved with 1e-8)
            for opt in (ctx.ppo.pi_opt, ctx.ppo.vf_opt):
                for pg in opt.param_groups:
                    pg['eps'] = 1e-5
            if 'ret_rms' in ckpt:
                r = ckpt['ret_rms']
                ctx.ppo._ret_rms.mean = r['mean']
                ctx.ppo._ret_rms.var = r['var']
                ctx.ppo._ret_rms.count = r['count']
                print(f"Resumed from best_model.pt (with optimizer + ret_rms state)")
            else:
                print(f"Resumed from best_model.pt (with optimizer state, ret_rms fresh)")
        else:
            print(f"Resumed from best_model.pt (weights only)")
        resumed = True
    else:
        all_csvs = sorted((ROOT / 'data').glob('*.csv'))
        pretrain_bc(ctx.ac, all_csvs)

    vm, vs = evaluate(ctx, ctx.va_f)
    ctx.best, ctx.best_ep = vm, 'init'
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")
    ctx.save_best()

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  workers={WORKERS}  dev={DEV}  (batched chunks)")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}"
          f"  dist=Beta  Δscale={DELTA_SCALE}"
          f"  layers={A_LAYERS}+{C_LAYERS}  K={K_EPOCHS}  dim={STATE_DIM}\n")

    warmup_off = CRITIC_WARMUP if resumed else 0
    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx, warmup_off=warmup_off)

    print(f"\nDone. Best val: {ctx.best:.1f} (epoch {ctx.best_ep})")
    ctx.save_ckpt(EXP_DIR / 'final_model.pt')
    if ctx.pool:
        ctx.pool.terminate()
        ctx.pool.join()
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
