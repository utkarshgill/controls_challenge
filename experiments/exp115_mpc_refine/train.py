# exp115 — Clean fork of exp095 + receding-horizon MPC refinement
#
# Phase 1: BC pretrain (from MPC expert data or CSV steer commands)
# Phase 2: PPO fine-tune (batch-of-batch, same as exp055/exp095)
# Phase 3 (separate script): MPC action refinement using trained prior
#
# Stripped: DELTA_SCALE_DECAY, RESUME_DS, RESET_CRITIC, COMPILE,
#           CVAR_ALPHA, ADV_CLIP, EXPECTED_REWARD, TEMP_RAMP, ACT_SMOOTH

import numpy as np, pandas as pd, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    ACC_G, State, FuturePlan,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEV = torch.device("cuda")

# ── architecture ──────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS, C_LAYERS = 4, 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))

# ── scaling ───────────────────────────────────────────────────
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02

# ── PPO ───────────────────────────────────────────────────────
PI_LR       = float(os.getenv("PI_LR", "3e-4"))
VF_LR       = float(os.getenv("VF_LR", "3e-4"))
LR_MIN      = 5e-5
GAMMA       = float(os.getenv("GAMMA", "0.95"))
LAMDA       = float(os.getenv("LAMDA", "0.9"))
K_EPOCHS    = int(os.getenv("K_EPOCHS", "4"))
EPS_CLIP    = 0.2
VF_COEF     = 1.0
ENT_COEF    = float(os.getenv("ENT_COEF", "0.003"))
SIGMA_FLOOR = float(os.getenv("SIGMA_FLOOR", "0.01"))
SIGMA_FLOOR_COEF = float(os.getenv("SIGMA_FLOOR_COEF", "0.5"))
MINI_BS     = int(os.getenv("MINI_BS", "25_000"))
CRITIC_WARMUP = int(os.getenv("CRITIC_WARMUP", "3"))

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS   = int(os.getenv("BC_EPOCHS", "20"))
BC_LR       = float(os.getenv("BC_LR", "0.01"))
BC_BS       = int(os.getenv("BC_BS", "2048"))
BC_GRAD_CLIP = 2.0
BC_DATA     = os.getenv("BC_DATA", "")  # path to bc_data.npz (MPC expert pairs)

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH  = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "10"))
MAX_EP      = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY  = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N      = 100
RESUME      = os.getenv("RESUME", "0") == "1"
RESUME_OPT  = os.getenv("RESUME_OPT", "1") == "1"
REWARD_RMS_NORM = os.getenv("REWARD_RMS_NORM", "1") == "1"
ADV_NORM    = os.getenv("ADV_NORM", "1") == "1"

def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + np.cos(np.pi * epoch / max_ep))

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

# ── obs layout ────────────────────────────────────────────────
C = 16
H1 = C + HIST_LEN          # 36
H2 = H1 + HIST_LEN         # 56
F_LAT = H2                  # 56
F_ROLL = F_LAT + FUTURE_K   # 106
F_V = F_ROLL + FUTURE_K     # 156
F_A = F_V + FUTURE_K        # 206
OBS_DIM = F_A + FUTURE_K    # 256


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

def _precompute_future_windows(dg):
    def _w(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()
    return {k: _w(dg[k]) for k in ("target_lataccel", "roll_lataccel", "v_ego", "a_ego")}

def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale; return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale

def fill_obs(buf, target, current, roll_la, v_ego, a_ego,
             h_act, h_lat, hist_head, ei, future, step_idx):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    pa = h_act[:, hist_head]; pa2 = h_act[:, (hist_head - 1) % HIST_LEN]
    pl = h_lat[:, hist_head]
    buf[:, 0] = target / S_LAT;  buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV;  buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO;  buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL; buf[:, 9] = pa / S_STEER
    buf[:, 10] = ei / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - pl) / DEL_T / S_LAT
    buf[:, 13] = (pa - pa2) / DEL_T / S_STEER
    buf[:, 14] = fric; buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)
    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  Rollout
# ══════════════════════════════════════════════════════════════

def rollout(csv_files, ac, mdl_path, ort_session, csv_cache,
            deterministic=False, ds=DELTA_SCALE):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session,
                           cached_data=data, cached_rng=rng)
    N = sim.N; dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX
    future = _precompute_future_windows(dg)

    h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    if not deterministic:
        all_obs  = torch.empty((max_steps, N, OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw  = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_logp = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")
        all_val  = torch.empty((max_steps, N), dtype=torch.float32, device="cuda")

    si = 0; hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_err = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_err
        ei = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0; h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32; hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        fill_obs(obs_buf, target.float(), cur32,
                 dg["roll_lataccel"][:, step_idx].float(),
                 dg["v_ego"][:, step_idx].float(),
                 dg["a_ego"][:, step_idx].float(),
                 h_act32, h_lat, hist_head, ei, future, step_idx)

        with torch.inference_mode():
            logits = ac.actor(obs_buf)
            val = ac.critic(obs_buf).squeeze(-1)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0

        if deterministic:
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            x = dist.sample(); raw = 2.0 * x - 1.0
            logp = dist.log_prob(x)

        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, next_head] = action; h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32; hist_head = next_head

        if not deterministic and step_idx < COST_END_IDX:
            all_obs[si] = obs_buf; all_raw[si] = raw
            all_logp[si] = logp; all_val[si] = val; si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    if deterministic:
        return costs.tolist()

    S = si; start = CONTROL_START_IDX; end = start + S
    pred = sim.current_lataccel_history[:, start:end].float()
    target = dg["target_lataccel"][:, start:end].float()
    lat_r = (target - pred)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    rew = -(lat_r + jerk**2 * 100).float()
    dones = torch.zeros((N, S), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    return dict(
        obs=all_obs[:S].permute(1, 0, 2).reshape(-1, OBS_DIM),
        raw=all_raw[:S].T.reshape(-1),
        old_logp=all_logp[:S].T.reshape(-1),
        val_2d=all_val[:S].T,
        rew=rew, done=dones, costs=costs)


# ══════════════════════════════════════════════════════════════
#  BC Pretrain
# ══════════════════════════════════════════════════════════════

def _future_raw(fplan, attr, fallback, k=FUTURE_K):
    vals = getattr(fplan, attr, None) if fplan else None
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    elif vals is not None and len(vals) > 0:
        return np.pad(np.array(vals, np.float32), (0, k - len(vals)), "edge")
    return np.full(k, fallback, dtype=np.float32)

def _build_obs_bc(target, current, state, fplan, hist_act, hist_lat):
    k_tgt = (target - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    k_cur = (current - state.roll_lataccel) / max(state.v_ego**2, 1.0)
    fp0 = (getattr(fplan, "lataccel", None) or [target])[0]
    fric = np.sqrt(current**2 + state.a_ego**2) / 7.0
    core = np.array([
        target/S_LAT, current/S_LAT, (target-current)/S_LAT,
        k_tgt/S_CURV, k_cur/S_CURV, (k_tgt-k_cur)/S_CURV,
        state.v_ego/S_VEGO, state.a_ego/S_AEGO, state.roll_lataccel/S_ROLL,
        hist_act[-1]/S_STEER, 0.0,
        (fp0-target)/DEL_T/S_LAT, (current-hist_lat[-1])/DEL_T/S_LAT,
        (hist_act[-1]-hist_act[-2])/DEL_T/S_STEER, fric, max(0.0,1.0-fric),
    ], np.float32)
    return np.clip(np.concatenate([
        core,
        np.array(hist_act, np.float32) / S_STEER,
        np.array(hist_lat, np.float32) / S_LAT,
        _future_raw(fplan, "lataccel", target) / S_LAT,
        _future_raw(fplan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
        _future_raw(fplan, "v_ego", state.v_ego) / S_VEGO,
        _future_raw(fplan, "a_ego", state.a_ego) / S_AEGO,
    ]), -5.0, 5.0)

def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df["roll"].values) * ACC_G
    v_ego = df["vEgo"].values; a_ego = df["aEgo"].values
    tgt = df["targetLateralAcceleration"].values; steer = -df["steerCommand"].values
    obs_list, raw_list = [], []
    h_act, h_lat = [0.0]*HIST_LEN, [0.0]*HIST_LEN
    for si in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        state = State(roll_lataccel=roll_la[si], v_ego=v_ego[si], a_ego=a_ego[si])
        fplan = FuturePlan(
            lataccel=tgt[si+1:si+FUTURE_PLAN_STEPS].tolist(),
            roll_lataccel=roll_la[si+1:si+FUTURE_PLAN_STEPS].tolist(),
            v_ego=v_ego[si+1:si+FUTURE_PLAN_STEPS].tolist(),
            a_ego=a_ego[si+1:si+FUTURE_PLAN_STEPS].tolist())
        obs_list.append(_build_obs_bc(tgt[si], tgt[si], state, fplan, h_act, h_lat))
        raw_list.append(np.clip((steer[si]-h_act[-1])/DELTA_SCALE, -1.0, 1.0))
        h_act = h_act[1:] + [steer[si]]; h_lat = h_lat[1:] + [tgt[si]]
    return (np.array(obs_list, np.float32), np.array(raw_list, np.float32))

def pretrain_bc(ac, all_csvs):
    bc_path = Path(BC_DATA) if BC_DATA else None
    if bc_path and bc_path.exists():
        print(f"BC pretrain: loading MPC expert data from {bc_path}")
        d = np.load(bc_path)
        obs_t = torch.FloatTensor(d["obs"]).to(DEV)
        raw_t = torch.FloatTensor(d["raw"]).to(DEV)
    else:
        print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
        results = process_map(_bc_worker, [str(f) for f in all_csvs],
                              max_workers=10, chunksize=50)
        obs_t = torch.FloatTensor(np.concatenate([r[0] for r in results])).to(DEV)
        raw_t = torch.FloatTensor(np.concatenate([r[1] for r in results])).to(DEV)
    N = len(obs_t)
    print(f"BC pretrain: {N} samples, {BC_EPOCHS} epochs")
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)
    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            a_p, b_p = ac.beta_params(obs_t[idx])
            x = ((raw_t[idx]+1)/2).clamp(1e-6, 1-1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step(); total += loss.item(); nb += 1
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
        self.var = (self.var*self.count + batch_var*batch_count
                    + delta**2*self.count*batch_count/tot) / tot
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
    def _beta_sigma_raw(a, b):
        return 2.0 * torch.sqrt(a*b / ((a+b)**2 * (a+b+1)))

    def _gae(self, rew, val, done):
        if REWARD_RMS_NORM:
            with torch.no_grad():
                flat = rew.reshape(-1)
                self._rms.update(flat.mean().item(), flat.var().item(), flat.numel())
            rew = rew / max(self._rms.std, 1e-8)
        N, S = rew.shape
        adv = torch.empty_like(rew)
        g = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(S-1, -1, -1):
            nv = val[:, t+1] if t < S-1 else g
            mask = 1.0 - done[:, t]
            g = (rew[:, t] + GAMMA*nv*mask - val[:, t]) + GAMMA*LAMDA*mask*g
            adv[:, t] = g
        return adv.reshape(-1), (adv + val).reshape(-1)

    def update(self, gd, critic_only=False, ds=DELTA_SCALE):
        obs = gd["obs"]; raw = gd["raw"].unsqueeze(-1)
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])
        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_2d = adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)
        x_t = ((raw+1)/2).clamp(1e-6, 1-1e-6); ds = float(ds)

        n_vf, n_actor, vf_sum, pi_sum, ent_sum, sp_sum = 0, 0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            old_lp = gd.get("old_logp")
            if old_lp is None:
                a_old, b_old = self.ac.beta_params(obs)
                old_lp = torch.distributions.Beta(a_old, b_old).log_prob(x_t.squeeze(-1))

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(obs), device="cuda").split(MINI_BS):
                mb_adv = adv_t[idx]
                if ADV_NORM:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                val = self.ac.critic(obs[idx]).squeeze(-1)
                vf_loss = F.mse_loss(val, ret_t[idx])
                bs = idx.numel(); vf_sum += vf_loss.item()*bs; n_vf += bs

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True); vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.vf_opt.step()
                else:
                    a_c, b_c = self.ac.beta_params(obs[idx])
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    pi_loss = -torch.min(
                        ratio*mb_adv, ratio.clamp(1-EPS_CLIP, 1+EPS_CLIP)*mb_adv).mean()
                    ent = dist.entropy().mean()
                    sigma_pen = F.relu(SIGMA_FLOOR - self._beta_sigma_raw(a_c, b_c).mean()*ds)
                    loss = pi_loss + VF_COEF*vf_loss - ENT_COEF*ent + SIGMA_FLOOR_COEF*sigma_pen
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                    self.pi_opt.step(); self.vf_opt.step()
                    pi_sum += pi_loss.item()*bs; ent_sum += ent.item()*bs
                    sp_sum += sigma_pen.item()*bs; n_actor += bs

        with torch.no_grad():
            a_d, b_d = self.ac.beta_params(obs[:1000])
            σraw = self._beta_sigma_raw(a_d, b_d).mean().item()
        return dict(
            pi=pi_sum/max(1,n_actor) if not critic_only else 0.0,
            vf=vf_sum/max(1,n_vf),
            ent=ent_sum/max(1,n_actor) if not critic_only else 0.0,
            σ=σraw*ds, σraw=σraw,
            σpen=sp_sum/max(1,n_actor) if not critic_only else 0.0,
            lr=self.pi_opt.param_groups[0]["lr"])


# ══════════════════════════════════════════════════════════════
#  Evaluate + TrainingContext + train_one_epoch
# ══════════════════════════════════════════════════════════════

def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE):
    costs = rollout(files, ac, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
    return float(np.mean(costs)), float(np.std(costs))

class TrainingContext:
    def __init__(self):
        self.ac = ActorCritic().to(DEV)
        self.ppo = PPO(self.ac)
        self.mdl_path = ROOT / "models" / "tinyphysics.onnx"
        self.ort_sess = make_ort_session(self.mdl_path)
        self.all_csv = sorted((ROOT / "data").glob("*.csv"))
        self.tr_f = self.all_csv
        self.va_f = self.all_csv[:EVAL_N]
        self.csv_cache = CSVCache([str(f) for f in self.all_csv])
        self.best = float("inf"); self.best_ep = "init"

    def save_best(self):
        torch.save({
            "ac": self.ac.state_dict(),
            "pi_opt": self.ppo.pi_opt.state_dict(),
            "vf_opt": self.ppo.vf_opt.state_dict(),
            "ret_rms": {"mean": self.ppo._rms.mean, "var": self.ppo._rms.var,
                        "count": self.ppo._rms.count},
        }, BEST_PT)

    def resume(self):
        if RESUME and BEST_PT.exists():
            ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
            self.ac.load_state_dict(ckpt["ac"])
            if RESUME_OPT and "pi_opt" in ckpt:
                self.ppo.pi_opt.load_state_dict(ckpt["pi_opt"])
                self.ppo.vf_opt.load_state_dict(ckpt["vf_opt"])
                if "ret_rms" in ckpt:
                    r = ckpt["ret_rms"]
                    self.ppo._rms.mean, self.ppo._rms.var, self.ppo._rms.count = \
                        r["mean"], r["var"], r["count"]
                print("Resumed with optimizer state + RMS")
            else:
                print("Resumed weights only")
            print(f"Resumed from {BEST_PT.name}")
            return True
        return False

    def baseline(self):
        vm, vs = evaluate(self.ac, self.va_f, self.mdl_path, self.ort_sess,
                          self.csv_cache, ds=DELTA_SCALE)
        self.best = vm; self.best_ep = "init"
        print(f"Baseline: {vm:.1f} ± {vs:.1f}  (Δs={DELTA_SCALE:.4f})")

def train_one_epoch(epoch, ctx):
    """One epoch: rollout → PPO update → log → eval."""
    pi_lr = lr_schedule(epoch, MAX_EP, PI_LR)
    vf_lr = lr_schedule(epoch, MAX_EP, VF_LR)
    if RESUME and RESUME_OPT and epoch == 0:
        pi_lr = ctx.ppo.pi_opt.param_groups[0]["lr"]
        vf_lr = ctx.ppo.vf_opt.param_groups[0]["lr"]
    for pg in ctx.ppo.pi_opt.param_groups: pg["lr"] = pi_lr
    for pg in ctx.ppo.vf_opt.param_groups: pg["lr"] = vf_lr

    t0 = time.time()
    n_routes = min(CSVS_EPOCH, len(ctx.tr_f)) // SAMPLES_PER_ROUTE
    batch = random.sample(ctx.tr_f, max(n_routes, 1))
    batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]
    res = rollout(batch, ctx.ac, ctx.mdl_path, ctx.ort_sess, ctx.csv_cache,
                  deterministic=False, ds=DELTA_SCALE)
    t1 = time.time()

    co = epoch < CRITIC_WARMUP
    info = ctx.ppo.update(res, critic_only=co, ds=DELTA_SCALE)
    tu = time.time() - t1

    phase = "  [critic warmup]" if co else ""
    line = (f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}"
            f"  σraw={info['σraw']:.4f}  σpen={info['σpen']:.4f}"
            f"  π={info['pi']:+.4f}  vf={info['vf']:.1f}  H={info['ent']:.2f}"
            f"  Δs={DELTA_SCALE:.4f}  lr={info['lr']:.1e}  ⏱{t1-t0:.0f}+{tu:.0f}s{phase}")

    if epoch % EVAL_EVERY == 0:
        vm, vs = evaluate(ctx.ac, ctx.va_f, ctx.mdl_path, ctx.ort_sess,
                          ctx.csv_cache, ds=DELTA_SCALE)
        mk = ""
        if vm < ctx.best:
            ctx.best, ctx.best_ep = vm, epoch
            ctx.save_best(); mk = " ★"
        line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"

    print(line)

def train():
    ctx = TrainingContext()
    if not ctx.resume():
        pretrain_bc(ctx.ac, ctx.all_csv)
    ctx.baseline()

    n_r = min(CSVS_EPOCH, len(ctx.tr_f)) // SAMPLES_PER_ROUTE
    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(f"  K={SAMPLES_PER_ROUTE} → {n_r} routes × {SAMPLES_PER_ROUTE} = {n_r*SAMPLES_PER_ROUTE}/epoch")
    print(f"  π_lr={PI_LR} vf_lr={VF_LR} ent={ENT_COEF} γ={GAMMA} λ={LAMDA}\n")

    for epoch in range(MAX_EP):
        train_one_epoch(epoch, ctx)

    print(f"\nDone. Best: {ctx.best:.1f} (epoch {ctx.best_ep})")
    torch.save({"ac": ctx.ac.state_dict()}, EXP_DIR / "final_model.pt")

if __name__ == "__main__":
    train()
