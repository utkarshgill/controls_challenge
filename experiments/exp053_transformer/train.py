# exp053 — Encoder-Decoder Transformer with continuous Beta delta control
#
# Encoder: future plan [50, 4] with bidirectional attention
# Decoder: steer history [20, 6] with causal self-attn + cross-attn to plan
# Action space: continuous delta via Beta distribution (exp052 style)
# All tensors GPU-resident.

import math
import numpy as np, pandas as pd, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
torch.set_float32_matmul_precision('high')
from contextlib import nullcontext
from pathlib import Path
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    ACC_G, State, FuturePlan)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42); np.random.seed(42)
DEV = torch.device('cuda')

# Some newer GPUs/drivers can crash in fused SDPA kernels with torch transformer ops.
# Use math SDPA by default for stability; override with FORCE_MATH_SDP=0.
FORCE_MATH_SDP = os.getenv('FORCE_MATH_SDP', '1') == '1'
if DEV.type == 'cuda' and FORCE_MATH_SDP and hasattr(torch.backends, 'cuda'):
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)

# ── architecture ──────────────────────────────────────────────
HIST_LEN   = 20
PLAN_LEN   = 50
D_MODEL    = int(os.getenv('D_MODEL', '64'))
N_HEADS    = int(os.getenv('N_HEADS', '4'))
N_ENC      = int(os.getenv('N_ENC', '2'))
N_DEC      = int(os.getenv('N_DEC', '2'))
D_FFN      = int(os.getenv('D_FFN', '256'))
DELTA_SCALE_MAX = float(os.getenv('DELTA_SCALE_MAX', '0.25'))
DELTA_SCALE_MIN = float(os.getenv('DELTA_SCALE_MIN', '0.25'))
MAX_DELTA       = 0.5

# ── scaling ───────────────────────────────────────────────────
S_LAT  = 5.0
S_ROLL = 2.0
S_VEGO = 40.0
S_AEGO = 4.0
S_STEER = 2.0

# ── PPO ───────────────────────────────────────────────────────
PI_LR       = float(os.getenv('LR', os.getenv('PI_LR', '3e-4')))
VF_LR       = float(os.getenv('VF_LR', os.getenv('LR', os.getenv('PI_LR', '3e-4'))))
LR_MIN      = 3e-5
GAMMA       = 0.95
LAMDA       = 0.9
K_EPOCHS    = int(os.getenv('K_EPOCHS', '2'))
EPS_CLIP    = 0.2
VF_COEF     = 1.0
ENT_COEF    = float(os.getenv('ENT_COEF', '0.001'))
ACT_SMOOTH  = float(os.getenv('ACT_SMOOTH', '10.0'))
MINI_BS     = int(os.getenv('MINI_BS', '8192'))
CRITIC_WARMUP = 3

# ── BC ────────────────────────────────────────────────────────
BC_EPOCHS    = int(os.getenv('BC_EPOCHS', '20'))
BC_LR        = float(os.getenv('BC_LR', '1e-3'))
BC_BS        = int(os.getenv('BC_BS', '8192'))
BC_GRAD_CLIP = 2.0

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv('CSVS', '500'))
MAX_EP     = int(os.getenv('EPOCHS', '200'))
EVAL_EVERY = 5
EVAL_N     = 100
RESUME     = os.getenv('RESUME', '0') == '1'
RESUME_OPT = os.getenv('RESUME_OPT', '0') == '1'
DEBUG      = int(os.getenv('DEBUG', '0'))
USE_AMP    = os.getenv('AMP', '1') == '1'
USE_COMPILE = os.getenv('COMPILE', '0') == '1'
DELTA_SCALE_DECAY = os.getenv('DELTA_SCALE_DECAY', '0') == '1'
ACTION_MODE = 'continuous_beta_v2_sep_critic'

def delta_scale(epoch, max_ep):
    return DELTA_SCALE_MIN + 0.5 * (DELTA_SCALE_MAX - DELTA_SCALE_MIN) * (1 + np.cos(np.pi * epoch / max_ep))

def lr_schedule(epoch, max_ep, lr_max):
    return LR_MIN + 0.5 * (lr_max - LR_MIN) * (1 + math.cos(math.pi * epoch / max_ep))

def base_model(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model

def load_model_state_compat(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        try:
            base_model(model).load_state_dict(state_dict)
        except RuntimeError as e2:
            raise RuntimeError(f"Failed to load checkpoint: {e2}") from e2

def amp_ctx():
    if DEV.type == 'cuda' and USE_AMP:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

EXP_DIR = Path(__file__).parent
TMP     = EXP_DIR / '.ckpt.pt'
BEST_PT = EXP_DIR / 'best_model.pt'


# ══════════════════════════════════════════════════════════════
#  Model — Encoder-Decoder Transformer with separate critic net
# ══════════════════════════════════════════════════════════════

class EncDecPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: plan features -> d_model
        self.plan_proj = nn.Linear(4, D_MODEL)
        self.plan_pos = nn.Embedding(PLAN_LEN, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FFN,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_ENC)

        # Decoder: history features -> d_model (no token embedding needed)
        self.hist_proj = nn.Linear(6, D_MODEL)
        self.hist_pos = nn.Embedding(HIST_LEN, D_MODEL)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FFN,
            dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=N_DEC)

        # Causal mask for decoder self-attention
        self.register_buffer('_causal_mask',
            nn.Transformer.generate_square_subsequent_mask(HIST_LEN), persistent=False)

        # Output heads
        self.actor_head = nn.Linear(D_MODEL, 2)
        # Separate critic consumes transformer features, but does not backprop to trunk.
        self.critic = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic[-1].bias)

    def encode_plan(self, plan_feat):
        pos = torch.arange(PLAN_LEN, device=plan_feat.device)
        x = self.plan_proj(plan_feat) + self.plan_pos(pos)
        return self.encoder(x)

    def decode_hist(self, hist_feat, memory):
        pos = torch.arange(HIST_LEN, device=hist_feat.device)
        x = self.hist_proj(hist_feat) + self.hist_pos(pos)
        return self.decoder(x, memory, tgt_mask=self._causal_mask, tgt_is_causal=True)

    def forward_last(self, plan_feat, hist_feat):
        memory = self.encode_plan(plan_feat)
        decoded = self.decode_hist(hist_feat, memory)
        last = decoded[:, -1, :]
        out = self.actor_head(last)
        a = F.softplus(out[:, 0]) + 1.0
        b = F.softplus(out[:, 1]) + 1.0
        val = self.critic(last.detach()).squeeze(-1)
        return a, b, val


# ══════════════════════════════════════════════════════════════
#  Input construction (GPU, batched)
# ══════════════════════════════════════════════════════════════

def build_plan(dg, step_idx, T, N):
    plan = torch.zeros((N, PLAN_LEN, 4), dtype=torch.float32, device='cuda')
    end = min(step_idx + PLAN_LEN, T)
    w = end - step_idx - 1
    if w <= 0:
        last = step_idx if step_idx < T else T - 1
        plan[:, :, 0] = (dg['target_lataccel'][:, last] / S_LAT).float().unsqueeze(1)
        plan[:, :, 1] = (dg['roll_lataccel'][:, last] / S_ROLL).float().unsqueeze(1)
        plan[:, :, 2] = (dg['v_ego'][:, last] / S_VEGO).float().unsqueeze(1)
        plan[:, :, 3] = (dg['a_ego'][:, last] / S_AEGO).float().unsqueeze(1)
    else:
        s = step_idx + 1
        plan[:, :w, 0] = dg['target_lataccel'][:, s:end].float() / S_LAT
        plan[:, :w, 1] = dg['roll_lataccel'][:, s:end].float() / S_ROLL
        plan[:, :w, 2] = dg['v_ego'][:, s:end].float() / S_VEGO
        plan[:, :w, 3] = dg['a_ego'][:, s:end].float() / S_AEGO
        if w < PLAN_LEN:
            plan[:, w:, :] = plan[:, w-1:w, :]
    return plan


def build_hist_feat(dg, step_idx, cur_lat_hist, steer_hist, N):
    feat = torch.zeros((N, HIST_LEN, 6), dtype=torch.float32, device='cuda')
    for j in range(HIST_LEN):
        t = step_idx - HIST_LEN + 1 + j
        t = max(t, 0)
        feat[:, j, 0] = dg['target_lataccel'][:, t].float() / S_LAT
        feat[:, j, 1] = cur_lat_hist[:, j].float() / S_LAT
        feat[:, j, 2] = dg['roll_lataccel'][:, t].float() / S_ROLL
        feat[:, j, 3] = dg['v_ego'][:, t].float() / S_VEGO
        feat[:, j, 4] = dg['a_ego'][:, t].float() / S_AEGO
        feat[:, j, 5] = steer_hist[:, j].float() / S_STEER
    return feat


# ══════════════════════════════════════════════════════════════
#  GPU Rollout
# ══════════════════════════════════════════════════════════════

def rollout(csv_files, model, mdl_path, ort_session, csv_cache,
            deterministic=False, ds=DELTA_SCALE_MAX):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_session,
                           cached_data=data, cached_rng=rng)
    N, T = sim.N, sim.T
    dg = sim.data_gpu
    max_steps = COST_END_IDX - CONTROL_START_IDX

    steer_hist   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')
    cur_lat_hist = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')

    if not deterministic:
        all_plan  = torch.empty((max_steps, N, PLAN_LEN, 4), dtype=torch.float32, device='cuda')
        all_hfeat = torch.empty((max_steps, N, HIST_LEN, 6), dtype=torch.float32, device='cuda')
        all_raw   = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        all_lp    = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        all_val   = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
        tgt_hist  = torch.empty((max_steps, N), dtype=torch.float64, device='cuda')
        cur_hist  = torch.empty((max_steps, N), dtype=torch.float64, device='cuda')
        act_f_hist = torch.empty((max_steps, N), dtype=torch.float64, device='cuda')
    si = 0

    def ctrl(step_idx, sim_ref):
        nonlocal si
        current = sim_ref.current_lataccel

        if step_idx < CONTROL_START_IDX:
            steer_hist[:, :-1] = steer_hist[:, 1:].clone()
            steer_hist[:, -1] = 0.0
            cur_lat_hist[:, :-1] = cur_lat_hist[:, 1:].clone()
            cur_lat_hist[:, -1] = current
            return torch.zeros(N, dtype=torch.float64, device='cuda')

        plan_feat = build_plan(dg, step_idx, T, N)
        hist_feat = build_hist_feat(dg, step_idx, cur_lat_hist, steer_hist, N)

        with torch.inference_mode():
            with amp_ctx():
                a_p, b_p, val = model.forward_last(plan_feat, hist_feat)
            a_p = a_p.float(); b_p = b_p.float(); val = val.float()

        if deterministic:
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
            lp = None
        else:
            dist = torch.distributions.Beta(a_p, b_p)
            x = dist.sample()
            raw = 2.0 * x - 1.0
            lp = dist.log_prob(x).float()

        delta  = (raw.double() * ds).clamp(-MAX_DELTA, MAX_DELTA)
        action = (steer_hist[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        steer_hist[:, :-1] = steer_hist[:, 1:].clone()
        steer_hist[:, -1] = action
        cur_lat_hist[:, :-1] = cur_lat_hist[:, 1:].clone()
        cur_lat_hist[:, -1] = current

        if not deterministic and step_idx < COST_END_IDX:
            all_plan[si] = plan_feat
            all_hfeat[si] = hist_feat
            all_raw[si] = raw
            all_lp[si] = lp
            all_val[si] = val
            tgt_hist[si] = dg['target_lataccel'][:, step_idx]
            cur_hist[si] = current
            act_f_hist[si] = action
            si += 1

        return action

    costs = sim.rollout(ctrl)['total_cost']

    if deterministic:
        return costs.tolist()

    S = si
    tgt = tgt_hist[:S].T; cur = cur_hist[:S].T; act = act_f_hist[:S].T
    lat_r = (tgt - cur)**2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(cur, dim=1, prepend=cur[:, :1]) / DEL_T
    act_d = torch.diff(act, dim=1, prepend=act[:, :1]) / DEL_T
    rew = (-(lat_r + jerk**2 * 100 + act_d**2 * ACT_SMOOTH) / 500.0).float()
    dones = torch.zeros((N, S), dtype=torch.float32, device='cuda')
    dones[:, -1] = 1.0

    return dict(
        plan=all_plan[:S].permute(1, 0, 2, 3).reshape(-1, PLAN_LEN, 4),
        hfeat=all_hfeat[:S].permute(1, 0, 2, 3).reshape(-1, HIST_LEN, 6),
        raw=all_raw[:S].T.reshape(-1),
        lp=all_lp[:S].T.reshape(-1),
        val_2d=all_val[:S].T,
        rew=rew, done=dones, costs=costs)


# ══════════════════════════════════════════════════════════════
#  BC Pretrain — Beta NLL on continuous delta targets
# ══════════════════════════════════════════════════════════════

def _bc_worker(csv_path):
    df = pd.read_csv(csv_path)
    roll_la = np.sin(df['roll'].values) * ACC_G
    v_ego   = df['vEgo'].values
    a_ego   = df['aEgo'].values
    tgt     = df['targetLateralAcceleration'].values
    steer   = -df['steerCommand'].values
    T       = len(df)

    plans, hfeats, raw_targets = [], [], []
    steer_hist   = np.zeros(HIST_LEN, dtype=np.float32)
    cur_lat_hist = np.zeros(HIST_LEN, dtype=np.float32)

    for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
        plan = np.zeros((PLAN_LEN, 4), dtype=np.float32)
        end = min(step_idx + PLAN_LEN, T)
        w = end - step_idx - 1
        if w > 0:
            s = step_idx + 1
            plan[:w, 0] = tgt[s:end] / S_LAT
            plan[:w, 1] = roll_la[s:end] / S_ROLL
            plan[:w, 2] = v_ego[s:end] / S_VEGO
            plan[:w, 3] = a_ego[s:end] / S_AEGO
            if w < PLAN_LEN:
                plan[w:] = plan[w-1:w]
        else:
            plan[:, 0] = tgt[min(step_idx, T-1)] / S_LAT
            plan[:, 1] = roll_la[min(step_idx, T-1)] / S_ROLL
            plan[:, 2] = v_ego[min(step_idx, T-1)] / S_VEGO
            plan[:, 3] = a_ego[min(step_idx, T-1)] / S_AEGO

        hfeat = np.zeros((HIST_LEN, 6), dtype=np.float32)
        for j in range(HIST_LEN):
            t = step_idx - HIST_LEN + 1 + j
            t = max(t, 0)
            hfeat[j, 0] = tgt[t] / S_LAT
            hfeat[j, 1] = cur_lat_hist[j] / S_LAT
            hfeat[j, 2] = roll_la[t] / S_ROLL
            hfeat[j, 3] = v_ego[t] / S_VEGO
            hfeat[j, 4] = a_ego[t] / S_AEGO
            hfeat[j, 5] = steer_hist[j] / S_STEER

        raw_target = np.clip((steer[step_idx] - steer_hist[-1]) / DELTA_SCALE_MAX, -1.0, 1.0)

        plans.append(plan)
        hfeats.append(hfeat)
        raw_targets.append(raw_target)

        steer_hist[:-1] = steer_hist[1:]
        steer_hist[-1] = steer[step_idx]
        cur_lat_hist[:-1] = cur_lat_hist[1:]
        cur_lat_hist[-1] = tgt[step_idx]

    return (np.array(plans), np.array(hfeats), np.array(raw_targets, dtype=np.float32))


def pretrain_bc(model, all_csvs):
    print(f"BC pretrain: extracting from {len(all_csvs)} CSVs ...")
    results = process_map(_bc_worker, [str(f) for f in all_csvs],
                          max_workers=10, chunksize=50, disable=False)
    all_plan = torch.from_numpy(np.concatenate([r[0] for r in results])).to(DEV)
    all_hfeat = torch.from_numpy(np.concatenate([r[1] for r in results])).to(DEV)
    all_raw = torch.from_numpy(np.concatenate([r[2] for r in results])).to(DEV)
    N = len(all_plan)
    print(f"BC pretrain: {N} samples, {BC_EPOCHS} epochs")

    actor_params = [p for n, p in base_model(model).named_parameters()
                    if not n.startswith('critic')]
    opt = optim.AdamW(actor_params, lr=BC_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    for ep in range(BC_EPOCHS):
        total, nb = 0.0, 0
        for idx in torch.randperm(N).split(BC_BS):
            with amp_ctx():
                a_p, b_p, _ = model.forward_last(all_plan[idx], all_hfeat[idx])
            a_p = a_p.float(); b_p = b_p.float()
            x = ((all_raw[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor_params, BC_GRAD_CLIP)
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
    def __init__(self, model):
        self.model = model
        self.actor_params, self.critic_params = [], []
        for name, p in base_model(model).named_parameters():
            if name.startswith('critic'):
                self.critic_params.append(p)
            else:
                self.actor_params.append(p)
        self.pi_opt = optim.Adam(self.actor_params, lr=PI_LR, eps=1e-5)
        self.vf_opt = optim.Adam(self.critic_params, lr=VF_LR, eps=1e-5)
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
        plan = gd['plan']
        hfeat = gd['hfeat']
        raw = gd['raw'].unsqueeze(-1)
        old_lp = gd['lp']
        adv_t, ret_t = self._gae(gd['rew'], gd['val_2d'], gd['done'])
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        x_t = ((raw + 1) / 2).clamp(1e-6, 1 - 1e-6)
        old_val = gd['val_2d'].reshape(-1)

        for _ in range(K_EPOCHS):
            for idx in torch.randperm(len(plan), device='cuda').split(MINI_BS):
                with amp_ctx():
                    a_c, b_c, val = self.model.forward_last(plan[idx], hfeat[idx])
                a_c = a_c.float(); b_c = b_c.float(); val = val.float()

                vc = old_val[idx] + (val - old_val[idx]).clamp(-10, 10)
                vf_loss = torch.max(
                    F.huber_loss(val, ret_t[idx], delta=10.0, reduction='none'),
                    F.huber_loss(vc,  ret_t[idx], delta=10.0, reduction='none')).mean()

                if critic_only:
                    self.vf_opt.zero_grad(set_to_none=True)
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic_params, 0.5)
                    self.vf_opt.step()
                else:
                    dist = torch.distributions.Beta(a_c, b_c)
                    lp = dist.log_prob(x_t[idx].squeeze(-1))
                    ratio = (lp - old_lp[idx]).exp()
                    mb_adv = adv_t[idx]
                    pi_loss = -torch.min(
                        ratio * mb_adv,
                        ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv).mean()
                    ent = dist.entropy().mean()
                    loss = pi_loss + VF_COEF * vf_loss - ENT_COEF * ent
                    self.pi_opt.zero_grad(set_to_none=True)
                    self.vf_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_params, 0.5)
                    nn.utils.clip_grad_norm_(self.critic_params, 0.5)
                    self.pi_opt.step()
                    self.vf_opt.step()

        with torch.no_grad():
            with amp_ctx():
                a_d, b_d, _ = self.model.forward_last(plan[:1000], hfeat[:1000])
            a_d = a_d.float(); b_d = b_d.float()
            sigma = (2.0 * torch.sqrt(a_d*b_d / ((a_d+b_d)**2 * (a_d+b_d+1)))).mean().item()
        return dict(
            pi=pi_loss.item() if not critic_only else 0.0,
            vf=vf_loss.item(),
            ent=ent.item() if not critic_only else 0.0,
            sigma=sigma,
            lr=self.pi_opt.param_groups[0]['lr'])


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════

def evaluate(model, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE_MAX):
    costs = rollout(files, model, mdl_path, ort_session, csv_cache, deterministic=True, ds=ds)
    return float(np.mean(costs)), float(np.std(costs))


def train():
    model = EncDecPolicy().to(DEV)
    if USE_COMPILE:
        model = torch.compile(model)
    ppo = PPO(model)
    mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
    ort_sess = make_ort_session(mdl_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    all_csv = sorted((ROOT / 'data').glob('*.csv'))
    va_f = all_csv[:EVAL_N]
    tr_f = all_csv[EVAL_N:]
    random.seed(42); random.shuffle(tr_f)
    csv_cache = CSVCache(sorted(set(str(f) for f in tr_f + va_f)))

    warmup_off = 0
    resumed = False
    resumed_optim = False
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        if ckpt.get('action_mode') == ACTION_MODE:
            load_model_state_compat(model, ckpt['model'])
            resumed = True
            if RESUME_OPT and 'pi_opt' in ckpt and 'vf_opt' in ckpt:
                ppo.pi_opt.load_state_dict(ckpt['pi_opt'])
                ppo.vf_opt.load_state_dict(ckpt['vf_opt'])
                for pg in ppo.pi_opt.param_groups: pg['lr'] = PI_LR; pg['eps'] = 1e-5
                for pg in ppo.vf_opt.param_groups: pg['lr'] = VF_LR; pg['eps'] = 1e-5
                if 'ret_rms' in ckpt:
                    r = ckpt['ret_rms']
                    ppo._rms.mean, ppo._rms.var, ppo._rms.count = r['mean'], r['var'], r['count']
                resumed_optim = True
                warmup_off = CRITIC_WARMUP
            print(f"Resumed from {BEST_PT.name}")
            if not resumed_optim:
                print("Loaded model weights only; keeping critic warmup.")
        else:
            print(f"Checkpoint mode mismatch; running BC pretrain.")
    if not resumed:
        all_csvs = sorted((ROOT / 'data').glob('*.csv'))
        pretrain_bc(model, all_csvs)

    vm, vs = evaluate(model, va_f, mdl_path, ort_sess, csv_cache)
    best, best_ep = vm, 'init'
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")

    def save_best():
        torch.save({
            'action_mode': ACTION_MODE,
            'model': base_model(model).state_dict(),
            'pi_opt': ppo.pi_opt.state_dict(),
            'vf_opt': ppo.vf_opt.state_dict(),
            'ret_rms': {'mean': ppo._rms.mean, 'var': ppo._rms.var, 'count': ppo._rms.count},
        }, BEST_PT)
    save_best()

    print(f"\nPPO  csvs={CSVS_EPOCH}  epochs={MAX_EP}  dev={DEV}")
    print(f"  π_lr={PI_LR}  vf_lr={VF_LR}  ent={ENT_COEF}  act_smooth={ACT_SMOOTH}"
          f"  Δscale={'decay' if DELTA_SCALE_DECAY else 'fixed'} {DELTA_SCALE_MAX}→{DELTA_SCALE_MIN}"
          f"  K={K_EPOCHS}  d={D_MODEL}  heads={N_HEADS}  enc={N_ENC}  dec={N_DEC}"
          f"  compile={int(USE_COMPILE)}  math_sdp={int(FORCE_MATH_SDP)}\n")

    for epoch in range(MAX_EP):
        cur_lr = lr_schedule(epoch, MAX_EP, PI_LR)
        for pg in ppo.pi_opt.param_groups: pg['lr'] = cur_lr
        for pg in ppo.vf_opt.param_groups: pg['lr'] = cur_lr
        cur_ds = delta_scale(epoch, MAX_EP) if DELTA_SCALE_DECAY else DELTA_SCALE_MAX
        t0 = time.time()
        batch = random.sample(tr_f, min(CSVS_EPOCH, len(tr_f)))
        res = rollout(batch, model, mdl_path, ort_sess, csv_cache,
                      deterministic=False, ds=cur_ds)

        t1 = time.time()
        co = epoch < (CRITIC_WARMUP - warmup_off)
        info = ppo.update(res, critic_only=co)
        tu = time.time() - t1

        phase = "  [critic warmup]" if co else ""
        line = (f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}"
                f"  σ={info['sigma']:.4f}  π={info['pi']:+.4f}  vf={info['vf']:.1f}"
                f"  H={info['ent']:.2f}  lr={info['lr']:.1e}"
                f"  Δs={cur_ds:.2f}  ⏱{t1-t0:.0f}+{tu:.0f}s{phase}")

        if epoch % EVAL_EVERY == 0:
            vm, vs = evaluate(model, va_f, mdl_path, ort_sess, csv_cache, ds=cur_ds)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:6.1f}±{vs:4.1f}{mk}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")
    torch.save({'action_mode': ACTION_MODE, 'model': base_model(model).state_dict()},
               EXP_DIR / 'final_model.pt')
    if TMP.exists(): TMP.unlink()


if __name__ == '__main__':
    train()
