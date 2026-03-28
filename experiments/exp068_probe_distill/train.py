import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    STEER_RANGE,
    DEL_T,
)
from tinyphysics_batched import CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
MAX_DELTA = 0.5
MAX_RESIDUAL = float(os.getenv("MAX_RESIDUAL", "0.35"))
REF_INPUT = os.getenv("REF_INPUT", "1") == "1"

LR = float(os.getenv("LR", "3e-4"))
WD = float(os.getenv("WD", "1e-6"))
BS = int(os.getenv("BS", "256"))
EPOCHS = int(os.getenv("EPOCHS", "400"))
HOLDOUT_FRAC = float(os.getenv("HOLDOUT_FRAC", "0.2"))
ZERO_COEF = float(os.getenv("ZERO_COEF", "0.01"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "10"))
VAL_LIMIT = int(os.getenv("VAL_LIMIT", "64"))

WIN_BANK = Path(
    os.getenv(
        "WIN_BANK",
        str(ROOT / "experiments" / "exp067_reference_probe" / "probe_wins.pt"),
    )
)
EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
OBS_DIM = F_A + FUTURE_K


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*layers)

    def raw(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        return 2.0 * a_p / (a_p + b_p) - 1.0


class ResidualChunkActor(nn.Module):
    def __init__(self, chunk_k):
        super().__init__()
        self.chunk_k = chunk_k
        in_dim = STATE_DIM + (chunk_k if REF_INPUT else 0)
        layers = [nn.Linear(in_dim, HIDDEN), nn.ReLU()]
        for _ in range(2):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, chunk_k))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs, base_raw):
        x = obs
        if REF_INPUT:
            x = torch.cat([obs, base_raw.unsqueeze(-1).expand(-1, self.chunk_k)], dim=-1)
        return torch.tanh(self.net(x)) * MAX_RESIDUAL


class DistillModel(nn.Module):
    def __init__(self, chunk_k):
        super().__init__()
        self.chunk_k = chunk_k
        self.base_actor = BaseActor()
        self.residual_actor = ResidualChunkActor(chunk_k)

    def base_raw(self, obs):
        return self.base_actor.raw(obs)

    def residual_chunk(self, obs):
        base_raw = self.base_raw(obs)
        return self.residual_actor(obs, base_raw)


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


def build_obs_from_hist(sim_ref, future, step_idx):
    dg = sim_ref.data_gpu
    target = dg["target_lataccel"][:, step_idx].float()
    current = sim_ref.current_lataccel.float()
    roll_la = dg["roll_lataccel"][:, step_idx].float()
    v_ego = dg["v_ego"][:, step_idx].float()
    a_ego = dg["a_ego"][:, step_idx].float()

    h_act = sim_ref.action_history[:, step_idx - HIST_LEN : step_idx].float()
    h_lat = sim_ref.current_lataccel_history[:, step_idx - HIST_LEN : step_idx].float()
    h_tgt = dg["target_lataccel"][:, step_idx - HIST_LEN : step_idx].float()

    error_integral = (h_tgt - h_lat).mean(dim=1) * DEL_T
    prev_act = h_act[:, -1]
    prev_act2 = h_act[:, -2]
    prev_lat = h_lat[:, -1]

    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0

    obs = torch.empty((sim_ref.N, OBS_DIM), dtype=torch.float32, device="cuda")
    obs[:, 0] = target / S_LAT
    obs[:, 1] = current / S_LAT
    obs[:, 2] = (target - current) / S_LAT
    obs[:, 3] = k_tgt / S_CURV
    obs[:, 4] = k_cur / S_CURV
    obs[:, 5] = (k_tgt - k_cur) / S_CURV
    obs[:, 6] = v_ego / S_VEGO
    obs[:, 7] = a_ego / S_AEGO
    obs[:, 8] = roll_la / S_ROLL
    obs[:, 9] = prev_act / S_STEER
    obs[:, 10] = error_integral / S_LAT
    obs[:, 11] = (fp0 - target) / DEL_T / S_LAT
    obs[:, 12] = (current - prev_lat) / DEL_T / S_LAT
    obs[:, 13] = (prev_act - prev_act2) / DEL_T / S_STEER
    obs[:, 14] = fric
    obs[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    obs[:, C:H1] = h_act / S_STEER
    obs[:, H1:H2] = h_lat / S_LAT
    obs[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    obs[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    obs[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    obs[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    obs.clamp_(-5.0, 5.0)
    return obs


def rollout_cost(csv_files, model, mdl_path, ort_sess, csv_cache):
    from tinyphysics_batched import BatchedSimulator

    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    future = _precompute_future_windows(sim.data_gpu)
    sim.use_expected = True
    planned = None
    step_in_chunk = 0

    def ctrl(step_idx, sim_ref):
        nonlocal planned, step_in_chunk
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(sim_ref.N, dtype=torch.float64, device="cuda")
        if step_idx % model.chunk_k == 0 or planned is None:
            with torch.inference_mode():
                obs = build_obs_from_hist(sim_ref, future, step_idx)
                planned = model.residual_chunk(obs)
            step_in_chunk = 0

        with torch.inference_mode():
            obs = build_obs_from_hist(sim_ref, future, step_idx)
            base_raw = model.base_raw(obs)
        residual = planned[:, step_in_chunk]
        step_in_chunk = min(step_in_chunk + 1, model.chunk_k - 1)
        raw = (base_raw + residual).clamp(-1.0, 1.0)
        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)["total_cost"]
    return float(np.mean(costs)), float(np.std(costs))


def split_bank(bank):
    csv_files = bank["csv_file"]
    unique = sorted(set(csv_files))
    random.shuffle(unique)
    n_val = max(1, int(len(unique) * HOLDOUT_FRAC))
    val_csv = set(unique[:n_val])
    tr_idx = [i for i, f in enumerate(csv_files) if f not in val_csv]
    va_idx = [i for i, f in enumerate(csv_files) if f in val_csv]
    if not tr_idx:
        tr_idx = list(range(len(csv_files)))
    if not va_idx:
        va_idx = tr_idx[:]
    return tr_idx, va_idx, sorted(val_csv)[:VAL_LIMIT]


def batch_select(x, idx):
    if isinstance(x, list):
        return [x[i] for i in idx]
    return x[idx]


def train():
    if not WIN_BANK.exists():
        raise FileNotFoundError(f"Win bank not found: {WIN_BANK}")
    bank = torch.load(WIN_BANK, map_location="cpu", weights_only=False)
    chunk_k = int(bank["meta"]["chunk_k"])
    obs = bank["obs"].float()
    target = bank["best_resid_raw"].float().clamp(-MAX_RESIDUAL, MAX_RESIDUAL)
    improvement = bank["improvement"].float()
    csv_files = bank["csv_file"]

    tr_idx, va_idx, val_csvs = split_bank(bank)
    obs_tr = batch_select(obs, tr_idx).to(DEV)
    tgt_tr = batch_select(target, tr_idx).to(DEV)
    imp_tr = batch_select(improvement, tr_idx).to(DEV)
    obs_va = batch_select(obs, va_idx).to(DEV)
    tgt_va = batch_select(target, va_idx).to(DEV)
    imp_va = batch_select(improvement, va_idx).to(DEV)

    model = DistillModel(chunk_k).to(DEV)
    base_pt = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
    ckpt = torch.load(base_pt, map_location="cpu", weights_only=False)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    model.base_actor.load_state_dict(actor_state, strict=False)
    model.base_actor.eval()
    for p in model.base_actor.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(model.residual_actor.parameters(), lr=LR, weight_decay=WD)
    best_val = float("inf")

    print(
        f"exp068 probe distill  bank={WIN_BANK.name}  N={len(obs)}  train={len(tr_idx)}  val={len(va_idx)}"
    )
    print(
        f"chunk={chunk_k}  ref_input={'on' if REF_INPUT else 'off'}  max_residual={MAX_RESIDUAL:g}"
        f"  lr={LR:g}  bs={BS}  epochs={EPOCHS}  zero_coef={ZERO_COEF:g}"
    )

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    eval_cache = CSVCache(val_csvs if val_csvs else sorted(set(csv_files)))

    def weighted_loss(pred, tgt, imp):
        w = (imp / imp.mean().clamp_min(1e-6)).sqrt().clamp(0.25, 4.0)
        per = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)
        return (per * w).mean() + ZERO_COEF * pred.pow(2).mean()

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(obs_tr), device=DEV)
        model.train()
        train_sum = 0.0
        n_seen = 0
        for idx in perm.split(BS):
            pred = model.residual_chunk(obs_tr[idx])
            loss = weighted_loss(pred, tgt_tr[idx], imp_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.residual_actor.parameters(), 1.0)
            opt.step()
            train_sum += loss.detach().item() * idx.numel()
            n_seen += idx.numel()

        model.eval()
        with torch.no_grad():
            val_pred = model.residual_chunk(obs_va)
            val_loss = weighted_loss(val_pred, tgt_va, imp_va).item()
            mae = (val_pred - tgt_va).abs().mean().item()
            pred_mag = val_pred.abs().mean().item()

        line = (
            f"E{epoch:3d}  train={train_sum / max(1, n_seen):.5f}  val={val_loss:.5f}"
            f"  mae={mae:.4f}  |pred|={pred_mag:.4f}"
        )

        if epoch % EVAL_EVERY == 0 and val_csvs:
            vm, vs = rollout_cost(val_csvs, model, mdl_path, ort_sess, eval_cache)
            line += f"  rollout={vm:.2f}±{vs:.2f}"
            metric = vm
        else:
            metric = val_loss

        if metric < best_val:
            best_val = metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "chunk_k": chunk_k,
                    "delta_scale": DELTA_SCALE,
                    "max_residual": MAX_RESIDUAL,
                    "ref_input": REF_INPUT,
                    "win_bank": str(WIN_BANK),
                },
                BEST_PT,
            )
            line += " ★"
        print(line)

    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
