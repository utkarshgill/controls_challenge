import csv
import os
import random
import sys
import time
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
    LAT_ACCEL_COST_MULTIPLIER,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── controller / obs (must match exp055) ─────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
MAX_DELTA = 0.5

C = 16
H1 = C + HIST_LEN
H2 = H1 + HIST_LEN
F_LAT = H2
F_ROLL = F_LAT + FUTURE_K
F_V = F_ROLL + FUTURE_K
F_A = F_V + FUTURE_K
OBS_DIM = F_A + FUTURE_K

# ── local-search bank generation ──────────────────────────────────────────────
CHUNK_K = int(os.getenv("CHUNK_K", "5"))
HORIZON_H = int(os.getenv("HORIZON_H", "20"))
ROUTES = int(os.getenv("ROUTES", "128"))
VAL_ROUTES = int(os.getenv("VAL_ROUTES", "32"))
TOPK_PER_ROUTE = int(os.getenv("TOPK_PER_ROUTE", "2"))
RAND_PER_ROUTE = int(os.getenv("RAND_PER_ROUTE", "2"))
PROBE_M = int(os.getenv("PROBE_M", "32"))
NOISE_STD_SMALL = float(os.getenv("NOISE_STD_SMALL", "0.03"))
NOISE_STD_LARGE = float(os.getenv("NOISE_STD_LARGE", "0.08"))
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", str(max(2 * CHUNK_K, 8))))
USE_EXPECTED = os.getenv("USE_EXPECTED", "1") == "1"
IMPROVE_MARGIN = float(os.getenv("IMPROVE_MARGIN", "10.0"))
CSV_MODE = os.getenv("CSV_MODE", "sample")

# ── student training ──────────────────────────────────────────────────────────
MAX_RESIDUAL = float(os.getenv("MAX_RESIDUAL", "0.30"))
REF_INPUT = os.getenv("REF_INPUT", "1") == "1"
HARD_GATE = os.getenv("HARD_GATE", "1") == "1"
GATE_THRESH = float(os.getenv("GATE_THRESH", "0.55"))
LR = float(os.getenv("LR", "3e-4"))
WD = float(os.getenv("WD", "1e-6"))
BS = int(os.getenv("BS", "256"))
EPOCHS = int(os.getenv("EPOCHS", "300"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "10"))
REG_COEF = float(os.getenv("REG_COEF", "1.0"))
GATE_COEF = float(os.getenv("GATE_COEF", "0.5"))
POS_RAW_COEF = float(os.getenv("POS_RAW_COEF", "0.2"))
ZERO_RAW_COEF = float(os.getenv("ZERO_RAW_COEF", "0.02"))

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BANK_PT = EXP_DIR / "bank.pt"
BANK_CSV = EXP_DIR / "bank.csv"
SUMMARY_TXT = EXP_DIR / "summary.txt"
BASE_PT = Path(
    os.getenv(
        "BASE_MODEL",
        str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"),
    )
)


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


def _chunk_criticality(future_target, current):
    horizon = min(CRIT_LOOKAHEAD, future_target.shape[1])
    fp = future_target[:, :horizon]
    if horizon <= 1:
        return (fp[:, 0] - current).abs() / S_LAT
    slope = torch.diff(fp, dim=1, prepend=fp[:, :1]) / DEL_T
    mismatch = (fp[:, min(CHUNK_K - 1, horizon - 1)] - current).abs() / S_LAT
    span = (fp[:, -1] - fp[:, 0]).abs() / S_LAT
    peak_slope = slope.abs().amax(dim=1) / max(S_LAT / DEL_T, 1e-6)
    flip = ((fp[:, :-1] * fp[:, 1:]) < 0).any(dim=1).float()
    return mismatch + 0.5 * span + 0.5 * peak_slope + 0.5 * flip


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


class ResidualGateStudent(nn.Module):
    def __init__(self, chunk_k):
        super().__init__()
        self.chunk_k = chunk_k
        self.base_actor = BaseActor()
        in_dim = STATE_DIM + 1 + (chunk_k if REF_INPUT else 0)

        res_layers = [nn.Linear(in_dim, HIDDEN), nn.ReLU()]
        for _ in range(2):
            res_layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        res_layers.append(nn.Linear(HIDDEN, chunk_k))
        self.residual_head = nn.Sequential(*res_layers)

        gate_layers = [nn.Linear(in_dim, HIDDEN // 2), nn.ReLU(), nn.Linear(HIDDEN // 2, 1)]
        self.gate_head = nn.Sequential(*gate_layers)

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def base_raw(self, obs):
        return self.base_actor.raw(obs)

    def _input(self, obs, base_raw, criticality):
        crit = criticality.unsqueeze(-1)
        x = torch.cat([obs, crit], dim=-1)
        if REF_INPUT:
            x = torch.cat([x, base_raw.unsqueeze(-1).expand(-1, self.chunk_k)], dim=-1)
        return x

    def residual_and_gate(self, obs, criticality):
        base_raw = self.base_raw(obs)
        x = self._input(obs, base_raw, criticality)
        resid = torch.tanh(self.residual_head(x)) * MAX_RESIDUAL
        gate_logit = self.gate_head(x).squeeze(-1)
        return resid, gate_logit, base_raw

    def runtime_chunk(self, obs, criticality):
        resid, gate_logit, base_raw = self.residual_and_gate(obs, criticality)
        gate_prob = torch.sigmoid(gate_logit)
        if HARD_GATE:
            gated = resid * (gate_prob >= GATE_THRESH).float().unsqueeze(-1)
        else:
            gated = resid * gate_prob.unsqueeze(-1)
        return gated, gate_prob, base_raw


def load_base_actor():
    if not BASE_PT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_PT}")
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    actor = BaseActor().to(DEV)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=False)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


def build_obs_from_sim(sim_ref, future, step_idx):
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


def sample_candidate_chunks(base_chunk, num_cand):
    chunks = base_chunk.unsqueeze(0).repeat(num_cand, 1)
    if num_cand <= 1:
        return chunks
    n = num_cand - 1
    u = torch.linspace(-1.0, 1.0, CHUNK_K, device=base_chunk.device)
    basis = torch.stack(
        [torch.ones_like(u), u, u.square() - u.square().mean()],
        dim=0,
    )
    coeff = torch.randn(n, basis.shape[0], device=base_chunk.device)
    resid = coeff @ basis
    resid = resid / resid.std(dim=1, keepdim=True).clamp_min(1e-6)
    sigma = torch.empty(n, 1, device=base_chunk.device)
    half = n // 2
    sigma[:half] = NOISE_STD_SMALL
    sigma[half:] = NOISE_STD_LARGE
    if n > 1:
        sigma = sigma[torch.randperm(n, device=base_chunk.device)]
    resid = resid * sigma + 0.2 * sigma * torch.randn(n, CHUNK_K, device=base_chunk.device)
    chunks[1:] = (base_chunk.unsqueeze(0) + resid).clamp(-1.0, 1.0)
    return chunks


def choose_routes(all_csv):
    total = min(ROUTES + VAL_ROUTES, len(all_csv))
    if CSV_MODE == "first":
        picked = list(all_csv[:total])
    else:
        picked = random.sample(list(all_csv), total)
    random.shuffle(picked)
    n_val = min(VAL_ROUTES, max(1, total // 5))
    val_routes = picked[:n_val]
    train_routes = picked[n_val:]
    if not train_routes:
        train_routes = val_routes
    return train_routes, val_routes


def run_base_rollout(csv_files, base_actor, mdl_path, ort_sess, csv_cache):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    if USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(sim_ref.N, dtype=torch.float64, device="cuda")
        with torch.inference_mode():
            obs = build_obs_from_sim(sim_ref, future, step_idx)
            raw = base_actor.raw(obs)
        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)
    return sim, future, costs


def select_windows(base_sim, future, csv_files):
    boundaries = list(range(CONTROL_START_IDX, COST_END_IDX - CHUNK_K + 1, CHUNK_K))
    b_idx = torch.tensor(boundaries, device=base_sim.action_history.device, dtype=torch.long)
    rows = []
    for route_idx, csv_file in enumerate(csv_files):
        current_before = base_sim.current_lataccel_history[route_idx, b_idx - 1].float()
        crit = _chunk_criticality(future["target_lataccel"][route_idx, b_idx].float(), current_before)
        topk = min(TOPK_PER_ROUTE, len(boundaries))
        _, top_idx = torch.topk(crit, k=topk, largest=True, sorted=True)
        chosen = set(int(i) for i in top_idx.tolist())
        if RAND_PER_ROUTE > 0:
            remaining = [i for i in range(len(boundaries)) if i not in chosen]
            if remaining:
                rand_pick = random.sample(remaining, min(RAND_PER_ROUTE, len(remaining)))
                chosen.update(rand_pick)

        for local_idx in sorted(chosen):
            step_idx = boundaries[local_idx]
            prev = base_sim.action_history[route_idx, step_idx - 1 : step_idx - 1 + CHUNK_K]
            nxt = base_sim.action_history[route_idx, step_idx : step_idx + CHUNK_K]
            base_chunk_raw = ((nxt - prev) / DELTA_SCALE).float().clamp(-1.0, 1.0)
            rows.append(
                {
                    "csv_file": str(csv_file),
                    "route_idx": route_idx,
                    "step_idx": int(step_idx),
                    "chunk_idx": (step_idx - CONTROL_START_IDX) // CHUNK_K,
                    "criticality": float(crit[local_idx].item()),
                    "base_chunk_raw": base_chunk_raw,
                }
            )
    rows.sort(key=lambda r: (r["route_idx"], r["step_idx"]))
    return rows


def evaluate_window(csv_file, csv_cache, base_actor, mdl_path, ort_sess, base_sim, route_idx, step_idx, base_chunk_raw):
    tiled = [csv_file for _ in range(PROBE_M)]
    data, rng = csv_cache.slice(tiled)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    if USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    h = step_idx
    sim._hist_len = h
    sim.action_history[:, :h] = base_sim.action_history[route_idx : route_idx + 1, :h].repeat(PROBE_M, 1)
    sim.current_lataccel_history[:, :h] = base_sim.current_lataccel_history[route_idx : route_idx + 1, :h].repeat(PROBE_M, 1)
    sim.current_lataccel = sim.current_lataccel_history[:, h - 1].clone()
    sim.state_history[:, :h, 0] = sim.data_gpu["roll_lataccel"][:, :h]
    sim.state_history[:, :h, 1] = sim.data_gpu["v_ego"][:, :h]
    sim.state_history[:, :h, 2] = sim.data_gpu["a_ego"][:, :h]

    with torch.inference_mode():
        obs0 = build_obs_from_sim(sim, future, step_idx)[0].detach().clone()

    cand_raw = sample_candidate_chunks(base_chunk_raw, PROBE_M)
    costs = torch.zeros(PROBE_M, dtype=torch.float32, device="cuda")
    end_step = min(step_idx + HORIZON_H, COST_END_IDX)

    for t in range(step_idx, end_step):
        if t < step_idx + CHUNK_K:
            raw = cand_raw[:, t - step_idx]
        else:
            with torch.inference_mode():
                obs = build_obs_from_sim(sim, future, t)
                raw = base_actor.raw(obs)

        prev_action = sim.action_history[:, sim._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        actions = (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        sim.step(t, actions)

        cur = sim.current_lataccel.float()
        tgt = sim.data_gpu["target_lataccel"][:, t].float()
        prev_pred = sim.current_lataccel_history[:, t - 1].float()
        jerk = (cur - prev_pred) / DEL_T
        costs += (tgt - cur).square() * (100 * LAT_ACCEL_COST_MULTIPLIER) + jerk.square() * 100

    best_idx = int(costs.argmin().item())
    base_cost = float(costs[0].item())
    best_cost = float(costs[best_idx].item())
    improvement = base_cost - best_cost
    best_chunk = cand_raw[best_idx].detach().clone()
    target_chunk = best_chunk if improvement > IMPROVE_MARGIN else cand_raw[0].detach().clone()
    return {
        "obs": obs0,
        "base_chunk_raw": cand_raw[0].detach().clone(),
        "target_chunk_raw": target_chunk,
        "best_chunk_raw": best_chunk,
        "apply_label": float(improvement > IMPROVE_MARGIN),
        "base_cost": base_cost,
        "best_cost": best_cost,
        "improvement": improvement,
        "best_idx": best_idx,
    }


def build_bank(routes, split_name, base_actor, mdl_path, ort_sess, csv_cache):
    base_sim, future, costs = run_base_rollout(routes, base_actor, mdl_path, ort_sess, csv_cache)
    windows = select_windows(base_sim, future, routes)
    rows = []
    for i, win in enumerate(windows, start=1):
        out = evaluate_window(
            csv_file=win["csv_file"],
            csv_cache=csv_cache,
            base_actor=base_actor,
            mdl_path=mdl_path,
            ort_sess=ort_sess,
            base_sim=base_sim,
            route_idx=win["route_idx"],
            step_idx=win["step_idx"],
            base_chunk_raw=win["base_chunk_raw"].to(DEV),
        )
        row = {
            "split": split_name,
            "csv_file": win["csv_file"],
            "route_idx": win["route_idx"],
            "chunk_idx": win["chunk_idx"],
            "step_idx": win["step_idx"],
            "criticality": win["criticality"],
            "obs": out["obs"],
            "base_chunk_raw": out["base_chunk_raw"],
            "target_chunk_raw": out["target_chunk_raw"],
            "best_chunk_raw": out["best_chunk_raw"],
            "target_resid_raw": out["target_chunk_raw"] - out["base_chunk_raw"],
            "apply_label": out["apply_label"],
            "base_cost": out["base_cost"],
            "best_cost": out["best_cost"],
            "improvement": out["improvement"],
            "best_idx": out["best_idx"],
        }
        rows.append(row)
        print(
            f"[{split_name} {i:4d}/{len(windows):4d}] route={row['route_idx']:3d} step={row['step_idx']:3d}"
            f" crit={row['criticality']:.3f} apply={int(row['apply_label'])}"
            f" Δ={row['improvement']:+8.3f}"
        )
    return rows, np.asarray(costs["total_cost"], dtype=np.float64)


def save_bank(rows, train_base_costs, val_base_costs):
    payload = {
        "obs": torch.stack([r["obs"].cpu() for r in rows], dim=0),
        "base_chunk_raw": torch.stack([r["base_chunk_raw"].cpu() for r in rows], dim=0),
        "target_chunk_raw": torch.stack([r["target_chunk_raw"].cpu() for r in rows], dim=0),
        "target_resid_raw": torch.stack([r["target_resid_raw"].cpu() for r in rows], dim=0),
        "apply_label": torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32),
        "improvement": torch.tensor([r["improvement"] for r in rows], dtype=torch.float32),
        "criticality": torch.tensor([r["criticality"] for r in rows], dtype=torch.float32),
        "route_idx": torch.tensor([r["route_idx"] for r in rows], dtype=torch.long),
        "chunk_idx": torch.tensor([r["chunk_idx"] for r in rows], dtype=torch.long),
        "step_idx": torch.tensor([r["step_idx"] for r in rows], dtype=torch.long),
        "split": [r["split"] for r in rows],
        "csv_file": [r["csv_file"] for r in rows],
        "meta": {
            "chunk_k": CHUNK_K,
            "horizon_h": HORIZON_H,
            "probe_m": PROBE_M,
            "delta_scale": DELTA_SCALE,
            "improve_margin": IMPROVE_MARGIN,
            "train_base_mean": float(train_base_costs.mean()) if len(train_base_costs) else 0.0,
            "val_base_mean": float(val_base_costs.mean()) if len(val_base_costs) else 0.0,
        },
    }
    torch.save(payload, BANK_PT)

    with open(BANK_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "csv_file",
                "route_idx",
                "chunk_idx",
                "step_idx",
                "criticality",
                "apply_label",
                "base_cost",
                "best_cost",
                "improvement",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["split"],
                    r["csv_file"],
                    r["route_idx"],
                    r["chunk_idx"],
                    r["step_idx"],
                    r["criticality"],
                    r["apply_label"],
                    r["base_cost"],
                    r["best_cost"],
                    r["improvement"],
                ]
            )


def index_split(rows, split_name):
    return [i for i, r in enumerate(rows) if r["split"] == split_name]


def weighted_losses(model, obs, criticality, target_resid, apply_label, improvement):
    pred_raw, gate_logit, _ = model.residual_and_gate(obs, criticality)
    gate_prob = torch.sigmoid(gate_logit)
    pred = pred_raw * gate_prob.unsqueeze(-1)

    pos_mask = apply_label > 0.5
    imp_w = torch.ones_like(improvement)
    if pos_mask.any():
        pos_scale = 1.0 + (improvement / improvement[pos_mask].mean().clamp_min(1e-6)).clamp(0.0, 8.0)
        imp_w = torch.where(pos_mask, pos_scale, imp_w)
    reg_per = F.smooth_l1_loss(pred, target_resid, reduction="none").mean(dim=1)
    reg_loss = (reg_per * imp_w).mean()
    gate_loss = F.binary_cross_entropy_with_logits(gate_logit, apply_label)

    if pos_mask.any():
        pos_raw = F.smooth_l1_loss(pred_raw[pos_mask], target_resid[pos_mask], reduction="mean")
    else:
        pos_raw = pred_raw.sum() * 0.0
    zero_mask = ~pos_mask
    if zero_mask.any():
        zero_raw = pred_raw[zero_mask].pow(2).mean()
    else:
        zero_raw = pred_raw.sum() * 0.0

    loss = REG_COEF * reg_loss + GATE_COEF * gate_loss + POS_RAW_COEF * pos_raw + ZERO_RAW_COEF * zero_raw
    return loss, {
        "reg": reg_loss.detach().item(),
        "gate": gate_loss.detach().item(),
        "pos_raw": pos_raw.detach().item(),
        "zero_raw": zero_raw.detach().item(),
        "apply_rate": gate_prob.detach().mean().item(),
        "pred_mag": pred.detach().abs().mean().item(),
        "raw_mag": pred_raw.detach().abs().mean().item(),
    }


def rollout_cost(csv_files, model, mdl_path, ort_sess, csv_cache):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    if USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)
    planned = None
    step_in_chunk = 0

    def ctrl(step_idx, sim_ref):
        nonlocal planned, step_in_chunk
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(sim_ref.N, dtype=torch.float64, device="cuda")
        with torch.inference_mode():
            obs = build_obs_from_sim(sim_ref, future, step_idx)
            crit = _chunk_criticality(future["target_lataccel"][:, step_idx], sim_ref.current_lataccel.float())
            base_raw = model.base_raw(obs)
            if planned is None or step_idx % CHUNK_K == 0:
                planned, _, _ = model.runtime_chunk(obs, crit)
                step_in_chunk = 0
        residual = planned[:, step_in_chunk]
        step_in_chunk = min(step_in_chunk + 1, CHUNK_K - 1)
        raw = (base_raw + residual).clamp(-1.0, 1.0)
        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)["total_cost"]
    return float(np.mean(costs)), float(np.std(costs))


def write_summary(train_rows, val_rows, train_base_costs, val_base_costs):
    def summarize(rows):
        imp = np.asarray([r["improvement"] for r in rows], dtype=np.float64)
        apply = np.asarray([r["apply_label"] for r in rows], dtype=np.float64)
        return {
            "windows": len(rows),
            "apply_frac": float(apply.mean()) if len(apply) else 0.0,
            "improvement_mean": float(imp.mean()) if len(imp) else 0.0,
            "improvement_median": float(np.median(imp)) if len(imp) else 0.0,
        }

    ts = summarize(train_rows)
    vs = summarize(val_rows)
    lines = [
        "exp069 balanced probe distill",
        f"train_routes={len(train_base_costs)}  val_routes={len(val_base_costs)}",
        f"train_base_mean={train_base_costs.mean():.3f}  val_base_mean={val_base_costs.mean():.3f}",
        f"train_windows={ts['windows']}  train_apply_frac={ts['apply_frac']:.3f}"
        f"  train_imp_mean={ts['improvement_mean']:.3f}  train_imp_median={ts['improvement_median']:.3f}",
        f"val_windows={vs['windows']}  val_apply_frac={vs['apply_frac']:.3f}"
        f"  val_imp_mean={vs['improvement_mean']:.3f}  val_imp_median={vs['improvement_median']:.3f}",
    ]
    text = "\n".join(lines)
    SUMMARY_TXT.write_text(text + "\n")
    return text


def train():
    print(
        f"exp069 balanced probe distill  chunk={CHUNK_K}  horizon={HORIZON_H}  probe_m={PROBE_M}"
        f"  topk={TOPK_PER_ROUTE}  rand={RAND_PER_ROUTE}  improve_margin={IMPROVE_MARGIN:g}"
    )
    print(
        f"routes={ROUTES}  val_routes={VAL_ROUTES}  use_expected={'on' if USE_EXPECTED else 'off'}"
        f"  hard_gate={'on' if HARD_GATE else 'off'}@{GATE_THRESH:g}"
    )

    all_csv = sorted((ROOT / "data").glob("*.csv"))
    train_routes, val_routes = choose_routes(all_csv)
    selected = train_routes + [f for f in val_routes if f not in set(train_routes)]
    csv_cache = CSVCache(selected)
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    base_actor = load_base_actor()

    t0 = time.time()
    train_rows, train_base_costs = build_bank(train_routes, "train", base_actor, mdl_path, ort_sess, csv_cache)
    val_rows, val_base_costs = build_bank(val_routes, "val", base_actor, mdl_path, ort_sess, csv_cache)
    rows = train_rows + val_rows
    save_bank(rows, train_base_costs, val_base_costs)
    summary = write_summary(train_rows, val_rows, train_base_costs, val_base_costs)
    print(summary)
    print(f"bank built in {time.time() - t0:.1f}s")

    model = ResidualGateStudent(CHUNK_K).to(DEV)
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    model.base_actor.load_state_dict(actor_state, strict=False)
    model.base_actor.eval()
    for p in model.base_actor.parameters():
        p.requires_grad_(False)

    obs = torch.stack([r["obs"] for r in rows], dim=0).to(DEV)
    criticality = torch.tensor([r["criticality"] for r in rows], dtype=torch.float32, device=DEV)
    target_resid = torch.stack([r["target_resid_raw"] for r in rows], dim=0).to(DEV)
    apply_label = torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32, device=DEV)
    improvement = torch.tensor([max(r["improvement"], 0.0) for r in rows], dtype=torch.float32, device=DEV)
    tr_idx = torch.tensor(index_split(rows, "train"), dtype=torch.long, device=DEV)
    va_idx = torch.tensor(index_split(rows, "val"), dtype=torch.long, device=DEV)

    opt = optim.Adam(
        list(model.residual_head.parameters()) + list(model.gate_head.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    best_metric = float("inf")

    eval_cache = CSVCache(val_routes) if val_routes else None

    for epoch in range(EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=DEV)]
        acc = {"loss": 0.0, "reg": 0.0, "gate": 0.0, "pos_raw": 0.0, "zero_raw": 0.0}
        n_seen = 0
        for idx in perm.split(BS):
            loss, info = weighted_losses(
                model,
                obs[idx],
                criticality[idx],
                target_resid[idx],
                apply_label[idx],
                improvement[idx],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.residual_head.parameters()) + list(model.gate_head.parameters()),
                1.0,
            )
            opt.step()
            bs = int(idx.numel())
            acc["loss"] += loss.detach().item() * bs
            for k in ("reg", "gate", "pos_raw", "zero_raw"):
                acc[k] += info[k] * bs
            n_seen += bs

        model.eval()
        with torch.no_grad():
            vloss, vinfo = weighted_losses(
                model,
                obs[va_idx],
                criticality[va_idx],
                target_resid[va_idx],
                apply_label[va_idx],
                improvement[va_idx],
            )

        line = (
            f"E{epoch:3d}  train={acc['loss']/max(1,n_seen):.5f}  val={vloss.item():.5f}"
            f"  reg={vinfo['reg']:.5f} gate={vinfo['gate']:.5f}"
            f"  apply={vinfo['apply_rate']:.3f}  |pred|={vinfo['pred_mag']:.4f}"
            f"  |raw|={vinfo['raw_mag']:.4f}"
        )

        metric = vloss.item()
        if epoch % EVAL_EVERY == 0 and val_routes:
            vm, vs = rollout_cost(val_routes, model, mdl_path, ort_sess, eval_cache)
            line += f"  rollout={vm:.2f}±{vs:.2f}"
            metric = vm

        if metric < best_metric:
            best_metric = metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "chunk_k": CHUNK_K,
                    "delta_scale": DELTA_SCALE,
                    "max_residual": MAX_RESIDUAL,
                    "ref_input": REF_INPUT,
                    "hard_gate": HARD_GATE,
                    "gate_thresh": GATE_THRESH,
                },
                BEST_PT,
            )
            line += " ★"
        print(line)

    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
