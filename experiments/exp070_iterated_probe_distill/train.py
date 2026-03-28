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

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

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

CHUNK_K = int(os.getenv("CHUNK_K", "5"))
HORIZON_H = int(os.getenv("HORIZON_H", "20"))
ROUTES = int(os.getenv("ROUTES", "128"))
VAL_ROUTES = int(os.getenv("VAL_ROUTES", "32"))
TOPK_PER_ROUTE = int(os.getenv("TOPK_PER_ROUTE", "2"))
RAND_PER_ROUTE = int(os.getenv("RAND_PER_ROUTE", "4"))
PROBE_M = int(os.getenv("PROBE_M", "32"))
NOISE_STD_SMALL = float(os.getenv("NOISE_STD_SMALL", "0.03"))
NOISE_STD_LARGE = float(os.getenv("NOISE_STD_LARGE", "0.08"))
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", str(max(2 * CHUNK_K, 8))))
USE_EXPECTED = os.getenv("USE_EXPECTED", "1") == "1"
IMPROVE_MARGIN = float(os.getenv("IMPROVE_MARGIN", "10.0"))
CSV_MODE = os.getenv("CSV_MODE", "sample")

MAX_RESIDUAL = float(os.getenv("MAX_RESIDUAL", "0.25"))
REF_INPUT = os.getenv("REF_INPUT", "1") == "1"
HARD_GATE = os.getenv("HARD_GATE", "1") == "1"
GATE_THRESH = float(os.getenv("GATE_THRESH", "0.65"))
LR = float(os.getenv("LR", "3e-4"))
WD = float(os.getenv("WD", "1e-6"))
BS = int(os.getenv("BS", "256"))
OUTER_ITERS = int(os.getenv("OUTER_ITERS", "5"))
INNER_EPOCHS = int(os.getenv("INNER_EPOCHS", "60"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "20"))
REG_COEF = float(os.getenv("REG_COEF", "1.0"))
GATE_COEF = float(os.getenv("GATE_COEF", "0.5"))
POS_RAW_COEF = float(os.getenv("POS_RAW_COEF", "0.15"))
ZERO_RAW_COEF = float(os.getenv("ZERO_RAW_COEF", "0.03"))
GATE_SPARSE_COEF = float(os.getenv("GATE_SPARSE_COEF", "0.05"))

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


class IterStudent(nn.Module):
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

        self.gate_head = nn.Sequential(
            nn.Linear(in_dim, HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),
        )

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def base_raw(self, obs):
        return self.base_actor.raw(obs)

    def _inp(self, obs, base_raw, criticality):
        crit = criticality.unsqueeze(-1)
        x = torch.cat([obs, crit], dim=-1)
        if REF_INPUT:
            x = torch.cat([x, base_raw.unsqueeze(-1).expand(-1, self.chunk_k)], dim=-1)
        return x

    def residual_and_gate(self, obs, criticality):
        base_raw = self.base_raw(obs)
        x = self._inp(obs, base_raw, criticality)
        resid = torch.tanh(self.residual_head(x)) * MAX_RESIDUAL
        gate_logit = self.gate_head(x).squeeze(-1)
        return resid, gate_logit, base_raw

    def runtime_chunk(self, obs, criticality, hard_gate=HARD_GATE, gate_thresh=GATE_THRESH):
        resid, gate_logit, base_raw = self.residual_and_gate(obs, criticality)
        gate_prob = torch.sigmoid(gate_logit)
        if hard_gate:
            resid = resid * (gate_prob >= gate_thresh).float().unsqueeze(-1)
        else:
            resid = resid * gate_prob.unsqueeze(-1)
        return resid, gate_prob, base_raw


def load_base_into_model(model):
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    actor_state = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    model.base_actor.load_state_dict(actor_state, strict=False)
    model.base_actor.eval()
    for p in model.base_actor.parameters():
        p.requires_grad_(False)


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


def sample_candidate_chunks(ref_chunk, num_cand):
    chunks = ref_chunk.unsqueeze(0).repeat(num_cand, 1)
    if num_cand <= 1:
        return chunks
    n = num_cand - 1
    u = torch.linspace(-1.0, 1.0, CHUNK_K, device=ref_chunk.device)
    basis = torch.stack([torch.ones_like(u), u, u.square() - u.square().mean()], dim=0)
    coeff = torch.randn(n, basis.shape[0], device=ref_chunk.device)
    resid = coeff @ basis
    resid = resid / resid.std(dim=1, keepdim=True).clamp_min(1e-6)
    sigma = torch.empty(n, 1, device=ref_chunk.device)
    half = n // 2
    sigma[:half] = NOISE_STD_SMALL
    sigma[half:] = NOISE_STD_LARGE
    if n > 1:
        sigma = sigma[torch.randperm(n, device=ref_chunk.device)]
    resid = resid * sigma + 0.2 * sigma * torch.randn(n, CHUNK_K, device=ref_chunk.device)
    chunks[1:] = (ref_chunk.unsqueeze(0) + resid).clamp(-1.0, 1.0)
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


def rollout_policy(csv_files, model, mdl_path, ort_sess, csv_cache):
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
            if planned is None or step_idx % CHUNK_K == 0:
                planned, _, base_raw = model.runtime_chunk(obs, crit)
                step_in_chunk = 0
            else:
                base_raw = model.base_raw(obs)
        residual = planned[:, step_in_chunk]
        step_in_chunk = min(step_in_chunk + 1, CHUNK_K - 1)
        raw = (base_raw + residual).clamp(-1.0, 1.0)
        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)
    return sim, future, np.asarray(costs["total_cost"], dtype=np.float64)


def select_windows(policy_sim, future, csv_files):
    boundaries = list(range(CONTROL_START_IDX, COST_END_IDX - CHUNK_K + 1, CHUNK_K))
    b_idx = torch.tensor(boundaries, device=policy_sim.action_history.device, dtype=torch.long)
    rows = []
    for route_idx, csv_file in enumerate(csv_files):
        current_before = policy_sim.current_lataccel_history[route_idx, b_idx - 1].float()
        crit = _chunk_criticality(future["target_lataccel"][route_idx, b_idx].float(), current_before)
        topk = min(TOPK_PER_ROUTE, len(boundaries))
        _, top_idx = torch.topk(crit, k=topk, largest=True, sorted=True)
        chosen = set(int(i) for i in top_idx.tolist())
        if RAND_PER_ROUTE > 0:
            remaining = [i for i in range(len(boundaries)) if i not in chosen]
            if remaining:
                chosen.update(random.sample(remaining, min(RAND_PER_ROUTE, len(remaining))))

        for local_idx in sorted(chosen):
            step_idx = boundaries[local_idx]
            prev = policy_sim.action_history[route_idx, step_idx - 1 : step_idx - 1 + CHUNK_K]
            nxt = policy_sim.action_history[route_idx, step_idx : step_idx + CHUNK_K]
            ref_chunk_raw = ((nxt - prev) / DELTA_SCALE).float().clamp(-1.0, 1.0)
            rows.append(
                {
                    "csv_file": str(csv_file),
                    "route_idx": route_idx,
                    "step_idx": int(step_idx),
                    "chunk_idx": (step_idx - CONTROL_START_IDX) // CHUNK_K,
                    "criticality": float(crit[local_idx].item()),
                    "ref_chunk_raw": ref_chunk_raw,
                }
            )
    rows.sort(key=lambda r: (r["route_idx"], r["step_idx"]))
    return rows


def evaluate_window(csv_file, csv_cache, model, mdl_path, ort_sess, policy_sim, route_idx, step_idx, ref_chunk_raw):
    tiled = [csv_file for _ in range(PROBE_M)]
    data, rng = csv_cache.slice(tiled)
    sim = BatchedSimulator(str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng)
    if USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    h = step_idx
    sim._hist_len = h
    sim.action_history[:, :h] = policy_sim.action_history[route_idx : route_idx + 1, :h].repeat(PROBE_M, 1)
    sim.current_lataccel_history[:, :h] = policy_sim.current_lataccel_history[route_idx : route_idx + 1, :h].repeat(PROBE_M, 1)
    sim.current_lataccel = sim.current_lataccel_history[:, h - 1].clone()
    sim.state_history[:, :h, 0] = sim.data_gpu["roll_lataccel"][:, :h]
    sim.state_history[:, :h, 1] = sim.data_gpu["v_ego"][:, :h]
    sim.state_history[:, :h, 2] = sim.data_gpu["a_ego"][:, :h]

    with torch.inference_mode():
        obs0 = build_obs_from_sim(sim, future, step_idx)[0].detach().clone()

    cand_raw = sample_candidate_chunks(ref_chunk_raw, PROBE_M)
    costs = torch.zeros(PROBE_M, dtype=torch.float32, device="cuda")
    planned = None
    step_in_chunk = 0
    end_step = min(step_idx + HORIZON_H, COST_END_IDX)

    for t in range(step_idx, end_step):
        if t < step_idx + CHUNK_K:
            raw = cand_raw[:, t - step_idx]
        else:
            with torch.inference_mode():
                obs = build_obs_from_sim(sim, future, t)
                crit = _chunk_criticality(future["target_lataccel"][:, t], sim.current_lataccel.float())
                if planned is None or t % CHUNK_K == 0:
                    planned, _, base_raw = model.runtime_chunk(obs, crit)
                    step_in_chunk = 0
                else:
                    base_raw = model.base_raw(obs)
            residual = planned[:, step_in_chunk]
            step_in_chunk = min(step_in_chunk + 1, CHUNK_K - 1)
            raw = (base_raw + residual).clamp(-1.0, 1.0)

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
    ref_cost = float(costs[0].item())
    best_cost = float(costs[best_idx].item())
    improvement = ref_cost - best_cost
    best_chunk = cand_raw[best_idx].detach().clone()
    target_chunk = best_chunk if improvement > IMPROVE_MARGIN else cand_raw[0].detach().clone()
    return {
        "obs": obs0,
        "ref_chunk_raw": cand_raw[0].detach().clone(),
        "target_chunk_raw": target_chunk,
        "best_chunk_raw": best_chunk,
        "apply_label": float(improvement > IMPROVE_MARGIN),
        "ref_cost": ref_cost,
        "best_cost": best_cost,
        "improvement": improvement,
        "best_idx": best_idx,
    }


def collect_iteration(rows_accum, routes, split_name, model, mdl_path, ort_sess, csv_cache, iter_idx):
    policy_sim, future, rollout_costs = rollout_policy(routes, model, mdl_path, ort_sess, csv_cache)
    windows = select_windows(policy_sim, future, routes)
    new_rows = []
    for i, win in enumerate(windows, start=1):
        out = evaluate_window(
            csv_file=win["csv_file"],
            csv_cache=csv_cache,
            model=model,
            mdl_path=mdl_path,
            ort_sess=ort_sess,
            policy_sim=policy_sim,
            route_idx=win["route_idx"],
            step_idx=win["step_idx"],
            ref_chunk_raw=win["ref_chunk_raw"].to(DEV),
        )
        row = {
            "iter": iter_idx,
            "split": split_name,
            "csv_file": win["csv_file"],
            "route_idx": win["route_idx"],
            "chunk_idx": win["chunk_idx"],
            "step_idx": win["step_idx"],
            "criticality": win["criticality"],
            "obs": out["obs"],
            "ref_chunk_raw": out["ref_chunk_raw"],
            "target_chunk_raw": out["target_chunk_raw"],
            "best_chunk_raw": out["best_chunk_raw"],
            "target_resid_raw": out["target_chunk_raw"] - out["ref_chunk_raw"],
            "apply_label": out["apply_label"],
            "ref_cost": out["ref_cost"],
            "best_cost": out["best_cost"],
            "improvement": out["improvement"],
        }
        rows_accum.append(row)
        new_rows.append(row)
        print(
            f"[iter{iter_idx} {split_name} {i:4d}/{len(windows):4d}] route={row['route_idx']:3d} step={row['step_idx']:3d}"
            f" crit={row['criticality']:.3f} apply={int(row['apply_label'])} Δ={row['improvement']:+8.3f}"
        )
    return new_rows, rollout_costs


def write_bank(rows):
    payload = {
        "obs": torch.stack([r["obs"].cpu() for r in rows], dim=0),
        "ref_chunk_raw": torch.stack([r["ref_chunk_raw"].cpu() for r in rows], dim=0),
        "target_chunk_raw": torch.stack([r["target_chunk_raw"].cpu() for r in rows], dim=0),
        "target_resid_raw": torch.stack([r["target_resid_raw"].cpu() for r in rows], dim=0),
        "apply_label": torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32),
        "improvement": torch.tensor([r["improvement"] for r in rows], dtype=torch.float32),
        "criticality": torch.tensor([r["criticality"] for r in rows], dtype=torch.float32),
        "iter": torch.tensor([r["iter"] for r in rows], dtype=torch.long),
        "split": [r["split"] for r in rows],
        "csv_file": [r["csv_file"] for r in rows],
        "route_idx": torch.tensor([r["route_idx"] for r in rows], dtype=torch.long),
        "chunk_idx": torch.tensor([r["chunk_idx"] for r in rows], dtype=torch.long),
        "step_idx": torch.tensor([r["step_idx"] for r in rows], dtype=torch.long),
        "meta": {
            "chunk_k": CHUNK_K,
            "horizon_h": HORIZON_H,
            "probe_m": PROBE_M,
            "improve_margin": IMPROVE_MARGIN,
            "delta_scale": DELTA_SCALE,
        },
    }
    torch.save(payload, BANK_PT)

    with open(BANK_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iter",
                "split",
                "csv_file",
                "route_idx",
                "chunk_idx",
                "step_idx",
                "criticality",
                "apply_label",
                "ref_cost",
                "best_cost",
                "improvement",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["iter"],
                    r["split"],
                    r["csv_file"],
                    r["route_idx"],
                    r["chunk_idx"],
                    r["step_idx"],
                    r["criticality"],
                    r["apply_label"],
                    r["ref_cost"],
                    r["best_cost"],
                    r["improvement"],
                ]
            )


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
    sparse_loss = gate_prob.mean()

    if pos_mask.any():
        pos_raw = F.smooth_l1_loss(pred_raw[pos_mask], target_resid[pos_mask], reduction="mean")
    else:
        pos_raw = pred_raw.sum() * 0.0

    zero_mask = ~pos_mask
    if zero_mask.any():
        zero_raw = pred_raw[zero_mask].pow(2).mean()
    else:
        zero_raw = pred_raw.sum() * 0.0

    loss = (
        REG_COEF * reg_loss
        + GATE_COEF * gate_loss
        + POS_RAW_COEF * pos_raw
        + ZERO_RAW_COEF * zero_raw
        + GATE_SPARSE_COEF * sparse_loss
    )
    return loss, {
        "reg": reg_loss.detach().item(),
        "gate": gate_loss.detach().item(),
        "sparse": sparse_loss.detach().item(),
        "apply": gate_prob.detach().mean().item(),
        "pred_mag": pred.detach().abs().mean().item(),
        "raw_mag": pred_raw.detach().abs().mean().item(),
    }


def get_split_idx(rows, split_name):
    return torch.tensor([i for i, r in enumerate(rows) if r["split"] == split_name], dtype=torch.long, device=DEV)


def tensors_from_rows(rows):
    obs = torch.stack([r["obs"] for r in rows], dim=0).to(DEV)
    criticality = torch.tensor([r["criticality"] for r in rows], dtype=torch.float32, device=DEV)
    target_resid = torch.stack([r["target_resid_raw"] for r in rows], dim=0).to(DEV)
    apply_label = torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32, device=DEV)
    improvement = torch.tensor([max(r["improvement"], 0.0) for r in rows], dtype=torch.float32, device=DEV)
    return obs, criticality, target_resid, apply_label, improvement


def rollout_eval(csv_files, model, mdl_path, ort_sess, csv_cache):
    _, _, costs = rollout_policy(csv_files, model, mdl_path, ort_sess, csv_cache)
    return float(costs.mean()), float(costs.std())


def write_summary(text_lines):
    SUMMARY_TXT.write_text("\n".join(text_lines) + "\n")


def train():
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    train_routes, val_routes = choose_routes(all_csv)
    selected = train_routes + [f for f in val_routes if f not in set(train_routes)]
    csv_cache = CSVCache(selected)
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    model = IterStudent(CHUNK_K).to(DEV)
    load_base_into_model(model)

    opt = optim.Adam(
        list(model.residual_head.parameters()) + list(model.gate_head.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    best_metric = float("inf")
    rows = []
    summary_lines = [
        "exp070 iterated probe distill",
        f"train_routes={len(train_routes)} val_routes={len(val_routes)}",
        f"chunk={CHUNK_K} horizon={HORIZON_H} probe_m={PROBE_M} topk={TOPK_PER_ROUTE} rand={RAND_PER_ROUTE}",
        f"outer={OUTER_ITERS} inner={INNER_EPOCHS} improve_margin={IMPROVE_MARGIN:g} hard_gate={'on' if HARD_GATE else 'off'}@{GATE_THRESH:g}",
    ]

    eval_cache = CSVCache(val_routes) if val_routes else None
    t0 = time.time()

    for outer in range(OUTER_ITERS):
        print(f"\nCollect iteration {outer}")
        new_train, train_costs = collect_iteration(rows, train_routes, "train", model, mdl_path, ort_sess, csv_cache, outer)
        new_val, val_costs = collect_iteration(rows, val_routes, "val", model, mdl_path, ort_sess, csv_cache, outer)
        write_bank(rows)

        def summarize(split_rows, name):
            imp = np.asarray([r["improvement"] for r in split_rows], dtype=np.float64)
            apply = np.asarray([r["apply_label"] for r in split_rows], dtype=np.float64)
            summary_lines.append(
                f"iter{outer}_{name}: windows={len(split_rows)} apply_frac={apply.mean() if len(apply) else 0.0:.3f}"
                f" imp_mean={imp.mean() if len(imp) else 0.0:.3f} imp_median={np.median(imp) if len(imp) else 0.0:.3f}"
            )

        summarize(new_train, "train")
        summarize(new_val, "val")
        print(f"iter{outer} train_policy_cost={train_costs.mean():.2f} val_policy_cost={val_costs.mean():.2f}")

        obs, criticality, target_resid, apply_label, improvement = tensors_from_rows(rows)
        tr_idx = get_split_idx(rows, "train")
        va_idx = get_split_idx(rows, "val")

        for inner in range(INNER_EPOCHS):
            model.train()
            perm = tr_idx[torch.randperm(len(tr_idx), device=DEV)]
            train_loss_sum = 0.0
            n_seen = 0
            for idx in perm.split(BS):
                loss, _ = weighted_losses(model, obs[idx], criticality[idx], target_resid[idx], apply_label[idx], improvement[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.residual_head.parameters()) + list(model.gate_head.parameters()),
                    1.0,
                )
                opt.step()
                train_loss_sum += loss.detach().item() * idx.numel()
                n_seen += idx.numel()

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

            global_epoch = outer * INNER_EPOCHS + inner
            line = (
                f"I{outer} E{inner:3d}  train={train_loss_sum/max(1,n_seen):.5f}  val={vloss.item():.5f}"
                f"  reg={vinfo['reg']:.5f} gate={vinfo['gate']:.5f} sparse={vinfo['sparse']:.3f}"
                f"  apply={vinfo['apply']:.3f}  |pred|={vinfo['pred_mag']:.4f}  |raw|={vinfo['raw_mag']:.4f}"
            )
            metric = vloss.item()
            if global_epoch % EVAL_EVERY == 0 and val_routes:
                vm, vs = rollout_eval(val_routes, model, mdl_path, ort_sess, eval_cache)
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

    summary_lines.append(f"elapsed={time.time()-t0:.1f}s")
    write_summary(summary_lines)
    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
