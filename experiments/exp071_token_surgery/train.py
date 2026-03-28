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

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER, STEER_RANGE
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
MAX_DELTA = 0.5

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

ROUTES = int(os.getenv("ROUTES", "128"))
VAL_ROUTES = int(os.getenv("VAL_ROUTES", "32"))
TOPK_PER_ROUTE = int(os.getenv("TOPK_PER_ROUTE", "4"))
RAND_PER_ROUTE = int(os.getenv("RAND_PER_ROUTE", "8"))
STEP_STRIDE = int(os.getenv("STEP_STRIDE", "5"))
HORIZON_H = int(os.getenv("HORIZON_H", "20"))
PROBE_M = int(os.getenv("PROBE_M", "32"))
NOISE_STD_SMALL = float(os.getenv("NOISE_STD_SMALL", "0.03"))
NOISE_STD_LARGE = float(os.getenv("NOISE_STD_LARGE", "0.08"))
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", "10"))
USE_EXPECTED = os.getenv("USE_EXPECTED", "1") == "1"
CSV_MODE = os.getenv("CSV_MODE", "sample")
OUTER_ITERS = int(os.getenv("OUTER_ITERS", "1"))
INNER_EPOCHS = int(os.getenv("INNER_EPOCHS", "120"))
IMPROVE_MARGIN = float(os.getenv("IMPROVE_MARGIN", "10.0"))
APPLY_EPS = float(os.getenv("APPLY_EPS", "0.02"))
BASE_IMPROVE_MARGIN = float(os.getenv("BASE_IMPROVE_MARGIN", "10.0"))
CRIT_GATE_THRESH = float(os.getenv("CRIT_GATE_THRESH", "0.10"))

RES_HIDDEN = int(os.getenv("RES_HIDDEN", "128"))
RES_LAYERS = int(os.getenv("RES_LAYERS", "2"))
MAX_CORRECTION = float(os.getenv("MAX_CORRECTION", "0.20"))
HARD_GATE = os.getenv("HARD_GATE", "1") == "1"
GATE_THRESH = float(os.getenv("GATE_THRESH", "0.65"))

LR = float(os.getenv("LR", "3e-4"))
WD = float(os.getenv("WD", "1e-6"))
BS = int(os.getenv("BS", "256"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "20"))
REG_COEF = float(os.getenv("REG_COEF", "1.0"))
GATE_COEF = float(os.getenv("GATE_COEF", "0.5"))
POS_RAW_COEF = float(os.getenv("POS_RAW_COEF", "0.20"))
ZERO_RAW_COEF = float(os.getenv("ZERO_RAW_COEF", "0.05"))
GATE_SPARSE_COEF = float(os.getenv("GATE_SPARSE_COEF", "0.10"))

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


def _criticality_batch(future_target, current):
    horizon = min(CRIT_LOOKAHEAD, future_target.shape[1])
    fp = future_target[:, :horizon]
    if horizon <= 1:
        return (fp[:, 0] - current).abs() / S_LAT
    slope = torch.diff(fp, dim=1, prepend=fp[:, :1]) / DEL_T
    mismatch = (fp[:, min(4, horizon - 1)] - current).abs() / S_LAT
    span = (fp[:, -1] - fp[:, 0]).abs() / S_LAT
    peak_slope = slope.abs().amax(dim=1) / max(S_LAT / DEL_T, 1e-6)
    flip = ((fp[:, :-1] * fp[:, 1:]) < 0).any(dim=1).float()
    return mismatch + 0.5 * span + 0.5 * peak_slope + 0.5 * flip


class BaseTokenActor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*layers)
        self._features = None
        self.actor[-2].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out

    def token_and_raw(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return raw, self._features


class SurgeryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_actor = BaseTokenActor()
        in_dim = HIDDEN + 2

        res = [nn.Linear(in_dim, RES_HIDDEN), nn.ReLU()]
        for _ in range(RES_LAYERS - 1):
            res += [nn.Linear(RES_HIDDEN, RES_HIDDEN), nn.ReLU()]
        res.append(nn.Linear(RES_HIDDEN, 1))
        self.residual_head = nn.Sequential(*res)

        self.gate_head = nn.Sequential(
            nn.Linear(in_dim, RES_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(RES_HIDDEN // 2, 1),
        )

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def encode(self, obs):
        base_raw, token = self.base_actor.token_and_raw(obs)
        return token, base_raw

    def correction_and_gate(self, token, base_raw, criticality):
        x = torch.cat([token, base_raw.unsqueeze(-1), criticality.unsqueeze(-1)], dim=-1)
        corr = torch.tanh(self.residual_head(x)).squeeze(-1) * MAX_CORRECTION
        gate_logit = self.gate_head(x).squeeze(-1)
        return corr, gate_logit

    def correction_and_gate_from_obs(self, obs, criticality):
        base_raw, token = self.base_actor.token_and_raw(obs)
        corr, gate_logit = self.correction_and_gate(token, base_raw, criticality)
        return corr, gate_logit, base_raw, token

    def runtime_raw(self, obs, criticality, hard_gate=HARD_GATE, gate_thresh=GATE_THRESH):
        corr, gate_logit, base_raw, token = self.correction_and_gate_from_obs(obs, criticality)
        gate_prob = torch.sigmoid(gate_logit)
        crit_mask = (criticality >= CRIT_GATE_THRESH).float()
        if hard_gate:
            corr = corr * (gate_prob >= gate_thresh).float() * crit_mask
        else:
            corr = corr * gate_prob * crit_mask
        raw = (base_raw + corr).clamp(-1.0, 1.0)
        return raw, corr, gate_prob, base_raw, token


def load_base_into_model(model):
    if not BASE_PT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_PT}")
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
    future = _precompute_future_windows(sim.data_gpu)
    raw_hist = torch.zeros((sim.N, sim.T), dtype=torch.float32, device="cuda")
    base_hist = torch.zeros((sim.N, sim.T), dtype=torch.float32, device="cuda")

    def ctrl(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(sim_ref.N, dtype=torch.float64, device="cuda")
        with torch.inference_mode():
            obs = build_obs_from_sim(sim_ref, future, step_idx)
            crit = _criticality_batch(future["target_lataccel"][:, step_idx], sim_ref.current_lataccel.float())
            raw, _, _, base_raw, _ = model.runtime_raw(obs, crit)
        raw_hist[:, step_idx] = raw
        base_hist[:, step_idx] = base_raw
        prev_action = sim_ref.action_history[:, sim_ref._hist_len - 1]
        delta = raw.to(prev_action.dtype) * DELTA_SCALE
        delta = delta.clamp(-MAX_DELTA, MAX_DELTA)
        return (prev_action + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

    costs = sim.rollout(ctrl)
    return sim, future, np.asarray(costs["total_cost"], dtype=np.float64), raw_hist, base_hist


def select_windows(policy_sim, future, csv_files):
    steps = list(range(CONTROL_START_IDX, COST_END_IDX, STEP_STRIDE))
    s_idx = torch.tensor(steps, device=policy_sim.action_history.device, dtype=torch.long)
    rows = []
    for route_idx, csv_file in enumerate(csv_files):
        current_before = policy_sim.current_lataccel_history[route_idx, s_idx - 1].float()
        crit = _criticality_batch(future["target_lataccel"][route_idx, s_idx].float(), current_before)
        topk = min(TOPK_PER_ROUTE, len(steps))
        _, top_idx = torch.topk(crit, k=topk, largest=True, sorted=True)
        chosen = set(int(i) for i in top_idx.tolist())
        if RAND_PER_ROUTE > 0:
            remaining = [i for i in range(len(steps)) if i not in chosen]
            if remaining:
                chosen.update(random.sample(remaining, min(RAND_PER_ROUTE, len(remaining))))
        for local_idx in sorted(chosen):
            step_idx = steps[local_idx]
            rows.append(
                {
                    "csv_file": str(csv_file),
                    "route_idx": route_idx,
                    "step_idx": int(step_idx),
                    "criticality": float(crit[local_idx].item()),
                }
            )
    rows.sort(key=lambda r: (r["route_idx"], r["step_idx"]))
    return rows


def sample_candidate_raw(ref_raw, base_raw, num_cand, device):
    cand = torch.empty(num_cand, dtype=torch.float32, device=device)
    cand[0] = ref_raw
    if num_cand == 1:
        return cand.clamp_(-1.0, 1.0)
    cand[1] = base_raw
    if num_cand == 2:
        return cand.clamp_(-1.0, 1.0)
    n = num_cand - 2
    sigma = torch.empty(n, device=device)
    half = n // 2
    sigma[:half] = NOISE_STD_SMALL
    sigma[half:] = NOISE_STD_LARGE
    if n > 1:
        sigma = sigma[torch.randperm(n, device=device)]
    cand[2:] = ref_raw + sigma * torch.randn(n, device=device)
    return cand.clamp_(-1.0, 1.0)


def evaluate_window(csv_file, csv_cache, model, mdl_path, ort_sess, policy_sim, route_idx, step_idx, criticality, ref_raw, base_raw):
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

    cand_raw = sample_candidate_raw(ref_raw, base_raw, PROBE_M, sim.current_lataccel.device)
    costs = torch.zeros(PROBE_M, dtype=torch.float32, device="cuda")
    end_step = min(step_idx + HORIZON_H, COST_END_IDX)

    for t in range(step_idx, end_step):
        if t == step_idx:
            raw = cand_raw
        else:
            with torch.inference_mode():
                obs = build_obs_from_sim(sim, future, t)
                crit = _criticality_batch(future["target_lataccel"][:, t], sim.current_lataccel.float())
                raw, _, _, _, _ = model.runtime_raw(obs, crit)
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
    base_cost = float(costs[1].item()) if PROBE_M > 1 else ref_cost
    best_cost = float(costs[best_idx].item())
    improvement = ref_cost - best_cost
    base_improvement = base_cost - best_cost
    best_raw = float(cand_raw[best_idx].item())
    accept_teacher = (
        criticality >= CRIT_GATE_THRESH
        and best_idx != 1
        and base_improvement > BASE_IMPROVE_MARGIN
    )
    if accept_teacher:
        target_raw = best_raw
    else:
        target_raw = float(base_raw)
    target_corr = float(np.clip(target_raw - base_raw, -MAX_CORRECTION, MAX_CORRECTION))
    apply_label = float(
        (abs(target_corr) > APPLY_EPS)
        and (criticality >= CRIT_GATE_THRESH)
    )
    return {
        "obs": obs0,
        "target_corr": target_corr,
        "target_raw": target_raw,
        "best_raw": best_raw,
        "apply_label": apply_label,
        "ref_cost": ref_cost,
        "base_cost": base_cost,
        "best_cost": best_cost,
        "improvement": improvement,
        "base_improvement": base_improvement,
    }


def collect_iteration(rows_accum, routes, split_name, model, mdl_path, ort_sess, csv_cache, iter_idx):
    policy_sim, future, rollout_costs, raw_hist, base_hist = rollout_policy(routes, model, mdl_path, ort_sess, csv_cache)
    windows = select_windows(policy_sim, future, routes)
    new_rows = []
    for i, win in enumerate(windows, start=1):
        route_idx = win["route_idx"]
        step_idx = win["step_idx"]
        out = evaluate_window(
            csv_file=win["csv_file"],
            csv_cache=csv_cache,
            model=model,
            mdl_path=mdl_path,
            ort_sess=ort_sess,
            policy_sim=policy_sim,
            route_idx=route_idx,
            step_idx=step_idx,
            criticality=win["criticality"],
            ref_raw=float(raw_hist[route_idx, step_idx].item()),
            base_raw=float(base_hist[route_idx, step_idx].item()),
        )
        row = {
            "iter": iter_idx,
            "split": split_name,
            "csv_file": win["csv_file"],
            "route_idx": route_idx,
            "step_idx": step_idx,
            "criticality": win["criticality"],
            "obs": out["obs"],
            "target_corr": out["target_corr"],
            "target_raw": out["target_raw"],
            "best_raw": out["best_raw"],
            "apply_label": out["apply_label"],
            "ref_cost": out["ref_cost"],
            "best_cost": out["best_cost"],
            "improvement": out["improvement"],
        }
        rows_accum.append(row)
        new_rows.append(row)
        print(
            f"[iter{iter_idx} {split_name} {i:4d}/{len(windows):4d}] route={route_idx:3d} step={step_idx:3d}"
            f" crit={row['criticality']:.3f} apply={int(row['apply_label'])} Δ={row['improvement']:+8.3f}"
        )
    return new_rows, rollout_costs


def write_bank(rows):
    payload = {
        "obs": torch.stack([r["obs"].cpu() for r in rows], dim=0),
        "criticality": torch.tensor([r["criticality"] for r in rows], dtype=torch.float32),
        "target_corr": torch.tensor([r["target_corr"] for r in rows], dtype=torch.float32),
        "apply_label": torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32),
        "improvement": torch.tensor([r["improvement"] for r in rows], dtype=torch.float32),
        "iter": torch.tensor([r["iter"] for r in rows], dtype=torch.long),
        "split": [r["split"] for r in rows],
        "csv_file": [r["csv_file"] for r in rows],
        "route_idx": torch.tensor([r["route_idx"] for r in rows], dtype=torch.long),
        "step_idx": torch.tensor([r["step_idx"] for r in rows], dtype=torch.long),
        "meta": {
            "step_stride": STEP_STRIDE,
            "horizon_h": HORIZON_H,
            "probe_m": PROBE_M,
            "improve_margin": IMPROVE_MARGIN,
            "delta_scale": DELTA_SCALE,
            "max_correction": MAX_CORRECTION,
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
                "step_idx",
                "criticality",
                "target_corr",
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
                    r["step_idx"],
                    r["criticality"],
                    r["target_corr"],
                    r["apply_label"],
                    r["ref_cost"],
                    r["best_cost"],
                    r["improvement"],
                ]
            )


def weighted_losses(model, obs, criticality, target_corr, apply_label, improvement):
    pred_corr_raw, gate_logit, base_raw, token = model.correction_and_gate_from_obs(obs, criticality)
    gate_prob = torch.sigmoid(gate_logit)
    crit_mask = (criticality >= CRIT_GATE_THRESH).float()
    apply_target = apply_label * crit_mask
    pred_corr = pred_corr_raw * gate_prob * crit_mask

    pos_mask = apply_target > 0.5
    imp_w = torch.ones_like(improvement)
    if pos_mask.any():
        pos_scale = 1.0 + (improvement / improvement[pos_mask].mean().clamp_min(1e-6)).clamp(0.0, 8.0)
        imp_w = torch.where(pos_mask, pos_scale, imp_w)

    reg_per = F.smooth_l1_loss(pred_corr, target_corr, reduction="none")
    reg_loss = (reg_per * imp_w).mean()
    gate_loss = F.binary_cross_entropy_with_logits(gate_logit, apply_target)
    sparse_loss = gate_prob.mean()

    if pos_mask.any():
        pos_raw = F.smooth_l1_loss(pred_corr_raw[pos_mask], target_corr[pos_mask], reduction="mean")
    else:
        pos_raw = pred_corr_raw.sum() * 0.0

    zero_mask = ~pos_mask
    if zero_mask.any():
        zero_raw = pred_corr_raw[zero_mask].pow(2).mean()
    else:
        zero_raw = pred_corr_raw.sum() * 0.0

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
        "corr_mag": pred_corr.detach().abs().mean().item(),
        "raw_mag": pred_corr_raw.detach().abs().mean().item(),
        "base_mag": base_raw.detach().abs().mean().item(),
        "token_mag": token.detach().abs().mean().item(),
    }


def get_split_idx(rows, split_name):
    return torch.tensor([i for i, r in enumerate(rows) if r["split"] == split_name], dtype=torch.long, device=DEV)


def tensors_from_rows(rows):
    obs = torch.stack([r["obs"] for r in rows], dim=0).to(DEV)
    criticality = torch.tensor([r["criticality"] for r in rows], dtype=torch.float32, device=DEV)
    target_corr = torch.tensor([r["target_corr"] for r in rows], dtype=torch.float32, device=DEV)
    apply_label = torch.tensor([r["apply_label"] for r in rows], dtype=torch.float32, device=DEV)
    improvement = torch.tensor([max(r["improvement"], 0.0) for r in rows], dtype=torch.float32, device=DEV)
    return obs, criticality, target_corr, apply_label, improvement


def rollout_eval(csv_files, model, mdl_path, ort_sess, csv_cache):
    _, _, costs, _, _ = rollout_policy(csv_files, model, mdl_path, ort_sess, csv_cache)
    return float(costs.mean()), float(costs.std())


def write_summary(lines):
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")


def train():
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    train_routes, val_routes = choose_routes(all_csv)
    selected = train_routes + [f for f in val_routes if f not in set(train_routes)]
    csv_cache = CSVCache(selected)
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    model = SurgeryModel().to(DEV)
    load_base_into_model(model)
    opt = optim.Adam(
        list(model.residual_head.parameters()) + list(model.gate_head.parameters()),
        lr=LR,
        weight_decay=WD,
    )

    rows = []
    best_metric = float("inf")
    eval_cache = CSVCache(val_routes) if val_routes else None
    summary_lines = [
        "exp071 token surgery",
        f"train_routes={len(train_routes)} val_routes={len(val_routes)}",
        f"step_stride={STEP_STRIDE} horizon={HORIZON_H} probe_m={PROBE_M} topk={TOPK_PER_ROUTE} rand={RAND_PER_ROUTE}",
        f"outer={OUTER_ITERS} inner={INNER_EPOCHS} improve_margin={IMPROVE_MARGIN:g} max_correction={MAX_CORRECTION:g}",
    ]
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

        obs, criticality, target_corr, apply_label, improvement = tensors_from_rows(rows)
        tr_idx = get_split_idx(rows, "train")
        va_idx = get_split_idx(rows, "val")

        for inner in range(INNER_EPOCHS):
            model.train()
            perm = tr_idx[torch.randperm(len(tr_idx), device=DEV)]
            train_loss_sum = 0.0
            n_seen = 0
            for idx in perm.split(BS):
                loss, _ = weighted_losses(model, obs[idx], criticality[idx], target_corr[idx], apply_label[idx], improvement[idx])
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
                    target_corr[va_idx],
                    apply_label[va_idx],
                    improvement[va_idx],
                )

            global_epoch = outer * INNER_EPOCHS + inner
            line = (
                f"I{outer} E{inner:3d}  train={train_loss_sum / max(1, n_seen):.5f}  val={vloss.item():.5f}"
                f"  reg={vinfo['reg']:.5f} gate={vinfo['gate']:.5f} sparse={vinfo['sparse']:.3f}"
                f"  apply={vinfo['apply']:.3f}  |corr|={vinfo['corr_mag']:.4f}  |raw|={vinfo['raw_mag']:.4f}"
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
                        "delta_scale": DELTA_SCALE,
                        "max_correction": MAX_CORRECTION,
                        "hard_gate": HARD_GATE,
                        "gate_thresh": GATE_THRESH,
                    },
                    BEST_PT,
                )
                line += " ★"
            print(line)

    summary_lines.append(f"elapsed={time.time() - t0:.1f}s")
    write_summary(summary_lines)
    print(f"saved best: {BEST_PT}")


if __name__ == "__main__":
    train()
