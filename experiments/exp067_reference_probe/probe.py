import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    ACC_G,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEV = torch.device("cuda")

# ── architecture / controller (must match exp055) ─────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
MAX_DELTA = 0.5

# ── probe parameters ──────────────────────────────────────────────────────────
CHUNK_K = int(os.getenv("CHUNK_K", "5"))
HORIZON_H = int(os.getenv("HORIZON_H", "20"))
ROUTES = int(os.getenv("ROUTES", "64"))
TOPK_PER_ROUTE = int(os.getenv("TOPK_PER_ROUTE", "2"))
PROBE_M = int(os.getenv("PROBE_M", "32"))  # total candidates includes base slot 0
NOISE_STD_SMALL = float(os.getenv("NOISE_STD_SMALL", "0.03"))
NOISE_STD_LARGE = float(os.getenv("NOISE_STD_LARGE", "0.08"))
CRIT_LOOKAHEAD = int(os.getenv("CRIT_LOOKAHEAD", str(max(2 * CHUNK_K, 8))))
USE_EXPECTED = os.getenv("USE_EXPECTED", "1") == "1"
CSV_MODE = os.getenv("CSV_MODE", "sample")  # sample | first
SAVE_CSV = os.getenv("SAVE_CSV", "1") == "1"
SAVE_PT = os.getenv("SAVE_PT", "1") == "1"

# ── scaling / obs layout (must match exp055) ─────────────────────────────────
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

EXP_DIR = Path(__file__).parent
OUT_CSV = EXP_DIR / "probe_results.csv"
OUT_SUMMARY = EXP_DIR / "probe_summary.txt"
OUT_PT = EXP_DIR / "probe_wins.pt"
BASE_PT = Path(
    os.getenv(
        "BASE_MODEL",
        str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"),
    )
)

assert CHUNK_K > 0
assert HORIZON_H >= CHUNK_K
assert PROBE_M >= 1


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


def load_base_actor():
    if not BASE_PT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_PT}")
    ckpt = torch.load(BASE_PT, weights_only=False, map_location="cpu")
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


def sample_candidate_chunks(base_chunk, num_cand):
    chunks = base_chunk.unsqueeze(0).repeat(num_cand, 1)
    if num_cand <= 1:
        return chunks

    n = num_cand - 1
    u = torch.linspace(-1.0, 1.0, CHUNK_K, device=base_chunk.device)
    basis = torch.stack(
        [
            torch.ones_like(u),
            u,
            u.square() - u.square().mean(),
        ],
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
    resid = resid * sigma
    resid = resid + 0.2 * sigma * torch.randn(n, CHUNK_K, device=base_chunk.device)
    chunks[1:] = (base_chunk.unsqueeze(0) + resid).clamp(-1.0, 1.0)
    return chunks


def run_base_rollout(csv_files, base_actor, mdl_path, ort_session, csv_cache):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
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


def evaluate_window(
    csv_file,
    csv_cache,
    base_actor,
    mdl_path,
    ort_session,
    base_sim,
    route_idx,
    step_idx,
    base_chunk_raw,
):
    tiled = [csv_file for _ in range(PROBE_M)]
    data, rng = csv_cache.slice(tiled)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    if USE_EXPECTED:
        sim.use_expected = True
    future = _precompute_future_windows(sim.data_gpu)

    h = step_idx
    sim._hist_len = h
    sim.action_history[:, :h] = base_sim.action_history[route_idx : route_idx + 1, :h].repeat(
        PROBE_M, 1
    )
    sim.current_lataccel_history[:, :h] = base_sim.current_lataccel_history[
        route_idx : route_idx + 1, :h
    ].repeat(PROBE_M, 1)
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
        lat_cost = (tgt - cur).square() * (100 * LAT_ACCEL_COST_MULTIPLIER)
        jerk_cost = jerk.square() * 100
        costs += lat_cost + jerk_cost

    best_idx = int(costs.argmin().item())
    base_cost = float(costs[0].item())
    best_cost = float(costs[best_idx].item())
    improvement = base_cost - best_cost
    return {
        "obs": obs0,
        "base_chunk_raw": cand_raw[0].detach().clone(),
        "best_chunk_raw": cand_raw[best_idx].detach().clone(),
        "base_cost": base_cost,
        "best_cost": best_cost,
        "improvement": improvement,
        "best_idx": best_idx,
        "best_mean_abs_resid": float((cand_raw[best_idx] - cand_raw[0]).abs().mean().item()),
        "best_max_abs_resid": float((cand_raw[best_idx] - cand_raw[0]).abs().amax().item()),
    }


def route_boundary_windows(base_sim, future, csv_files):
    boundaries = list(range(CONTROL_START_IDX, COST_END_IDX - CHUNK_K + 1, CHUNK_K))
    crit_rows = []
    act = base_sim.action_history
    b_idx = torch.tensor(boundaries, device=act.device, dtype=torch.long)
    for route_idx, csv_file in enumerate(csv_files):
        current_before = base_sim.current_lataccel_history[route_idx, b_idx - 1].float()
        crit = _chunk_criticality(
            future["target_lataccel"][route_idx, b_idx].float(),
            current_before,
        )
        topk = min(TOPK_PER_ROUTE, len(boundaries))
        vals, idx = torch.topk(crit, k=topk, largest=True, sorted=True)
        for score, local_idx in zip(vals.tolist(), idx.tolist()):
            step_idx = boundaries[local_idx]
            prev = act[route_idx, step_idx - 1 : step_idx - 1 + CHUNK_K]
            nxt = act[route_idx, step_idx : step_idx + CHUNK_K]
            base_chunk_raw = ((nxt - prev) / DELTA_SCALE).float().clamp(-1.0, 1.0)
            crit_rows.append(
                {
                    "csv_file": str(csv_file),
                    "route_idx": route_idx,
                    "step_idx": int(step_idx),
                    "chunk_idx": (step_idx - CONTROL_START_IDX) // CHUNK_K,
                    "criticality": float(score),
                    "base_chunk_raw": base_chunk_raw,
                }
            )
    crit_rows.sort(key=lambda x: x["criticality"], reverse=True)
    return crit_rows


def choose_csvs(all_csv):
    if ROUTES >= len(all_csv):
        return list(all_csv)
    if CSV_MODE == "first":
        return list(all_csv[:ROUTES])
    return random.sample(list(all_csv), ROUTES)


def write_summary(df, base_costs):
    improved = df["improvement"] > 0.0
    lines = [
        "exp067 reference-chunk probe",
        f"routes={len(base_costs)}  windows={len(df)}  probe_m={PROBE_M}  chunk_k={CHUNK_K}  horizon_h={HORIZON_H}",
        f"base_episode_cost mean={np.mean(base_costs):.3f} std={np.std(base_costs):.3f}",
        f"window_improved frac={improved.mean():.3f} count={int(improved.sum())}/{len(df)}",
        f"best_improvement mean={df['improvement'].mean():.3f} median={df['improvement'].median():.3f} p90={df['improvement'].quantile(0.9):.3f}",
        f"improved_only mean={(df.loc[improved, 'improvement'].mean() if improved.any() else 0.0):.3f}",
        f"best_h_cost mean={df['best_cost'].mean():.3f}  base_h_cost mean={df['base_cost'].mean():.3f}",
    ]
    text = "\n".join(lines)
    OUT_SUMMARY.write_text(text + "\n")
    return text


def save_win_bank(rows):
    improved = [r for r in rows if r["improvement"] > 0.0]
    if not improved:
        payload = {
            "obs": torch.empty((0, OBS_DIM), dtype=torch.float32),
            "base_chunk_raw": torch.empty((0, CHUNK_K), dtype=torch.float32),
            "best_chunk_raw": torch.empty((0, CHUNK_K), dtype=torch.float32),
            "best_resid_raw": torch.empty((0, CHUNK_K), dtype=torch.float32),
            "improvement": torch.empty((0,), dtype=torch.float32),
            "criticality": torch.empty((0,), dtype=torch.float32),
            "route_idx": torch.empty((0,), dtype=torch.long),
            "chunk_idx": torch.empty((0,), dtype=torch.long),
            "step_idx": torch.empty((0,), dtype=torch.long),
            "csv_file": [],
            "meta": {
                "chunk_k": CHUNK_K,
                "horizon_h": HORIZON_H,
                "probe_m": PROBE_M,
                "delta_scale": DELTA_SCALE,
                "use_expected": USE_EXPECTED,
            },
        }
    else:
        obs = torch.stack([r["obs"].cpu() for r in improved], dim=0)
        base_chunk = torch.stack([r["base_chunk_raw"].cpu() for r in improved], dim=0)
        best_chunk = torch.stack([r["best_chunk_raw"].cpu() for r in improved], dim=0)
        payload = {
            "obs": obs,
            "base_chunk_raw": base_chunk,
            "best_chunk_raw": best_chunk,
            "best_resid_raw": best_chunk - base_chunk,
            "improvement": torch.tensor([r["improvement"] for r in improved], dtype=torch.float32),
            "criticality": torch.tensor([r["criticality"] for r in improved], dtype=torch.float32),
            "route_idx": torch.tensor([r["route_idx"] for r in improved], dtype=torch.long),
            "chunk_idx": torch.tensor([r["chunk_idx"] for r in improved], dtype=torch.long),
            "step_idx": torch.tensor([r["step_idx"] for r in improved], dtype=torch.long),
            "csv_file": [r["csv_file"] for r in improved],
            "meta": {
                "chunk_k": CHUNK_K,
                "horizon_h": HORIZON_H,
                "probe_m": PROBE_M,
                "delta_scale": DELTA_SCALE,
                "use_expected": USE_EXPECTED,
            },
        }
    torch.save(payload, OUT_PT)
    return len(improved)


def main():
    t0 = time.time()
    print(
        f"exp067 probe  routes={ROUTES}  topk_per_route={TOPK_PER_ROUTE}  probe_m={PROBE_M}"
        f"  chunk_k={CHUNK_K}  horizon_h={HORIZON_H}  use_expected={'on' if USE_EXPECTED else 'off'}"
    )
    print(f"base_model={BASE_PT}")

    base_actor = load_base_actor()
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)

    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_files = choose_csvs(all_csv)
    csv_cache = CSVCache(csv_files)

    base_sim, future, costs = run_base_rollout(csv_files, base_actor, mdl_path, ort_sess, csv_cache)
    base_costs = np.asarray(costs["total_cost"], dtype=np.float64)
    print(
        f"base deterministic rollout: mean={base_costs.mean():.3f} std={base_costs.std():.3f}"
    )

    windows = route_boundary_windows(base_sim, future, csv_files)
    print(f"selected windows: {len(windows)}")

    rows = []
    sanity_deltas = []
    for i, win in enumerate(windows, start=1):
        out = evaluate_window(
            csv_file=win["csv_file"],
            csv_cache=csv_cache,
            base_actor=base_actor,
            mdl_path=mdl_path,
            ort_session=ort_sess,
            base_sim=base_sim,
            route_idx=win["route_idx"],
            step_idx=win["step_idx"],
            base_chunk_raw=win["base_chunk_raw"].to(DEV),
        )

        route_idx = win["route_idx"]
        step_idx = win["step_idx"]
        end_step = min(step_idx + HORIZON_H, COST_END_IDX)
        pred = base_sim.current_lataccel_history[route_idx, step_idx:end_step].float()
        prev = base_sim.current_lataccel_history[route_idx, step_idx - 1 : end_step - 1].float()
        tgt = base_sim.data_gpu["target_lataccel"][route_idx, step_idx:end_step].float()
        base_h_cost = float(
            (((tgt - pred).square() * (100 * LAT_ACCEL_COST_MULTIPLIER)) + (((pred - prev) / DEL_T).square() * 100))
            .sum()
            .item()
        )
        sanity_deltas.append(abs(base_h_cost - out["base_cost"]))

        row = {
            "csv_file": win["csv_file"],
            "route_idx": win["route_idx"],
            "chunk_idx": win["chunk_idx"],
            "step_idx": win["step_idx"],
            "criticality": win["criticality"],
            "obs": out["obs"],
            "base_chunk_raw": out["base_chunk_raw"],
            "best_chunk_raw": out["best_chunk_raw"],
            "base_cost": out["base_cost"],
            "best_cost": out["best_cost"],
            "improvement": out["improvement"],
            "best_idx": out["best_idx"],
            "best_mean_abs_resid": out["best_mean_abs_resid"],
            "best_max_abs_resid": out["best_max_abs_resid"],
            "base_h_cost_ref": base_h_cost,
        }
        rows.append(row)
        print(
            f"[{i:3d}/{len(windows):3d}] route={row['route_idx']:3d} step={row['step_idx']:3d}"
            f" crit={row['criticality']:.3f}  Δ={row['improvement']:+7.3f}"
            f"  best={row['best_cost']:.3f}  base={row['base_cost']:.3f}"
        )

    win_bank_count = save_win_bank(rows) if SAVE_PT else 0

    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "csv_file": r["csv_file"],
                "route_idx": r["route_idx"],
                "chunk_idx": r["chunk_idx"],
                "step_idx": r["step_idx"],
                "criticality": r["criticality"],
                "base_cost": r["base_cost"],
                "best_cost": r["best_cost"],
                "improvement": r["improvement"],
                "best_idx": r["best_idx"],
                "best_mean_abs_resid": r["best_mean_abs_resid"],
                "best_max_abs_resid": r["best_max_abs_resid"],
                "base_h_cost_ref": r["base_h_cost_ref"],
            }
        )
    df = pd.DataFrame(csv_rows).sort_values(["improvement", "criticality"], ascending=[False, False])
    summary = write_summary(df, base_costs)
    print(summary)
    print(f"candidate-0 base-horizon sanity max_abs_delta={max(sanity_deltas) if sanity_deltas else 0.0:.6f}")
    if SAVE_PT:
        print(f"saved improved wins: {win_bank_count}")

    if SAVE_CSV:
        df.to_csv(OUT_CSV, index=False)
        print(f"saved: {OUT_CSV}")
        print(f"saved: {OUT_SUMMARY}")
    if SAVE_PT:
        print(f"saved: {OUT_PT}")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
