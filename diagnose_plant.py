#!/usr/bin/env python3
"""Probe the ONNX plant during warmup (steps 20-99) where ground-truth steer exists.

For each step we call the model with return_expected=True and capture:
  - sampled prediction (what the stochastic plant would output)
  - expected prediction (probability-weighted mean over 1024 bins)
  - ground truth target_lataccel

The sim's current_lataccel stays pinned to target (normal warmup behavior),
so the model always sees clean ground-truth context.
"""

import os
import time
import numpy as np
from pathlib import Path
from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX,
                         MAX_ACC_DELTA, LATACCEL_RANGE, VOCAB_SIZE)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

MDL = Path('models/tinyphysics.onnx')
DATA = Path('data')
N_SEGS = 2000

def main():
    csv_files = sorted(DATA.glob('*.csv'))[:N_SEGS]
    print(f"Loading {len(csv_files)} segments...")

    ort_sess = make_ort_session(str(MDL))
    cache = CSVCache(csv_files)
    data, rng = cache.slice(csv_files)

    sim = BatchedSimulator(str(MDL), ort_session=ort_sess,
                           cached_data=data, cached_rng=rng)
    sim.compute_expected = True

    N, T = sim.N, sim.T
    CL = CONTEXT_LENGTH
    gpu = sim._gpu

    if gpu:
        import torch
        dg = sim.data_gpu

    n_warmup = CONTROL_START_IDX - CL  # 80 steps
    sampled  = np.zeros((N, n_warmup), np.float64)
    expected = np.zeros((N, n_warmup), np.float64)
    target   = np.zeros((N, n_warmup), np.float64)
    v_ego_arr = np.zeros((N, n_warmup), np.float64)
    steer_arr = np.zeros((N, n_warmup), np.float64)

    print(f"Running {N} segments × {n_warmup} warmup steps (gpu={gpu})...")
    t0 = time.time()

    for step_idx in range(CL, CONTROL_START_IDX):
        i = step_idx - CL
        h = sim._hist_len

        # Write state into sim history
        if gpu:
            sim.state_history[:, h, 0] = dg['roll_lataccel'][:, step_idx]
            sim.state_history[:, h, 1] = dg['v_ego'][:, step_idx]
            sim.state_history[:, h, 2] = dg['a_ego'][:, step_idx]
            actions = dg['steer_command'][:, step_idx]
        else:
            sim.state_history[:, h, 0] = data['roll_lataccel'][:, step_idx]
            sim.state_history[:, h, 1] = data['v_ego'][:, step_idx]
            sim.state_history[:, h, 2] = data['a_ego'][:, step_idx]
            actions = data['steer_command'][:, step_idx]

        # Write steer into action history
        sim.control_step(step_idx, actions)

        # Call model directly with return_expected=True
        rng_idx = step_idx - CL
        if gpu:
            rng_u = sim._rng_all_gpu[rng_idx]
            result = sim.sim_model.get_current_lataccel(
                sim_states=sim.state_history[:, h-CL+1:h+1, :],
                actions=sim.action_history[:, h-CL+1:h+1],
                past_preds=sim.current_lataccel_history[:, h-CL:h],
                rng_u=rng_u,
                return_expected=True,
            )
            pred_s, pred_e = result
            # Clamp sampled (same as sim_step)
            pred_s = torch.clamp(pred_s,
                                 sim.current_lataccel - MAX_ACC_DELTA,
                                 sim.current_lataccel + MAX_ACC_DELTA)
            sampled[: , i] = pred_s.cpu().numpy()
            expected[:, i] = pred_e.cpu().numpy()
            tgt = dg['target_lataccel'][:, step_idx]
            target[:, i] = tgt.cpu().numpy()
            v_ego_arr[:, i] = dg['v_ego'][:, step_idx].cpu().numpy()
            steer_arr[:, i] = dg['steer_command'][:, step_idx].cpu().numpy()

            # Pin current_lataccel to target (warmup behavior)
            sim.current_lataccel = tgt.clone()
            sim.current_lataccel_history[:, h] = sim.current_lataccel
        else:
            rng_u = sim._rng_all[rng_idx]
            result = sim.sim_model.get_current_lataccel(
                sim_states=sim.state_history[:, h-CL+1:h+1, :],
                actions=sim.action_history[:, h-CL+1:h+1],
                past_preds=sim.current_lataccel_history[:, h-CL:h],
                rng_u=rng_u,
                return_expected=True,
            )
            pred_s, pred_e = result
            pred_s = np.clip(pred_s,
                             sim.current_lataccel - MAX_ACC_DELTA,
                             sim.current_lataccel + MAX_ACC_DELTA)
            sampled[:, i] = pred_s
            expected[:, i] = pred_e
            tgt = data['target_lataccel'][:, step_idx]
            target[:, i] = tgt
            v_ego_arr[:, i] = data['v_ego'][:, step_idx]
            steer_arr[:, i] = data['steer_command'][:, step_idx]

            sim.current_lataccel = tgt.copy()
            sim.current_lataccel_history[:, h] = sim.current_lataccel

        sim._hist_len += 1

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s\n")

    # ── Analysis ──────────────────────────────────────────────────────
    noise     = sampled - expected        # sampling noise
    bias      = expected - target         # model systematic error
    total_err = sampled - target          # what the plant actually does vs target

    print("=" * 70)
    print(f"PLANT DIAGNOSTIC  ({N} segments × {n_warmup} steps = {N*n_warmup} samples)")
    print(f"Model sees perfect ground-truth context at every step.")
    print("=" * 70)

    print(f"\n  Model bias (E[pred] - target):")
    print(f"    mean  = {np.mean(bias):+.5f}")
    print(f"    |err| = {np.mean(np.abs(bias)):.5f}")
    print(f"    RMSE  = {np.sqrt(np.mean(bias**2)):.5f}")

    print(f"\n  Sampling noise (sampled - E[pred]):")
    print(f"    mean  = {np.mean(noise):+.5f}")
    print(f"    |err| = {np.mean(np.abs(noise)):.5f}")
    print(f"    RMSE  = {np.sqrt(np.mean(noise**2)):.5f}")

    print(f"\n  Total plant error (sampled - target):")
    print(f"    mean  = {np.mean(total_err):+.5f}")
    print(f"    |err| = {np.mean(np.abs(total_err)):.5f}")
    print(f"    RMSE  = {np.sqrt(np.mean(total_err**2)):.5f}")

    # Variance decomposition
    vt = np.var(total_err)
    vb = np.var(bias)
    vn = np.var(noise)
    cov = np.mean(bias * noise) - np.mean(bias) * np.mean(noise)
    print(f"\n  Var decomposition of total plant error:")
    print(f"    Var(total)  = {vt:.6f}")
    print(f"    Var(bias)   = {vb:.6f}  ({100*vb/vt:.1f}%)")
    print(f"    Var(noise)  = {vn:.6f}  ({100*vn/vt:.1f}%)")
    print(f"    2·Cov(b,n)  = {2*cov:.6f}  ({100*2*cov/vt:.1f}%)")

    # ── Noise distribution ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("NOISE DISTRIBUTION (|sampled - expected|)")
    print("=" * 70)
    an = np.abs(noise.flatten())
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d} = {np.percentile(an, p):.5f} m/s²")
    print(f"  Bin width = {10/1024:.5f} m/s²")
    print(f"  MAX_ACC_DELTA = {MAX_ACC_DELTA} m/s²")

    # ── Bias distribution ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BIAS DISTRIBUTION (expected - target)")
    print("=" * 70)
    ab = np.abs(bias.flatten())
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d} = {np.percentile(ab, p):.5f} m/s²")

    # ── Per-step evolution ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-STEP (averaged over segments)")
    print("=" * 70)
    for i in range(0, n_warmup, 10):
        t = CL + i
        b_rmse = np.sqrt(np.mean(bias[:, i]**2))
        n_rmse = np.sqrt(np.mean(noise[:, i]**2))
        t_rmse = np.sqrt(np.mean(total_err[:, i]**2))
        print(f"  step {t:3d}: bias_RMSE={b_rmse:.5f}  noise_RMSE={n_rmse:.5f}  "
              f"total_RMSE={t_rmse:.5f}")

    # ── Speed dependence ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SPEED DEPENDENCE")
    print("=" * 70)
    v_mean = np.mean(v_ego_arr, axis=1)
    for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 35), (35, 45)]:
        mask = (v_mean >= lo) & (v_mean < hi)
        if mask.sum() < 5:
            continue
        b = np.sqrt(np.mean(bias[mask]**2))
        n = np.sqrt(np.mean(noise[mask]**2))
        print(f"  v [{lo:2d}-{hi:2d}] m/s:  n={mask.sum():4d}  "
              f"bias_RMSE={b:.5f}  noise_RMSE={n:.5f}")

    # ── Target magnitude dependence ───────────────────────────────────
    print(f"\n{'='*70}")
    print("TARGET MAGNITUDE DEPENDENCE")
    print("=" * 70)
    tgt_abs_mean = np.mean(np.abs(target), axis=1)
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 5.0)]:
        mask = (tgt_abs_mean >= lo) & (tgt_abs_mean < hi)
        if mask.sum() < 5:
            continue
        b = np.sqrt(np.mean(bias[mask]**2))
        n = np.sqrt(np.mean(noise[mask]**2))
        print(f"  |target| [{lo:.1f}-{hi:.1f}]:  n={mask.sum():4d}  "
              f"bias_RMSE={b:.5f}  noise_RMSE={n:.5f}")

    # ── Key insight ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    bias_rmse = np.sqrt(np.mean(bias**2))
    noise_rmse = np.sqrt(np.mean(noise**2))
    total_rmse = np.sqrt(np.mean(total_err**2))
    print(f"  Under perfect context, the model's per-step error is:")
    print(f"    {total_rmse:.4f} m/s²  total")
    print(f"    {bias_rmse:.4f} m/s²  from model bias (systematic)")
    print(f"    {noise_rmse:.4f} m/s²  from sampling noise (stochastic)")
    noise_frac = vn / vt * 100
    bias_frac = vb / vt * 100
    print(f"  Variance split: {bias_frac:.0f}% bias, {noise_frac:.0f}% noise")
    if noise_frac > 60:
        print(f"  → Dominated by SAMPLING NOISE. Reducing temperature or using")
        print(f"    expected values in training could help significantly.")
    elif bias_frac > 60:
        print(f"  → Dominated by MODEL BIAS. The model systematically mispredicts.")
        print(f"    Controller must learn to compensate for this bias.")
    else:
        print(f"  → Mixed. Both bias and noise contribute meaningfully.")

if __name__ == '__main__':
    main()
