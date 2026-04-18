"""Diagnose the temp gap: are the temp=0.1 actions robust to temp=0.8 noise?

Uses exp055's rollout directly. Patches BatchedPhysicsModel to support temperature.
"""

import os, sys, torch, numpy as np
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
from tinyphysics import CONTROL_START_IDX, COST_END_IDX
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    STEER_RANGE,
    DEL_T,
)
import torch.nn.functional as F

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX


def rollout_with_temp(
    csv_files, ac, mdl_path, ort_session, csv_cache, ds, sim_temp, replay_actions=None
):
    """
    Run the policy or replay stored actions at a given sim temperature.
    If replay_actions is not None, those are replayed instead of running the policy.
    Returns (costs, stored_actions).
    """
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    sim.sim_model.sim_temperature = sim_temp

    N = sim.N
    dg = sim.data_gpu

    if replay_actions is not None:
        # Pure replay mode
        def ctrl(step_idx, sim_ref):
            if step_idx < CONTROL_START_IDX:
                return torch.zeros(N, dtype=torch.float64, device="cuda")
            ci = step_idx - CONTROL_START_IDX
            if ci >= N_CTRL:
                return torch.zeros(N, dtype=torch.float64, device="cuda")
            return replay_actions[:, ci].double()

        costs = sim.rollout(ctrl)["total_cost"]
        return costs, replay_actions

    # Policy mode — copied exactly from exp055 rollout (deterministic)
    future = _precompute_future_windows(dg)
    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1
    stored = torch.zeros((N, N_CTRL), dtype=torch.float64, device="cuda")

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_error = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_error
        ei = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        fill_obs(
            obs_buf,
            target.float(),
            cur32,
            dg["roll_lataccel"][:, step_idx].float(),
            dg["v_ego"][:, step_idx].float(),
            dg["a_ego"][:, step_idx].float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )

        with torch.inference_mode():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw_policy = 2.0 * a_p / (a_p + b_p) - 1.0

        delta = raw_policy.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        ci = step_idx - CONTROL_START_IDX
        if ci < N_CTRL:
            stored[:, ci] = action

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, stored


def main():
    N_ROUTES = int(os.getenv("N_ROUTES", "10"))

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    # Sanity: policy @ default temp (should match test_baseline ~36)
    print("Sanity: policy @ temp=0.8 (should be ~36) ...")
    costs_S, _ = rollout_with_temp(all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.8)
    print(f"   mean={np.mean(costs_S):.1f}")

    # A) Policy @ temp=0.1
    print("A) Policy @ temp=0.1 ...")
    costs_A, actions_A = rollout_with_temp(
        all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.1
    )
    print(f"   mean={np.mean(costs_A):.1f}")

    # Sanity: replay A @ temp=0.1 (must match A)
    print("   replay A @ 0.1 ...")
    costs_A2, _ = rollout_with_temp(
        all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.1, replay_actions=actions_A
    )
    print(
        f"   replay={np.mean(costs_A2):.1f}  Δ={abs(np.mean(costs_A2) - np.mean(costs_A)):.2f}"
    )

    # B) Replay A's actions @ temp=0.8
    print("B) Replay temp=0.1 actions @ temp=0.8 ...")
    costs_B, _ = rollout_with_temp(
        all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.8, replay_actions=actions_A
    )
    print(f"   mean={np.mean(costs_B):.1f}")

    # C) Policy @ temp=0.8
    print("C) Policy @ temp=0.8 ...")
    costs_C, actions_C = rollout_with_temp(
        all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.8
    )
    print(f"   mean={np.mean(costs_C):.1f}")

    # D) Replay C's actions @ temp=0.1
    print("D) Replay temp=0.8 actions @ temp=0.1 ...")
    costs_D, _ = rollout_with_temp(
        all_csv, ac, mdl_path, ort_sess, csv_cache, ds, 0.1, replay_actions=actions_C
    )
    print(f"   mean={np.mean(costs_D):.1f}")

    print(
        f"\n{'Route':<12s}  {'S:π@0.8':>8s}  {'A:π@0.1':>8s}  {'B:A→0.8':>8s}  {'C:π@0.8':>8s}  {'D:C→0.1':>8s}"
    )
    print("-" * 64)
    for i in range(len(all_csv)):
        print(
            f"{all_csv[i].name:<12s}  {costs_S[i]:8.1f}  {costs_A[i]:8.1f}  {costs_B[i]:8.1f}  {costs_C[i]:8.1f}  {costs_D[i]:8.1f}"
        )
    print(
        f"\n{'Mean':<12s}  {np.mean(costs_S):8.1f}  {np.mean(costs_A):8.1f}  {np.mean(costs_B):8.1f}  {np.mean(costs_C):8.1f}  {np.mean(costs_D):8.1f}"
    )

    gap_AC = np.mean(costs_C) - np.mean(costs_A)
    if abs(gap_AC) > 0.1:
        gap_AB = np.mean(costs_B) - np.mean(costs_A)
        gap_BC = np.mean(costs_C) - np.mean(costs_B)
        print(
            f"\n  A→B (noise degrades good actions):    {gap_AB:+.1f}  ({gap_AB / gap_AC * 100:.0f}%)"
        )
        print(
            f"  B→C (policy adapts to noise):         {gap_BC:+.1f}  ({gap_BC / gap_AC * 100:.0f}%)"
        )
        print(f"  Total A→C:                            {gap_AC:+.1f}")


if __name__ == "__main__":
    main()
