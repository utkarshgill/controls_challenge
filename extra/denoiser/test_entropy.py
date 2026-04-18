"""Test: does the physics model's output entropy correlate with tracking cost?

If high-entropy steps predict high cost, then a policy that minimizes
entropy (makes the physics model confident) would reduce noise.

Runs the policy at temp=0.8 with compute_expected=True, records per-step
entropy and tracking error.
"""

import os, sys, torch, numpy as np
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T
from tinyphysics_batched import (
    BatchedSimulator,
    CSVCache,
    make_ort_session,
    LATACCEL_RANGE,
    VOCAB_SIZE,
)

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH
BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=DEV)
N_ROUTES = int(os.getenv("N_ROUTES", "100"))


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Run sim with zero actions, recording per-step entropy
    data, rng = csv_cache.slice(all_csv)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    phys = sim.sim_model

    def ctrl(step_idx, sim_ref):
        return torch.zeros(N, dtype=torch.float64, device="cuda")

    sim.rollout(ctrl)

    # Now recompute entropies by replaying through stored histories
    sampled_la = sim.current_lataccel_history.float()
    action_hist = sim.action_history.float()
    state_hist = sim.state_history.float()

    entropies = torch.zeros((N, N_CTRL), device="cuda")
    errors = torch.zeros((N, N_CTRL), device="cuda")
    expected_vals = torch.zeros((N, N_CTRL), device="cuda")

    # Use a SEPARATE physics model instance to avoid buffer conflicts
    from tinyphysics_batched import BatchedPhysicsModel

    phys2 = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)

    print(f"Computing per-step entropy for {N} routes, {N_CTRL} steps...")
    for ci in range(N_CTRL):
        t = CONTROL_START_IDX + ci
        h = CL + t

        sim_states = state_hist[:, h - CL + 1 : h + 1, :]
        sim_actions = action_hist[:, h - CL + 1 : h + 1]
        sim_preds = sampled_la[:, h - CL : h]

        states_input = torch.zeros((N, CL, 4), dtype=torch.float32, device="cuda")
        states_input[:, :, 0] = sim_actions
        states_input[:, :, 1:] = sim_states

        clamped = sim_preds.clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1]).float()
        tokens = (
            torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()
        )

        phys2._predict_gpu(
            {"states": states_input, "tokens": tokens}, temperature=0.8, rng_u=None
        )
        probs = phys2._last_probs_gpu  # (N, 1024)

        # Entropy: -sum(p * log(p))
        log_probs = torch.log(probs.clamp(min=1e-10))
        ent = -(probs * log_probs).sum(dim=-1)  # (N,)
        entropies[:, ci] = ent

        # Expected value
        exp_val = (probs * BINS.float().unsqueeze(0)).sum(dim=-1)
        expected_vals[:, ci] = exp_val

        # Tracking error
        target = dg["target_lataccel"][:, t].float()
        sampled = sampled_la[:, h]
        errors[:, ci] = (target - sampled).abs()

    # Analysis
    ent_flat = entropies.cpu().numpy().flatten()
    err_flat = errors.cpu().numpy().flatten()

    # Noise: |sampled - expected|
    noise_flat = (
        (
            sampled_la[:, CL + CONTROL_START_IDX : CL + COST_END_IDX].float()
            - expected_vals
        )
        .abs()
        .cpu()
        .numpy()
        .flatten()
    )

    print(f"\n  Per-step statistics ({len(ent_flat)} samples):")
    print(
        f"    Entropy:  mean={np.mean(ent_flat):.3f}  std={np.std(ent_flat):.3f}  "
        f"min={np.min(ent_flat):.3f}  max={np.max(ent_flat):.3f}"
    )
    print(f"    |error|:  mean={np.mean(err_flat):.4f}  std={np.std(err_flat):.4f}")
    print(f"    |noise|:  mean={np.mean(noise_flat):.4f}  std={np.std(noise_flat):.4f}")

    # Correlation
    corr_ent_err = np.corrcoef(ent_flat, err_flat)[0, 1]
    corr_ent_noise = np.corrcoef(ent_flat, noise_flat)[0, 1]
    print(f"\n  Correlations:")
    print(f"    entropy vs |tracking error|:  r={corr_ent_err:.4f}")
    print(f"    entropy vs |sampling noise|:  r={corr_ent_noise:.4f}")

    # Binned analysis: split steps into low/high entropy
    median_ent = np.median(ent_flat)
    lo_mask = ent_flat < median_ent
    hi_mask = ent_flat >= median_ent
    print(f"\n  Binned by entropy (median={median_ent:.3f}):")
    print(
        f"    Low entropy steps:  mean|error|={np.mean(err_flat[lo_mask]):.4f}  "
        f"mean|noise|={np.mean(noise_flat[lo_mask]):.4f}"
    )
    print(
        f"    High entropy steps: mean|error|={np.mean(err_flat[hi_mask]):.4f}  "
        f"mean|noise|={np.mean(noise_flat[hi_mask]):.4f}"
    )
    print(
        f"    Ratio (high/low):   error={np.mean(err_flat[hi_mask]) / np.mean(err_flat[lo_mask]):.2f}x  "
        f"noise={np.mean(noise_flat[hi_mask]) / np.mean(noise_flat[lo_mask]):.2f}x"
    )

    # Per-route: does mean entropy predict route cost?
    route_entropy = entropies.mean(dim=1).cpu().numpy()
    route_cost = sim.rollout_costs if hasattr(sim, "rollout_costs") else None

    # Recompute route costs
    route_err = (
        errors.pow(2).mean(dim=1).cpu().numpy() * 100 * 50
    )  # approx lataccel cost
    corr_route = np.corrcoef(route_entropy, route_err)[0, 1]
    print(f"\n  Per-route (N={N}):")
    print(f"    entropy vs approx cost: r={corr_route:.4f}")

    # Key question: how much does entropy vary by ACTION?
    # Run again with different constant actions and measure entropy
    print(f"\n  Testing entropy vs constant steer action...")
    test_actions = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]
    phys3 = BatchedPhysicsModel(str(mdl_path), ort_session=ort_sess)
    for act_val in test_actions:
        data2, rng2 = csv_cache.slice(all_csv[:10])  # just 10 routes
        sim2 = BatchedSimulator(
            str(mdl_path), ort_session=ort_sess, cached_data=data2, cached_rng=rng2
        )
        N2 = sim2.N

        def ctrl2(step_idx, sim_ref, _a=act_val):
            return torch.full((N2,), _a, dtype=torch.float64, device="cuda")

        sim2.rollout(ctrl2)

        # Recompute entropies for this action
        sa2 = sim2.current_lataccel_history.float()
        ah2 = sim2.action_history.float()
        sh2 = sim2.state_history.float()
        ent_sum = 0.0
        n_steps = 0
        for ci in range(0, N_CTRL, 20):  # sample every 20 steps
            t = CONTROL_START_IDX + ci
            h = CL + t
            si2 = sh2[:, h - CL + 1 : h + 1, :]
            ai2 = ah2[:, h - CL + 1 : h + 1]
            pp2 = sa2[:, h - CL : h]
            st2 = torch.zeros((N2, CL, 4), dtype=torch.float32, device="cuda")
            st2[:, :, 0] = ai2
            st2[:, :, 1:] = si2
            cl2 = pp2.clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1]).float()
            tk2 = (
                torch.bucketize(cl2, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()
            )
            phys3._predict_gpu(
                {"states": st2, "tokens": tk2}, temperature=0.8, rng_u=None
            )
            p2 = phys3._last_probs_gpu
            e2 = -(p2 * torch.log(p2.clamp(min=1e-10))).sum(dim=-1)
            ent_sum += e2.mean().item()
            n_steps += 1

        mean_ent = ent_sum / n_steps
        cost2 = sim2.rollout(ctrl2) if False else None  # skip re-rollout
        print(f"    steer={act_val:+.1f}  mean_entropy={mean_ent:.3f}")


if __name__ == "__main__":
    main()
