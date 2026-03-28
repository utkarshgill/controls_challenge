#!/usr/bin/env python3
"""exp107 — Evolution Strategies on policy weights

Instead of CEM on actions (per-route), do ES on policy WEIGHTS (across routes).
Perturb the policy's output layer + last hidden layer, evaluate across N routes,
update toward better perturbations.

This finds a GENERALIZING controller, not a lookup table.
"""

import numpy as np, os, sys, time, torch, torch.nn.functional as F
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

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
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
)

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

N_ROUTES = int(os.getenv("N_ROUTES", "100"))
ES_POP = int(os.getenv("ES_POP", "32"))  # population size (pairs: 2*ES_POP evals)
ES_ITERS = int(os.getenv("ES_ITERS", "200"))
ES_SIGMA = float(os.getenv("ES_SIGMA", "0.02"))
ES_LR = float(os.getenv("ES_LR", "0.01"))
PERTURB_LAYERS = int(
    os.getenv("PERTURB_LAYERS", "2")
)  # how many layers from the end to perturb


def evaluate_policy(ac, csv_files, mdl_path, ort_sess, csv_cache, ds):
    """Run policy deterministically on routes, return mean cost."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)
    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum
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
        with torch.no_grad():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    return np.mean(costs), costs


def get_perturb_params(ac):
    """Get the parameters to perturb (last N layers of actor)."""
    actor_layers = list(ac.actor.children())
    # Actor is Sequential: [Linear, ReLU, Linear, ReLU, ..., Linear]
    # Get the last PERTURB_LAYERS linear layers
    linears = [l for l in actor_layers if isinstance(l, torch.nn.Linear)]
    params = []
    for l in linears[-PERTURB_LAYERS:]:
        params.append(l.weight)
        params.append(l.bias)
    return params


def flatten_params(params):
    """Flatten list of parameter tensors into one vector."""
    return torch.cat([p.data.reshape(-1) for p in params])


def unflatten_params(flat, params):
    """Unflatten vector back into parameter tensors."""
    idx = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].reshape(p.shape))
        idx += n


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Load policy
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    params = get_perturb_params(ac)
    n_params = sum(p.numel() for p in params)
    theta = flatten_params(params)
    print(f"exp107 — ES on policy weights")
    print(f"  Perturbing last {PERTURB_LAYERS} layers: {n_params} params")
    print(f"  Population: {ES_POP} (×2 antithetic), σ={ES_SIGMA}, lr={ES_LR}")
    print(f"  Routes: {N_ROUTES}, iters: {ES_ITERS}")

    # Baseline
    mean_cost, _ = evaluate_policy(ac, all_csv, mdl_path, ort_sess, csv_cache, ds)
    best_cost = mean_cost
    best_theta = theta.clone()
    print(f"  Baseline: {mean_cost:.1f}")

    for it in range(ES_ITERS):
        t0 = time.time()

        # Generate perturbation directions
        eps = torch.randn(ES_POP, n_params, device="cuda")

        # Evaluate positive and negative perturbations
        costs_pos = np.zeros(ES_POP)
        costs_neg = np.zeros(ES_POP)

        for j in range(ES_POP):
            # Positive perturbation
            unflatten_params(theta + ES_SIGMA * eps[j], params)
            c, _ = evaluate_policy(ac, all_csv, mdl_path, ort_sess, csv_cache, ds)
            costs_pos[j] = c

            # Negative perturbation (antithetic)
            unflatten_params(theta - ES_SIGMA * eps[j], params)
            c, _ = evaluate_policy(ac, all_csv, mdl_path, ort_sess, csv_cache, ds)
            costs_neg[j] = c

        # ES gradient estimate: weighted sum of perturbation directions
        # We want to MINIMIZE cost, so negate
        rewards = -(costs_pos - costs_neg) / (2 * ES_SIGMA)
        # Normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update theta
        grad = (
            eps.T @ torch.tensor(rewards, device="cuda", dtype=torch.float32)
        ) / ES_POP
        theta = theta + ES_LR * grad

        # Apply updated params
        unflatten_params(theta, params)
        mean_cost, _ = evaluate_policy(ac, all_csv, mdl_path, ort_sess, csv_cache, ds)

        if mean_cost < best_cost:
            best_cost = mean_cost
            best_theta = theta.clone()
            marker = " ★"
        else:
            marker = ""

        dt = time.time() - t0
        print(
            f"  iter {it:3d}  cost={mean_cost:.1f}  best={best_cost:.1f}  "
            f"⏱{dt:.0f}s{marker}"
        )

        # Save periodically
        if (it + 1) % 20 == 0:
            unflatten_params(best_theta, params)
            save_path = Path(__file__).parent / "best_model.pt"
            torch.save(
                {"ac": ac.state_dict(), "delta_scale": ds, "cost": best_cost}, save_path
            )
            print(f"    saved to {save_path}")
            # Restore current theta for continued optimization
            unflatten_params(theta, params)

    # Final: apply best and evaluate
    unflatten_params(best_theta, params)
    final_cost, final_costs = evaluate_policy(
        ac, all_csv, mdl_path, ort_sess, csv_cache, ds
    )
    print(f"\nFinal: {final_cost:.1f}")
    save_path = Path(__file__).parent / "best_model.pt"
    torch.save(
        {"ac": ac.state_dict(), "delta_scale": ds, "cost": final_cost}, save_path
    )


if __name__ == "__main__":
    main()
