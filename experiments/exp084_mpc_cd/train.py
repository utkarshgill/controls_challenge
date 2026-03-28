# exp084 — DAgger loop: deep MPC search (frozen policy) → BC distill → repeat
#
# Each iteration:
#   Phase 1: Freeze policy. Run 128-candidate, 15-iteration CEM with smooth
#            perturbations on R routes using the GPU batched sim. ~15 min.
#   Phase 2: BC-distill improved actions into policy. ~1 min.
#   Phase 3: Evaluate on stochastic sim. Save if improved.
#   Repeat from Phase 1 with updated policy.
#
# This separates search (which needs many candidates and iterations)
# from learning (which needs stable targets).

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
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
    FUTURE_K,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

# ── MPC search config ─────────────────────────────────────────
MPC_K = int(os.getenv("MPC_K", "128"))
N_BASIS = int(os.getenv("N_BASIS", "40"))
BASIS_WIDTH = int(os.getenv("BASIS_WIDTH", "20"))
PERTURB_STD = float(os.getenv("PERTURB_STD", "0.02"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "15"))
ELITE_FRAC = float(os.getenv("ELITE_FRAC", "0.125"))
SEARCH_ROUTES = int(os.getenv("SEARCH_ROUTES", "50"))

# ── BC config ─────────────────────────────────────────────────
BC_LR = float(os.getenv("BC_LR", "3e-5"))
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "3"))
BC_ANCHOR = float(os.getenv("BC_ANCHOR", "0.5"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "0.5"))
MINI_BS = int(os.getenv("MINI_BS", "25000"))

# ── runtime ───────────────────────────────────────────────────
MAX_ITERS = int(os.getenv("MAX_ITERS", "100"))
EVAL_N = 100

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ══════════════════════════════════════════════════════════════
#  Smooth basis
# ══════════════════════════════════════════════════════════════


def make_basis():
    centers = torch.linspace(0, N_CTRL - 1, N_BASIS, device=DEV)
    t = torch.arange(N_CTRL, dtype=torch.float32, device=DEV)
    basis = torch.exp(
        -0.5 * ((t.unsqueeze(0) - centers.unsqueeze(1)) / BASIS_WIDTH) ** 2
    )
    basis = basis / basis.max(dim=1, keepdim=True).values
    return basis


# ══════════════════════════════════════════════════════════════
#  MPC search on R routes × K candidates (all GPU, one sim call)
# ══════════════════════════════════════════════════════════════


def mpc_search(route_files, ac, basis, mdl_path, ort_session, csv_cache, ds):
    """Deep CEM search: R routes × K candidates × CEM_ITERS iterations.
    Returns dict with winning obs, raw, and stats."""
    R = len(route_files)

    # CEM state per route: mean and std of perturbation coefficients
    cem_mean = torch.zeros(R, N_BASIS, device=DEV)
    cem_std = torch.full((R, N_BASIS), PERTURB_STD, device=DEV)

    best_cost_per_route = torch.full((R,), float("inf"), device=DEV)
    best_obs = None  # will be set on first improvement
    best_raw = None

    for cem_iter in range(CEM_ITERS):
        # Sample coefficients: (R, K, N_BASIS)
        coeffs = torch.randn(R, MPC_K, N_BASIS, device=DEV) * cem_std.unsqueeze(
            1
        ) + cem_mean.unsqueeze(1)
        coeffs[:, 0, :] = cem_mean  # candidate 0 = current CEM mean

        # Convert to perturbations: (R*K, N_CTRL)
        flat_coeffs = coeffs.view(R * MPC_K, N_BASIS)
        perturbations = flat_coeffs @ basis  # (R*K, N_CTRL)

        # Tile routes: [r0c0, r0c1, ..., r0cK, r1c0, ...]
        csv_tiled = [f for f in route_files for _ in range(MPC_K)]
        data, rng = csv_cache.slice(csv_tiled)
        sim = BatchedSimulator(
            str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
        )
        sim.use_expected = True
        N = sim.N
        dg = sim.data_gpu
        future = _precompute_future_windows(dg)

        h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
        h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
        err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
        obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

        S = N_CTRL
        all_obs = torch.empty((S, N, OBS_DIM), dtype=torch.float32, device="cuda")
        all_raw = torch.empty((S, N), dtype=torch.float32, device="cuda")
        si = 0
        hist_head = HIST_LEN - 1

        def ctrl(step_idx, sim_ref):
            nonlocal si, hist_head, err_sum
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
            with torch.inference_mode():
                logits = ac.actor(obs_buf)
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            policy_raw = 2.0 * a_p / (a_p + b_p) - 1.0
            ci = step_idx - CONTROL_START_IDX
            if ci < N_CTRL:
                raw = (policy_raw + perturbations[:, ci]).clamp(-1.0, 1.0)
            else:
                raw = policy_raw.clamp(-1.0, 1.0)
            delta = raw.to(h_act.dtype) * ds
            action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
            h_act[:, next_head] = action
            h_act32[:, next_head] = action.float()
            h_lat[:, next_head] = cur32
            hist_head = next_head
            if step_idx < COST_END_IDX:
                all_obs[si] = obs_buf
                all_raw[si] = raw.float()
                si += 1
            return action

        costs = sim.rollout(ctrl)["total_cost"]  # (R*K,)
        S_actual = si

        # Reshape costs and update CEM
        costs_t = torch.from_numpy(costs).to(DEV).view(R, MPC_K)
        n_elite = max(1, int(MPC_K * ELITE_FRAC))
        _, elite_idx = costs_t.topk(n_elite, dim=1, largest=False)  # (R, n_elite)

        # Gather elite coefficients
        elite_coeffs = torch.gather(
            coeffs, 1, elite_idx.unsqueeze(-1).expand(-1, -1, N_BASIS)
        )
        cem_mean = elite_coeffs.mean(dim=1)
        cem_std = elite_coeffs.std(dim=1).clamp(min=0.001)

        # Track best per route (across all CEM iterations)
        iter_best_cost, iter_best_idx = costs_t.min(dim=1)
        iter_best_cost = iter_best_cost.float()
        improved = iter_best_cost < best_cost_per_route
        if improved.any():
            best_cost_per_route[improved] = iter_best_cost[improved]
            # Store best obs/raw
            flat_best_idx = torch.arange(R, device=DEV) * MPC_K + iter_best_idx
            if best_obs is None:
                best_obs = all_obs[:S_actual, :, :].clone()
                best_raw = all_raw[:S_actual, :].clone()
                best_flat_idx = flat_best_idx.clone()
            best_flat_idx[improved] = flat_best_idx[improved]

    # Extract final best obs/raw
    win_obs = all_obs[:S_actual, best_flat_idx, :]  # (S, R, OBS_DIM)
    win_raw = all_raw[:S_actual, best_flat_idx]  # (S, R)

    # Also get baseline (zero perturbation from first CEM iter)
    base_idx = torch.arange(R, device=DEV) * MPC_K
    base_obs = all_obs[:S_actual, base_idx, :]
    base_raw = all_raw[:S_actual, base_idx]
    baseline_costs = costs_t[:, 0].float()

    improve = (baseline_costs - best_cost_per_route).mean().item()
    frac = (best_cost_per_route < baseline_costs).float().mean().item()

    return dict(
        win_obs=win_obs.permute(1, 0, 2).reshape(-1, OBS_DIM),
        win_raw=win_raw.T.reshape(-1),
        base_obs=base_obs.permute(1, 0, 2).reshape(-1, OBS_DIM),
        base_raw=base_raw.T.reshape(-1),
        baseline_mean=baseline_costs.mean().item(),
        winner_mean=best_cost_per_route.mean().item(),
        improve=improve,
        frac=frac,
    )


# ══════════════════════════════════════════════════════════════
#  BC distillation
# ══════════════════════════════════════════════════════════════


def bc_distill(ac, opt, mpc_data):
    """BC toward winners, anchored to baseline."""
    obs_parts = [mpc_data["win_obs"], mpc_data["base_obs"]]
    raw_parts = [mpc_data["win_raw"], mpc_data["base_raw"]]
    w_win = torch.full((mpc_data["win_obs"].shape[0],), 1.0 - BC_ANCHOR, device=DEV)
    w_base = torch.full((mpc_data["base_obs"].shape[0],), BC_ANCHOR, device=DEV)

    obs = torch.cat(obs_parts)
    raw = torch.cat(raw_parts)
    weights = torch.cat([w_win, w_base])
    x_t = ((raw.unsqueeze(-1) + 1) / 2).clamp(1e-6, 1 - 1e-6)

    total_nll = 0.0
    n_total = 0
    for _ in range(BC_EPOCHS):
        for idx in torch.randperm(len(obs), device=DEV).split(MINI_BS):
            logits = ac.actor(obs[idx])
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            dist = torch.distributions.Beta(a_p, b_p)
            per_nll = -dist.log_prob(x_t[idx].squeeze(-1))
            loss = (per_nll * weights[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP)
            opt.step()
            bs = idx.numel()
            total_nll += per_nll.mean().item() * bs
            n_total += bs
    return total_nll / max(1, n_total)


# ══════════════════════════════════════════════════════════════
#  Eval (deterministic, stochastic physics)
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds):
    data, rng = csv_cache.slice(files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
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
        with torch.inference_mode():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw * ds
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return float(np.mean(costs)), float(np.std(costs))


# ══════════════════════════════════════════════════════════════
#  Train loop
# ══════════════════════════════════════════════════════════════


def train():
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE_MAX))
    print(f"Loaded from {BASE_PT} (Δs={ds})")

    opt = optim.Adam(ac.actor.parameters(), lr=BC_LR)
    basis = make_basis()

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
    best, best_ep = vm, "init"
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")
    print(
        f"\nDAgger: MPC_K={MPC_K} CEM_ITERS={CEM_ITERS} basis={N_BASIS} width={BASIS_WIDTH}"
    )
    print(f"  σ={PERTURB_STD} search_routes={SEARCH_ROUTES}")
    print(f"  BC: lr={BC_LR} epochs={BC_EPOCHS} anchor={BC_ANCHOR}")
    print()

    def save_best():
        torch.save({"ac": ac.state_dict(), "delta_scale": ds}, BEST_PT)

    for iteration in range(MAX_ITERS):
        # Phase 1: MPC search (frozen policy)
        ac.eval()
        t0 = time.time()
        batch = random.sample(all_csv, min(SEARCH_ROUTES, len(all_csv)))
        mpc_data = mpc_search(batch, ac, basis, mdl_path, ort_sess, csv_cache, ds=ds)
        t_search = time.time() - t0

        # Phase 2: BC distill
        ac.train()
        t1 = time.time()
        nll = bc_distill(ac, opt, mpc_data)
        t_bc = time.time() - t1

        # Phase 3: Eval
        ac.eval()
        vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
        mk = ""
        if vm < best:
            best, best_ep = vm, iteration
            save_best()
            mk = " ★"

        print(
            f"I{iteration:3d}  base={mpc_data['baseline_mean']:5.1f}"
            f"  win={mpc_data['winner_mean']:5.1f}"
            f"  Δ={mpc_data['improve']:.1f}  f={mpc_data['frac']:.2f}"
            f"  nll={nll:.3f}  val={vm:5.1f}±{vs:4.1f}{mk}"
            f"  ⏱{t_search:.0f}+{t_bc:.0f}s"
        )

    print(f"\nDone. Best: {best:.1f} (iter {best_ep})")


if __name__ == "__main__":
    train()
