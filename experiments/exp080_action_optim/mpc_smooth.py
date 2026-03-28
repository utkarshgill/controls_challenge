# exp080 — MPC with smooth temporal perturbations + real sim
#
# The policy proposes actions. MPC perturbs them with smooth basis functions
# (Gaussian bumps) and evaluates via the real batched sim. Picks the best.
#
# Why smooth: independent per-step noise always makes things worse because
# it increases jerk. Smooth perturbations shift the timing of the policy's
# actions without adding jerk — which is exactly what anticipatory corrections
# look like.
#
# This is N×K rollouts using existing parallel GPU sim. Nothing fancy.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F
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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

# ── Config ────────────────────────────────────────────────────
MPC_K = int(os.getenv("MPC_K", "128"))
N_BASIS = int(os.getenv("N_BASIS", "20"))
BASIS_WIDTH = int(os.getenv("BASIS_WIDTH", "40"))  # steps, σ of each Gaussian bump
PERTURB_STD = float(
    os.getenv("PERTURB_STD", "0.02")
)  # scale of perturbations in steer space
N_ROUTES = int(os.getenv("N_ROUTES", "100"))
N_ITERS = int(os.getenv("N_ITERS", "5"))  # CEM-like iterations per route batch
ELITE_FRAC = float(os.getenv("ELITE_FRAC", "0.125"))

# ── Policy ────────────────────────────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = 0.25
S_LAT, S_STEER, S_VEGO, S_AEGO, S_ROLL, S_CURV = 5.0, 2.0, 40.0, 4.0, 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "mpc_actions"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ══════════════════════════════════════════════════════════════
#  Smooth basis functions
# ══════════════════════════════════════════════════════════════


def make_basis(n_basis=N_BASIS, width=BASIS_WIDTH, n_steps=N_CTRL):
    """Create smooth Gaussian bump basis functions.
    Returns: (n_basis, n_steps) tensor on DEV.
    """
    centers = torch.linspace(0, n_steps - 1, n_basis, device=DEV)
    t = torch.arange(n_steps, dtype=torch.float32, device=DEV)
    # (n_basis, n_steps): each row is a Gaussian centered at centers[i]
    basis = torch.exp(-0.5 * ((t.unsqueeze(0) - centers.unsqueeze(1)) / width) ** 2)
    # Normalize each basis function to have unit max
    basis = basis / basis.max(dim=1, keepdim=True).values
    return basis


def sample_perturbations(basis, k, std=PERTURB_STD):
    """Sample K smooth perturbation curves.
    basis: (n_basis, n_steps)
    Returns: (K, n_steps) — smooth perturbations in action space
    """
    # Sample coefficients: (K, n_basis)
    coeffs = torch.randn(k, basis.shape[0], device=DEV) * std
    # Linear combination: (K, n_steps)
    return coeffs @ basis


# ══════════════════════════════════════════════════════════════
#  Policy
# ══════════════════════════════════════════════════════════════


class PolicyActor(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)

    def get_delta(self, obs):
        logits = self.actor(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        return 2.0 * a_p / (a_p + b_p) - 1.0


# ══════════════════════════════════════════════════════════════
#  Obs builder
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _w(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        k: _w(dg[k]) for k in ("target_lataccel", "roll_lataccel", "v_ego", "a_ego")
    }


def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale
        return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale


def fill_obs(
    buf,
    target,
    current,
    roll_la,
    v_ego,
    a_ego,
    h_act,
    h_lat,
    hist_head,
    ei,
    future,
    step_idx,
):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    pa = h_act[:, hist_head]
    pa2 = h_act[:, (hist_head - 1) % HIST_LEN]
    pl = h_lat[:, hist_head]
    buf[:, 0] = target / S_LAT
    buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV
    buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO
    buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL
    buf[:, 9] = pa / S_STEER
    buf[:, 10] = ei / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - pl) / DEL_T / S_LAT
    buf[:, 13] = (pa - pa2) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)
    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  Rollout with policy + additive perturbation
# ══════════════════════════════════════════════════════════════


def rollout_with_perturbations(
    csv_file, ac, perturbations, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """Roll out the policy on K copies of a single route, each with a different
    additive perturbation to the raw delta output.

    perturbations: (K, 400) — added to policy's raw delta at each step
    Returns: costs (K,), obs (400, K, OBS_DIM), raw (400, K)
    """
    K = perturbations.shape[0]
    csv_files = [csv_file] * K
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    sim.use_expected = True  # deterministic for fair comparison
    N = sim.N  # = K
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    all_obs = torch.empty((N_CTRL, N, OBS_DIM), dtype=torch.float32, device="cuda")
    all_raw = torch.empty((N_CTRL, N), dtype=torch.float32, device="cuda")

    perturb_gpu = perturbations.to(dtype=torch.float32, device="cuda")  # (K, 400)

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
            policy_raw = ac.get_delta(obs_buf)  # (K,)

        # Add smooth perturbation for this step
        ctrl_idx = step_idx - CONTROL_START_IDX
        if ctrl_idx < N_CTRL:
            raw = (policy_raw + perturb_gpu[:, ctrl_idx]).clamp(-1.0, 1.0)
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

    costs = sim.rollout(ctrl)["total_cost"]
    return costs, all_obs[:si], all_raw[:si]


# ══════════════════════════════════════════════════════════════
#  MPC search for one route: CEM over smooth perturbations
# ══════════════════════════════════════════════════════════════


def mpc_search_route(
    csv_file, ac, basis, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """CEM search over smooth perturbation coefficients for a single route."""
    n_elite = max(1, int(MPC_K * ELITE_FRAC))

    # CEM state
    mean = torch.zeros(N_BASIS, device=DEV)
    std = torch.full((N_BASIS,), PERTURB_STD, device=DEV)

    best_cost = float("inf")
    best_obs = None
    best_raw = None

    for it in range(N_ITERS):
        # Sample coefficients
        coeffs = torch.randn(MPC_K, N_BASIS, device=DEV) * std.unsqueeze(
            0
        ) + mean.unsqueeze(0)
        coeffs[0] = mean  # always include current best
        # Convert to smooth perturbations
        perturbations = coeffs @ basis  # (K, 400)

        # Evaluate
        costs, obs_all, raw_all = rollout_with_perturbations(
            csv_file, ac, perturbations, mdl_path, ort_session, csv_cache, ds=ds
        )

        # Select elites
        sorted_idx = np.argsort(costs)
        elite_idx = sorted_idx[:n_elite]
        elite_coeffs = coeffs[elite_idx]

        # Update CEM distribution
        mean = elite_coeffs.mean(dim=0)
        std = elite_coeffs.std(dim=0).clamp(min=0.001)

        # Track best
        if costs[sorted_idx[0]] < best_cost:
            best_cost = costs[sorted_idx[0]]
            bi = sorted_idx[0]
            best_obs = obs_all[:, bi, :].cpu()  # (400, OBS_DIM)
            best_raw = raw_all[:, bi].cpu()  # (400,)

        if it == 0 or it == N_ITERS - 1:
            baseline = costs[0] if it == 0 else costs[sorted_idx[0]]
            print(
                f"    iter {it}: best={best_cost:.1f}  elite_mean={costs[elite_idx].mean():.1f}"
                f"  std={std.mean().item():.4f}"
            )

    # Also get baseline (zero perturbation) cost
    zero_perturb = torch.zeros(1, N_CTRL, device=DEV)
    baseline_costs, _, _ = rollout_with_perturbations(
        csv_file, ac, zero_perturb, mdl_path, ort_session, csv_cache, ds=ds
    )
    baseline_cost = baseline_costs[0]

    return best_cost, baseline_cost, best_obs, best_raw


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def main():
    print(f"exp080 — MPC with smooth perturbations + real sim")
    print(f"  K={MPC_K}  basis={N_BASIS}  width={BASIS_WIDTH}  σ={PERTURB_STD}")
    print(f"  iters={N_ITERS}  elite={ELITE_FRAC}  routes={N_ROUTES}")

    # Load policy
    ac = PolicyActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_keys = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    ac.load_state_dict(actor_keys, strict=False)
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy (Δs={ds})")

    # Sim
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Basis functions
    basis = make_basis()
    print(f"Basis: {basis.shape}")

    routes = all_csv[:N_ROUTES]
    OUT_DIR.mkdir(exist_ok=True)

    all_best_costs = []
    all_baseline_costs = []
    all_obs = []
    all_raw = []

    for i, csv_file in enumerate(routes):
        t0 = time.time()
        best_cost, baseline_cost, obs, raw = mpc_search_route(
            csv_file, ac, basis, mdl_path, ort_sess, csv_cache, ds=ds
        )
        dt = time.time() - t0
        improve = baseline_cost - best_cost
        print(
            f"  Route {i}: baseline={baseline_cost:.1f}  mpc={best_cost:.1f}"
            f"  Δ={improve:.1f}  ⏱{dt:.0f}s"
        )

        all_best_costs.append(best_cost)
        all_baseline_costs.append(baseline_cost)
        if obs is not None:
            all_obs.append(obs)
            all_raw.append(raw)

        if (i + 1) % 10 == 0 or i == len(routes) - 1:
            mb = np.mean(all_baseline_costs)
            mm = np.mean(all_best_costs)
            print(
                f"\n  [{i + 1} routes] baseline={mb:.1f}  mpc={mm:.1f}  Δ={mb - mm:.1f}\n"
            )

            if all_obs:
                torch.save(
                    {
                        "obs": torch.stack(all_obs),
                        "raw": torch.stack(all_raw),
                        "baseline_costs": np.array(all_baseline_costs),
                        "mpc_costs": np.array(all_best_costs),
                    },
                    OUT_DIR / "mpc_smooth.pt",
                )

    mb = np.mean(all_baseline_costs)
    mm = np.mean(all_best_costs)
    print(
        f"\nDone. {len(routes)} routes. baseline={mb:.1f}  mpc={mm:.1f}  Δ={mb - mm:.1f}"
    )


if __name__ == "__main__":
    main()
