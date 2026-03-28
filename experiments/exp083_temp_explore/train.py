# exp083 — PGTO with massively parallel smooth MPC + BC distillation
#
# Each epoch:
#   1. Tile R routes × K candidates = R*K rollouts
#   2. Each candidate gets a smooth perturbation (basis functions) to the policy
#   3. All R*K rollouts run in one batched sim call (TRT, 3 seconds)
#   4. Per route, pick the lowest-cost candidate
#   5. BC the policy toward winning actions (with anchor to baseline)
#   6. Repeat
#
# This is MPC using the real sim, massively parallel, with the policy as prior.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

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
    LATACCEL_RANGE,
    VOCAB_SIZE,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture (match exp055) ───────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = 0.25
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256
N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

# ── MPC config ────────────────────────────────────────────────
MPC_K = int(os.getenv("MPC_K", "10"))  # candidates per route
N_BASIS = int(os.getenv("N_BASIS", "40"))  # smooth basis functions
BASIS_WIDTH = int(os.getenv("BASIS_WIDTH", "20"))  # width of each bump
PERTURB_STD = float(os.getenv("PERTURB_STD", "0.02"))
CEM_ITERS = int(
    os.getenv("CEM_ITERS", "1")
)  # CEM iterations per epoch (1 = no refinement)
ELITE_FRAC = float(os.getenv("ELITE_FRAC", "0.125"))

# ── distillation ──────────────────────────────────────────────
LR = float(os.getenv("LR", "3e-5"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "0.5"))
BC_ANCHOR = float(os.getenv("BC_ANCHOR", "0.8"))  # high anchor to prevent drift
MINI_BS = int(os.getenv("MINI_BS", "25000"))

# ── runtime ───────────────────────────────────────────────────
N_ROUTES = int(os.getenv("N_ROUTES", "500"))  # routes per epoch
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = 100

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)
RESUME = int(os.getenv("RESUME", "0"))


# ══════════════════════════════════════════════════════════════
#  Policy
# ══════════════════════════════════════════════════════════════


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.net = nn.Sequential(*a)

    def forward(self, obs):
        logits = self.net(obs)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        return a_p, b_p

    def get_delta(self, obs):
        a_p, b_p = self.forward(obs)
        return 2.0 * a_p / (a_p + b_p) - 1.0


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
    return basis  # (N_BASIS, N_CTRL)


# ══════════════════════════════════════════════════════════════
#  Obs builder (match exp055)
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
#  MPC rollout: R routes × K candidates, all parallel
# ══════════════════════════════════════════════════════════════


def mpc_rollout(
    route_files,
    actor,
    basis,
    perturbation_coeffs,
    mdl_path,
    ort_session,
    csv_cache,
    ds=DELTA_SCALE,
):
    """Run R routes × K candidates in one batched sim call.

    perturbation_coeffs: (R, K, N_BASIS) — per-route per-candidate coefficients.
                         Candidate 0 per route should have all-zero coeffs (baseline).

    Returns: costs (R, K), winner_obs (R, S, OBS_DIM), winner_raw (R, S)
    """
    R = len(route_files)
    Ktot = R * MPC_K

    # Tile: [r0c0, r0c1, ..., r0cK, r1c0, r1c1, ..., r1cK, ...]
    csv_tiled = [f for f in route_files for _ in range(MPC_K)]
    data, rng = csv_cache.slice(csv_tiled)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    sim.use_expected = True  # deterministic
    N = sim.N  # = R * K
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    # Precompute smooth perturbations: (R, K, N_CTRL) -> (N, N_CTRL)
    perturbations = torch.bmm(
        perturbation_coeffs.view(R * MPC_K, N_BASIS).unsqueeze(1),
        basis.unsqueeze(0).expand(R * MPC_K, -1, -1),
    ).squeeze(1)  # (N, N_CTRL)

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
            policy_raw = actor.get_delta(obs_buf)

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

    costs = sim.rollout(ctrl)["total_cost"]  # (N,)

    # Reshape to (R, K), find winners
    costs_2d = torch.from_numpy(costs).to("cuda").view(R, MPC_K)
    winners = costs_2d.argmin(dim=1)  # (R,)
    winner_flat = torch.arange(R, device="cuda") * MPC_K + winners

    # Extract winner obs and raw
    S_actual = si
    win_obs = all_obs[:S_actual, winner_flat, :]  # (S, R, OBS_DIM)
    win_raw = all_raw[:S_actual, winner_flat]  # (S, R)

    # Also extract baseline (candidate 0) obs and raw
    base_flat = torch.arange(R, device="cuda") * MPC_K
    base_obs = all_obs[:S_actual, base_flat, :]
    base_raw = all_raw[:S_actual, base_flat]

    baseline_costs = costs_2d[:, 0]
    winner_costs = costs_2d.gather(1, winners.unsqueeze(1)).squeeze(1)
    improved = winner_costs < baseline_costs

    return dict(
        costs_2d=costs_2d,
        baseline_mean=baseline_costs.mean().item(),
        winner_mean=winner_costs.mean().item(),
        improve=(baseline_costs - winner_costs).mean().item(),
        frac_improved=(winners != 0).float().mean().item(),
        n_improved=improved.sum().item(),
        # Only return obs/raw for routes where winner beat baseline
        win_obs=win_obs[:, improved].permute(1, 0, 2).reshape(-1, OBS_DIM)
        if improved.any()
        else None,
        win_raw=win_raw[:, improved].T.reshape(-1) if improved.any() else None,
        base_obs=base_obs.permute(1, 0, 2).reshape(-1, OBS_DIM),
        base_raw=base_raw.T.reshape(-1),
    )


# ══════════════════════════════════════════════════════════════
#  BC distillation with anchor
# ══════════════════════════════════════════════════════════════


def distill_step(actor, opt, res):
    obs_parts, raw_parts, w_parts = [], [], []
    if res["win_obs"] is not None:
        obs_parts.append(res["win_obs"])
        raw_parts.append(res["win_raw"])
        w_parts.append(
            torch.full((res["win_obs"].shape[0],), 1.0 - BC_ANCHOR, device="cuda")
        )
    obs_parts.append(res["base_obs"])
    raw_parts.append(res["base_raw"])
    w_parts.append(torch.full((res["base_obs"].shape[0],), BC_ANCHOR, device="cuda"))

    obs = torch.cat(obs_parts)
    raw = torch.cat(raw_parts)
    weights = torch.cat(w_parts)
    x_t = ((raw.unsqueeze(-1) + 1) / 2).clamp(1e-6, 1 - 1e-6)

    nll_sum, n = 0.0, 0
    for idx in torch.randperm(len(obs), device="cuda").split(MINI_BS):
        a_p, b_p = actor(obs[idx])
        dist = torch.distributions.Beta(a_p, b_p)
        per_nll = -dist.log_prob(x_t[idx].squeeze(-1))
        loss = (per_nll * weights[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        opt.step()
        bs = idx.numel()
        nll_sum += per_nll.mean().item() * bs
        n += bs
    return nll_sum / max(1, n)


# ══════════════════════════════════════════════════════════════
#  Eval
# ══════════════════════════════════════════════════════════════


def evaluate(actor, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE):
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
            raw = actor.get_delta(obs_buf)
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
#  Train
# ══════════════════════════════════════════════════════════════


def train():
    actor = Actor().to(DEV)

    # Load weights
    ckpt_path = BEST_PT if (RESUME and BEST_PT.exists()) else BASE_PT
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEV)
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        mapped = {
            f"net.{k[len('actor.') :]}": v
            for k, v in ckpt["ac"].items()
            if k.startswith("actor.")
        }
        actor.load_state_dict(mapped, strict=False)
    ds_ckpt = ckpt.get("delta_scale", None)
    if ds_ckpt is not None:
        global DELTA_SCALE
        DELTA_SCALE = float(ds_ckpt)
    ds = DELTA_SCALE
    print(f"Loaded from {ckpt_path} (Δs={ds})")
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")

    opt = optim.Adam(actor.parameters(), lr=LR)
    basis = make_basis()  # (N_BASIS, N_CTRL)

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    vm, vs = evaluate(actor, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
    best, best_ep = vm, "init"
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")
    print(
        f"\nPGTO-MPC: K={MPC_K}  basis={N_BASIS}  width={BASIS_WIDTH}  σ={PERTURB_STD}"
    )
    print(
        f"  lr={LR}  anchor={BC_ANCHOR}  routes/ep={N_ROUTES}  total_rollouts/ep={N_ROUTES * MPC_K}"
    )
    print()

    def save_best():
        torch.save({"actor": actor.state_dict(), "delta_scale": ds}, BEST_PT)

    for epoch in range(MAX_EP):
        actor.train()
        t0 = time.time()

        batch = random.sample(all_csv, min(N_ROUTES, len(all_csv)))

        # Generate perturbation coefficients: (R, K, N_BASIS)
        R = len(batch)
        coeffs = torch.randn(R, MPC_K, N_BASIS, device=DEV) * PERTURB_STD
        coeffs[:, 0, :] = 0.0  # candidate 0 = unperturbed baseline

        res = mpc_rollout(
            batch, actor, basis, coeffs, mdl_path, ort_sess, csv_cache, ds=ds
        )
        t1 = time.time()

        # Distill
        nll = distill_step(actor, opt, res)
        tu = time.time() - t1

        line = (
            f"E{epoch:3d}  base={res['baseline_mean']:5.1f}"
            f"  win={res['winner_mean']:5.1f}"
            f"  Δ={res['improve']:.1f}  f={res['frac_improved']:.2f}"
            f"  n={res['n_improved']}"
            f"  nll={nll:.3f}  ⏱{t1 - t0:.0f}+{tu:.0f}s"
        )

        if epoch % EVAL_EVERY == 0:
            actor.eval()
            vm, vs = evaluate(actor, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:5.1f}±{vs:4.1f}{mk}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")


if __name__ == "__main__":
    train()
