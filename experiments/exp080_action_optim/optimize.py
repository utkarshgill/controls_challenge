# exp080 — Per-route action sequence optimization via differentiable physics
#
# Approach proven by auriium2 (#1 at 13.6):
#   - Differentiable surrogate of physics model (onnx2torch)
#   - Adam optimize full 400-step action sequence per route
#   - Expected-value physics for clean gradients
#   - Save optimized (obs, action) pairs for BC distillation into policy
#
# This script optimizes action sequences. A separate train.py will
# BC-distill into the policy and PPO-finetune.

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
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 0.8
N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400

# ── Config ────────────────────────────────────────────────────
OPTIM_STEPS = int(os.getenv("OPTIM_STEPS", "300"))
OPTIM_LR = float(os.getenv("OPTIM_LR", "0.05"))
N_ROUTES = int(os.getenv("N_ROUTES", "5000"))
BATCH_SIZE = int(os.getenv("BATCH", "10"))  # routes to optimize in parallel
LOG_EVERY = int(os.getenv("LOG_EVERY", "50"))

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "optimized_actions"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)

# ── Policy architecture (must match exp055) ───────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
DELTA_SCALE = 0.25
S_LAT, S_STEER, S_VEGO, S_AEGO, S_ROLL, S_CURV = 5.0, 2.0, 40.0, 4.0, 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256


# ══════════════════════════════════════════════════════════════
#  Differentiable physics model
# ══════════════════════════════════════════════════════════════


class DiffPhysics(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        import onnx2torch

        self.model = onnx2torch.convert(str(onnx_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(
            np.float32
        )
        self.register_buffer("bins", torch.from_numpy(bins))

    def expected_lataccel(self, states, tokens):
        logits = self.model(states, tokens)
        probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)

    def tokenize(self, lataccel):
        clamped = lataccel.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(clamped, self.bins, right=False).clamp(0, VOCAB_SIZE - 1)


# ══════════════════════════════════════════════════════════════
#  Policy for warm-starting (from exp055)
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


def get_policy_actions_batch(
    csv_files, ac, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """Roll out policy on routes via batched sim, return (N, 400) action array and costs."""
    from tinyphysics_batched import BatchedSimulator

    data, rng = csv_cache.slice(csv_files)
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
    hist_head = HIST_LEN - 1
    actions_out = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")

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
            raw = ac.get_delta(obs_buf)
        delta = raw * ds
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        ci = step_idx - CONTROL_START_IDX
        if 0 <= ci < N_CTRL:
            actions_out[:, ci] = action.float()
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return actions_out, costs


# ══════════════════════════════════════════════════════════════
#  Load route data
# ══════════════════════════════════════════════════════════════


def load_route_data(csv_paths):
    import pandas as pd
    from tinyphysics import ACC_G

    batch = []
    for p in csv_paths:
        df = pd.read_csv(str(p))
        batch.append(
            {
                "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
                "v_ego": df["vEgo"].values,
                "a_ego": df["aEgo"].values,
                "target_lataccel": df["targetLateralAcceleration"].values,
                "steer_command": -df["steerCommand"].values,
            }
        )

    T = COST_END_IDX + 1
    data = {}
    for key in batch[0]:
        arrs = []
        for b in batch:
            arr = b[key][:T]
            if len(arr) < T:
                arr = np.pad(arr, (0, T - len(arr)), mode="edge")
            arrs.append(arr)
        data[key] = torch.tensor(np.stack(arrs), dtype=torch.float32, device=DEV)
    return data


# ══════════════════════════════════════════════════════════════
#  Differentiable rollout: actions → cost (backpropable)
# ══════════════════════════════════════════════════════════════


def diff_rollout(actions, data, diff_phys):
    """Forward pass: action sequence → total cost.

    actions: (B, 400) float, differentiable
    Returns: total_cost (B,), pred_lataccels (B, 400)
    """
    B = actions.shape[0]
    CL = CONTEXT_LENGTH

    init_actions = data["steer_command"][:, :CONTROL_START_IDX].float()
    init_preds = data["target_lataccel"][:, :CONTROL_START_IDX].float()

    all_actions = torch.cat(
        [init_actions, actions.clamp(STEER_RANGE[0], STEER_RANGE[1])], dim=1
    )

    current_la = init_preds[:, -1]
    pred_list = []

    for t in range(CONTROL_START_IDX, COST_END_IDX):
        ctx_start = t - CL + 1
        ctx_end = t + 1

        act_ctx = all_actions[:, ctx_start:ctx_end]
        roll_ctx = data["roll_lataccel"][:, ctx_start:ctx_end].float()
        v_ctx = data["v_ego"][:, ctx_start:ctx_end].float()
        a_ctx = data["a_ego"][:, ctx_start:ctx_end].float()
        states = torch.stack([act_ctx, roll_ctx, v_ctx, a_ctx], dim=-1)

        tok_start = t - CL
        tok_end = t
        pred_so_far = (
            torch.cat([init_preds] + pred_list, dim=1) if pred_list else init_preds
        )
        pred_ctx = pred_so_far[:, tok_start:tok_end]
        tokens = diff_phys.tokenize(pred_ctx)

        pred_la = diff_phys.expected_lataccel(states, tokens)
        pred_la = torch.clamp(
            pred_la, current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA
        )

        pred_list.append(pred_la.unsqueeze(1))
        current_la = pred_la

    pred_lataccels = torch.cat(pred_list, dim=1)
    target_lataccels = data["target_lataccel"][
        :, CONTROL_START_IDX:COST_END_IDX
    ].float()

    lat_cost = ((target_lataccels - pred_lataccels) ** 2).mean(dim=1) * 100
    jerk = torch.diff(pred_lataccels, dim=1) / DEL_T
    jerk_cost = (jerk**2).mean(dim=1) * 100
    total_cost = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

    return total_cost, pred_lataccels


# ══════════════════════════════════════════════════════════════
#  Optimize a batch of routes
# ══════════════════════════════════════════════════════════════


def optimize_batch(csv_paths, diff_phys, policy_init=None):
    """policy_init: (B, 400) tensor of policy actions to warm-start from."""
    data = load_route_data(csv_paths)
    B = len(csv_paths)

    if policy_init is not None:
        actions = policy_init.float().to(DEV).clone().detach().requires_grad_(True)
    else:
        last_steer = data["steer_command"][:, CONTROL_START_IDX - 1 : CONTROL_START_IDX]
        actions = (
            last_steer.expand(-1, N_CTRL).clone().float().detach().requires_grad_(True)
        )

    optimizer = torch.optim.Adam([actions], lr=OPTIM_LR)

    # Track best per route
    best_cost = torch.full((B,), float("inf"), device=DEV)
    best_actions = actions.data.clone()

    for step in range(OPTIM_STEPS):
        optimizer.zero_grad()
        cost, preds = diff_rollout(actions, data, diff_phys)
        loss = cost.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            actions.data.clamp_(STEER_RANGE[0], STEER_RANGE[1])
            improved = cost < best_cost
            best_cost[improved] = cost[improved]
            best_actions[improved] = actions.data[improved]

        if step % LOG_EVERY == 0 or step == OPTIM_STEPS - 1:
            print(
                f"    step {step:3d}  mean={cost.mean().item():.2f}  "
                f"best={best_cost.mean().item():.2f}  "
                f"min={best_cost.min().item():.2f}  max={best_cost.max().item():.2f}"
            )

    return best_actions.detach(), best_cost.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def main():
    print(f"exp080 — Gradient-based action optimization (PPO warm-start)")
    print(
        f"  routes={N_ROUTES}  batch={BATCH_SIZE}  steps={OPTIM_STEPS}  lr={OPTIM_LR}"
    )
    print(f"  device={DEV}")

    # Load diff physics
    diff_phys = DiffPhysics(ROOT / "models" / "tinyphysics.onnx").to(DEV)
    print(f"Loaded differentiable physics model on {DEV}")

    # Load policy for warm-starting
    ac = PolicyActor().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    actor_keys = {k: v for k, v in ckpt["ac"].items() if k.startswith("actor.")}
    ac.load_state_dict(actor_keys, strict=False)
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))
    print(f"Loaded policy from {BASE_PT} (Δs={ds})")

    # ONNX session for policy rollout
    from tinyphysics_batched import make_ort_session, CSVCache

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    csv_cache = CSVCache([str(f) for f in all_csv])
    routes = all_csv[:N_ROUTES]

    OUT_DIR.mkdir(exist_ok=True)

    all_actions = []
    all_costs = []
    all_policy_costs = []
    all_paths = []

    for i in range(0, len(routes), BATCH_SIZE):
        batch_paths = routes[i : i + BATCH_SIZE]
        n_batch = len(batch_paths)
        print(
            f"\nBatch {i // BATCH_SIZE + 1}/{(len(routes) + BATCH_SIZE - 1) // BATCH_SIZE}"
            f"  routes {i}..{i + n_batch - 1}"
        )

        # Step 1: Get policy actions via fast batched sim (warm start)
        t0 = time.time()
        policy_actions, policy_costs = get_policy_actions_batch(
            batch_paths, ac, mdl_path, ort_sess, csv_cache, ds=ds
        )
        t_policy = time.time() - t0
        print(
            f"  Policy rollout: {t_policy:.1f}s  mean_cost={np.mean(policy_costs):.1f}"
        )

        # Step 2: Gradient-optimize starting from policy actions
        t1 = time.time()
        best_act, best_cost = optimize_batch(
            batch_paths, diff_phys, policy_init=policy_actions
        )
        t_optim = time.time() - t1
        improve = np.mean(policy_costs) - np.mean(best_cost)
        print(
            f"  Optimized: {t_optim:.0f}s  mean_cost={np.mean(best_cost):.1f}"
            f"  Δ={improve:.1f}"
        )

        all_actions.append(best_act.cpu())
        all_costs.extend(best_cost.tolist())
        all_policy_costs.extend(
            policy_costs.tolist()
            if hasattr(policy_costs, "tolist")
            else list(policy_costs)
        )
        all_paths.extend([str(p) for p in batch_paths])

        # Periodic save
        if (i + BATCH_SIZE) % 50 == 0 or i + BATCH_SIZE >= len(routes):
            torch.save(
                {
                    "actions": torch.cat(all_actions, dim=0),
                    "costs": np.array(all_costs),
                    "policy_costs": np.array(all_policy_costs),
                    "paths": all_paths,
                },
                OUT_DIR / "optimized.pt",
            )
            mc = np.mean(all_costs)
            mp = np.mean(all_policy_costs)
            print(
                f"  [{len(all_costs)} routes] policy={mp:.1f} → optimized={mc:.1f}"
                f"  Δ={mp - mc:.1f}"
            )

    mp = np.mean(all_policy_costs)
    mc = np.mean(all_costs)
    print(f"\nDone. {len(routes)} routes.")
    print(f"  Policy:    {mp:.1f}")
    print(f"  Optimized: {mc:.1f}")
    print(f"  Δ:         {mp - mc:.1f}")
    print(f"Saved to {OUT_DIR / 'optimized.pt'}")


if __name__ == "__main__":
    main()
