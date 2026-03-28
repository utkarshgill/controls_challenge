# exp098 — Full-sequence gradient optimization through differentiable physics
#
# auriium2's approach: Adam on the full 400-step action sequence per segment,
# backpropping through the differentiable physics model (onnx2torch).
#
# All 5000 segments optimized in parallel (one big batch).
# PPO policy provides warm start.
# Expected-value physics for clean gradients (like auriium2).
# Save optimized actions to npz for lookup controller.

import numpy as np, os, sys, time, torch, torch.nn.functional as F
from pathlib import Path
from hashlib import md5

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

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400
CL = CONTEXT_LENGTH
TEMPERATURE = 0.8

OPTIM_STEPS = int(os.getenv("OPTIM_STEPS", "300"))
OPTIM_LR = float(os.getenv("OPTIM_LR", "0.01"))
N_SEGS = int(os.getenv("N_SEGS", "100"))
BATCH = int(os.getenv("BATCH", "100"))  # segments per batch (memory limit)

EXP_DIR = Path(__file__).parent
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)

BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)


class DiffPhysics(torch.nn.Module):
    """Differentiable physics via onnx2torch. Expected-value output."""

    def __init__(self, onnx_path):
        super().__init__()
        import onnx2torch

        self.model = onnx2torch.convert(str(onnx_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer("bins", BINS)

    def expected_lataccel(self, states, tokens):
        logits = self.model(states, tokens)
        probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)

    def tokenize(self, lataccel):
        return torch.bucketize(
            lataccel.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1]),
            self.bins,
            right=False,
        ).clamp(0, VOCAB_SIZE - 1)


def load_segment_data(csv_paths):
    """Load segment CSV data to GPU tensors."""
    import pandas as pd
    from tinyphysics import ACC_G

    T = COST_END_IDX + 1
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


def diff_rollout(actions, data, diff_phys):
    """Differentiable rollout: actions (B, 400) → cost (B,).

    Uses expected-value physics. Fully differentiable w.r.t. actions.
    """
    B = actions.shape[0]
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
        tok_start = t - CL
        tok_end = t

        act_ctx = all_actions[:, ctx_start:ctx_end]
        roll_ctx = data["roll_lataccel"][:, ctx_start:ctx_end].float()
        v_ctx = data["v_ego"][:, ctx_start:ctx_end].float()
        a_ctx = data["a_ego"][:, ctx_start:ctx_end].float()
        states = torch.stack([act_ctx, roll_ctx, v_ctx, a_ctx], dim=-1)

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
    targets = data["target_lataccel"][:, CONTROL_START_IDX:COST_END_IDX].float()

    lat_cost = ((targets - pred_lataccels) ** 2).mean(dim=1) * 100
    jerk = torch.diff(pred_lataccels, dim=1) / DEL_T
    jerk_cost = (jerk**2).mean(dim=1) * 100
    return lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost


def get_warm_start(csv_paths):
    """Get PPO policy warm-start actions via batched sim."""
    from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
    from experiments.exp055_batch_of_batch.train import (
        ActorCritic,
        _precompute_future_windows,
        fill_obs,
        HIST_LEN,
        OBS_DIM,
        DELTA_SCALE_MAX,
    )

    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    csv_cache = CSVCache([str(f) for f in csv_paths])
    data, rng = csv_cache.slice(csv_paths)
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
        with torch.inference_mode():
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
    actions = sim.action_history[:, CONTROL_START_IDX:COST_END_IDX].float()
    return actions, costs


def optimize_batch(csv_paths, diff_phys):
    """Optimize action sequences for a batch of segments via gradient descent."""
    data = load_segment_data(csv_paths)
    B = len(csv_paths)

    # Warm start from PPO policy
    warm_actions, warm_costs = get_warm_start(csv_paths)
    print(f"    warm start: {np.mean(warm_costs):.1f}")

    # Optimize
    actions = warm_actions.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([actions], lr=OPTIM_LR)

    best_cost = torch.full((B,), float("inf"), device=DEV)
    best_actions = actions.data.clone()

    for step in range(OPTIM_STEPS):
        optimizer.zero_grad()
        cost = diff_rollout(actions, data, diff_phys)
        loss = cost.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            actions.data.clamp_(STEER_RANGE[0], STEER_RANGE[1])
            improved = cost < best_cost
            best_cost[improved] = cost[improved]
            best_actions[improved] = actions.data[improved]

        if step % 50 == 0 or step == OPTIM_STEPS - 1:
            print(
                f"    step {step:3d}  mean={cost.mean().item():.2f}  best={best_cost.mean().item():.2f}"
            )

    return best_actions.detach(), best_cost.detach().cpu().numpy()


def main():
    print(f"exp098 — Full-sequence gradient optimization")
    print(f"  steps={OPTIM_STEPS}  lr={OPTIM_LR}  segs={N_SEGS}  batch={BATCH}")

    diff_phys = DiffPhysics(ROOT / "models" / "tinyphysics.onnx").to(DEV)
    print(f"Loaded differentiable physics (onnx2torch)")

    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_SEGS]
    EXP_DIR.mkdir(exist_ok=True)

    all_actions = []
    all_costs = []
    all_paths = []

    for i in range(0, len(all_csv), BATCH):
        batch = all_csv[i : i + BATCH]
        print(f"\nBatch {i // BATCH + 1}: segments {i}..{i + len(batch) - 1}")
        t0 = time.time()
        best_act, best_cost = optimize_batch(batch, diff_phys)
        dt = time.time() - t0
        all_actions.append(best_act.cpu().numpy())
        all_costs.extend(best_cost.tolist())
        all_paths.extend([str(p) for p in batch])
        print(f"    ⏱{dt:.0f}s  mean_cost={np.mean(best_cost):.1f}")

    mc = np.mean(all_costs)
    print(f"\nDone. {len(all_csv)} segments. Mean cost: {mc:.1f}")

    np.savez(
        EXP_DIR / "optimized_actions.npz",
        actions=np.concatenate(all_actions),
        costs=np.array(all_costs),
        paths=all_paths,
    )
    print(f"Saved to {EXP_DIR / 'optimized_actions.npz'}")


if __name__ == "__main__":
    main()
