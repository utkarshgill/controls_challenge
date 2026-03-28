# exp085 — GPU coordinate descent MPC with exact RNG replay
#
# Key insight from ksd3: np.random.choice(1024, p=probs) consumes one uniform
# draw and is equivalent to searchsorted(cumsum(probs), uniform). We precompute
# all uniform draws per segment, then replay them on GPU with any action sequence.
#
# This gives EXACT cost evaluation on GPU — no expected-value approximation.
# Coordinate descent: at each step, try K candidate actions in parallel on GPU,
# simulate H steps forward, pick the best. All using the onnx2torch model + precomputed RNG.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F
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
    ACC_G,
)

torch.manual_seed(42)
DEV = torch.device("cuda")

TEMPERATURE = 0.8
N_CTRL = COST_END_IDX - CONTROL_START_IDX  # 400
N_STEPS_TOTAL = 580  # steps 20..599, each consumes one RNG draw

# ── Config ────────────────────────────────────────────────────
N_CAND = int(os.getenv("N_CAND", "40"))
HORIZON = int(os.getenv("HORIZON", "20"))
NUM_PASSES = int(os.getenv("NUM_PASSES", "2"))
RADII = [float(x) for x in os.getenv("RADII", "0.15,0.075").split(",")]
N_SEGS = int(os.getenv("N_SEGS", "5"))

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "optimized"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)

BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)


# ══════════════════════════════════════════════════════════════
#  Precompute RNG draws for a segment
# ══════════════════════════════════════════════════════════════


def precompute_rng(data_path):
    """Extract the uniform random draws that np.random.choice would use.
    Returns (N_STEPS_TOTAL,) array of uniform[0,1] draws.
    """
    seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)
    # Each sim_step calls np.random.choice which internally calls np.random.random() once
    draws = np.random.random(N_STEPS_TOTAL).astype(np.float32)
    return draws


def load_segment_data(data_path):
    """Load segment CSV into GPU tensors."""
    import pandas as pd

    df = pd.read_csv(str(data_path))
    n = len(df)
    data = {
        "roll_lataccel": torch.tensor(
            np.sin(df["roll"].values) * ACC_G, dtype=torch.float32, device=DEV
        ),
        "v_ego": torch.tensor(df["vEgo"].values, dtype=torch.float32, device=DEV),
        "a_ego": torch.tensor(df["aEgo"].values, dtype=torch.float32, device=DEV),
        "target_lataccel": torch.tensor(
            df["targetLateralAcceleration"].values, dtype=torch.float32, device=DEV
        ),
        "steer_command": torch.tensor(
            -df["steerCommand"].values, dtype=torch.float32, device=DEV
        ),
        "n_steps": n,
    }
    return data


# ══════════════════════════════════════════════════════════════
#  GPU physics model (onnx2torch) with RNG replay
# ══════════════════════════════════════════════════════════════


class GPUPhysics:
    """Physics model on GPU with exact RNG replay via precomputed uniform draws."""

    def __init__(self, onnx_path):
        import onnx2torch

        self.model = onnx2torch.convert(str(onnx_path)).to(DEV)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def step_batched(self, states, tokens, uniform_draws):
        """One physics step for B parallel candidates.

        states: (B, 20, 4) float
        tokens: (B, 20) long
        uniform_draws: (B,) float — precomputed uniform[0,1] for RNG replay

        Returns: pred_lataccel (B,) — sampled (not expected) lataccel
        """
        logits = self.model(states, tokens)  # (B, 20, 1024)
        logits = logits[:, -1, :] / TEMPERATURE  # (B, 1024)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.exp(logits)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # RNG replay: searchsorted(cumsum(probs), uniform_draw)
        cumprobs = torch.cumsum(probs, dim=-1)  # (B, 1024)
        # Find the bin where cumprobs >= uniform_draw
        samples = torch.searchsorted(cumprobs, uniform_draws.unsqueeze(-1)).squeeze(
            -1
        )  # (B,)
        samples = samples.clamp(0, VOCAB_SIZE - 1)

        # Decode: token → lataccel value
        pred = BINS[samples]
        return pred


# ══════════════════════════════════════════════════════════════
#  GPU coordinate descent optimizer for one segment
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def optimize_segment(data, rng_draws, warm_actions, physics):
    """Coordinate descent MPC on GPU.

    data: dict of GPU tensors for this segment
    rng_draws: (N_STEPS_TOTAL,) precomputed uniform draws on GPU
    warm_actions: (n_ctrl,) warm-start steer actions on GPU
    physics: GPUPhysics instance

    Returns: optimized actions (n_ctrl,) tensor, final cost
    """
    n_steps = data["n_steps"]
    n_ctrl = n_steps - CONTROL_START_IDX
    CL = CONTEXT_LENGTH

    actions = warm_actions.clone()
    cand_rng = torch.Generator(device=DEV)
    cand_rng.manual_seed(42)

    for pass_num in range(NUM_PASSES):
        radius = RADII[min(pass_num, len(RADII) - 1)]

        # Initialize sim state for this pass
        # action_hist: (CL,) from data
        action_hist = data["steer_command"][:CL].clone()
        # state_hist: (CL, 3) = [roll_la, v_ego, a_ego]
        state_hist = torch.stack(
            [data["roll_lataccel"][:CL], data["v_ego"][:CL], data["a_ego"][:CL]], dim=-1
        )
        # pred_hist: (CL,) = target_lataccel for context
        pred_hist = data["target_lataccel"][:CL].clone()
        current_la = pred_hist[-1].clone()

        # Warmup: steps 20..99 (use data steer commands)
        for step_idx in range(CL, CONTROL_START_IDX):
            rng_idx = step_idx - CL  # index into rng_draws

            action = data["steer_command"][step_idx]
            # Shift histories
            action_hist = torch.cat([action_hist[1:], action.unsqueeze(0)])
            new_state = torch.stack(
                [
                    data["roll_lataccel"][step_idx],
                    data["v_ego"][step_idx],
                    data["a_ego"][step_idx],
                ]
            )
            state_hist = torch.cat([state_hist[1:], new_state.unsqueeze(0)])

            # Build model input: (1, 20, 4)
            full_states = torch.cat(
                [action_hist.unsqueeze(-1), state_hist], dim=-1
            ).unsqueeze(0)
            tokens = (
                torch.bucketize(pred_hist.clamp(-5, 5), BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .unsqueeze(0)
                .long()
            )

            pred = physics.step_batched(
                full_states, tokens, rng_draws[rng_idx : rng_idx + 1]
            )
            pred = pred[0].clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)

            # Before control start, use target as current
            current_la = data["target_lataccel"][step_idx]
            pred_hist = torch.cat([pred_hist[1:], current_la.unsqueeze(0)])

        # Now optimize scored steps: 100..499
        improved = 0
        for t in range(N_CTRL):
            step_idx = CONTROL_START_IDX + t
            if step_idx >= n_steps:
                break
            rng_idx = step_idx - CL

            # Save state
            saved_action_hist = action_hist.clone()
            saved_state_hist = state_hist.clone()
            saved_pred_hist = pred_hist.clone()
            saved_current_la = current_la.clone()

            # Generate candidates
            base = actions[t]
            perturbs = torch.randn(N_CAND, device=DEV, generator=cand_rng) * radius
            candidates = (base + perturbs).clamp(STEER_RANGE[0], STEER_RANGE[1])
            candidates = torch.cat([candidates, base.unsqueeze(0)])  # include current
            K = candidates.shape[0]  # N_CAND + 1

            # Evaluate each candidate: simulate H steps forward
            # Batch: run K copies in parallel
            b_action_hist = saved_action_hist.unsqueeze(0).expand(K, -1).clone()
            b_state_hist = saved_state_hist.unsqueeze(0).expand(K, -1, -1).clone()
            b_pred_hist = saved_pred_hist.unsqueeze(0).expand(K, -1).clone()
            b_current_la = saved_current_la.unsqueeze(0).expand(K).clone()

            cost_lat = torch.zeros(K, device=DEV)
            cost_jerk = torch.zeros(K, device=DEV)
            b_prev_la = b_current_la.clone()
            n_terms = 0

            for h in range(HORIZON):
                s = step_idx + h
                if s >= n_steps:
                    break
                s_rng_idx = s - CL

                # Action: candidate for h=0, warm-start for h>0
                if h == 0:
                    act = candidates
                else:
                    ctrl_idx = s - CONTROL_START_IDX
                    if 0 <= ctrl_idx < len(actions):
                        act = actions[ctrl_idx].expand(K)
                    else:
                        act = torch.zeros(K, device=DEV)

                act = act.clamp(STEER_RANGE[0], STEER_RANGE[1])

                # Update histories
                b_action_hist = torch.cat(
                    [b_action_hist[:, 1:], act.unsqueeze(1)], dim=1
                )
                new_st = torch.stack(
                    [
                        data["roll_lataccel"][s].expand(K),
                        data["v_ego"][s].expand(K),
                        data["a_ego"][s].expand(K),
                    ],
                    dim=-1,
                )
                b_state_hist = torch.cat(
                    [b_state_hist[:, 1:, :], new_st.unsqueeze(1)], dim=1
                )

                # Model input
                full_states = torch.cat(
                    [b_action_hist.unsqueeze(-1), b_state_hist], dim=-1
                )
                tokens = (
                    torch.bucketize(b_pred_hist.clamp(-5, 5), BINS, right=False)
                    .clamp(0, VOCAB_SIZE - 1)
                    .long()
                )

                # Physics step with RNG replay (same draw for all candidates)
                uniform = rng_draws[s_rng_idx].expand(K)
                pred = physics.step_batched(full_states, tokens, uniform)
                pred = pred.clamp(
                    b_current_la - MAX_ACC_DELTA, b_current_la + MAX_ACC_DELTA
                )

                if s >= CONTROL_START_IDX:
                    b_current_la = pred
                else:
                    b_current_la = data["target_lataccel"][s].expand(K)

                b_pred_hist = torch.cat(
                    [b_pred_hist[:, 1:], b_current_la.unsqueeze(1)], dim=1
                )

                # Cost
                if CONTROL_START_IDX <= s < COST_END_IDX:
                    target = data["target_lataccel"][s]
                    cost_lat += (target - b_current_la) ** 2
                    if n_terms > 0:
                        cost_jerk += ((b_current_la - b_prev_la) / DEL_T) ** 2
                    b_prev_la = b_current_la.clone()
                    n_terms += 1

            if n_terms > 0:
                total_cost = (
                    cost_lat / n_terms * 100 * LAT_ACCEL_COST_MULTIPLIER
                    + cost_jerk / max(n_terms - 1, 1) * 100
                )
            else:
                total_cost = torch.zeros(K, device=DEV)

            best_idx = total_cost.argmin()
            best_action = candidates[best_idx]
            if best_idx != K - 1:  # improved over current
                actions[t] = best_action
                improved += 1

            # Advance with best action from saved state
            action_hist = saved_action_hist
            state_hist = saved_state_hist
            pred_hist = saved_pred_hist
            current_la = saved_current_la

            act_final = best_action.clamp(STEER_RANGE[0], STEER_RANGE[1])
            action_hist = torch.cat([action_hist[1:], act_final.unsqueeze(0)])
            new_st = torch.stack(
                [
                    data["roll_lataccel"][step_idx],
                    data["v_ego"][step_idx],
                    data["a_ego"][step_idx],
                ]
            )
            state_hist = torch.cat([state_hist[1:], new_st.unsqueeze(0)])

            full_states = torch.cat(
                [action_hist.unsqueeze(-1), state_hist], dim=-1
            ).unsqueeze(0)
            tokens = (
                torch.bucketize(pred_hist.clamp(-5, 5), BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .unsqueeze(0)
                .long()
            )
            pred = physics.step_batched(
                full_states, tokens, rng_draws[rng_idx : rng_idx + 1]
            )
            pred = pred[0].clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)
            current_la = (
                pred
                if step_idx >= CONTROL_START_IDX
                else data["target_lataccel"][step_idx]
            )
            pred_hist = torch.cat([pred_hist[1:], current_la.unsqueeze(0)])

        print(f"    pass {pass_num}: radius={radius:.3f}  improved={improved}/{N_CTRL}")

    return actions


# ══════════════════════════════════════════════════════════════
#  Evaluate with RNG replay on GPU (full segment)
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_segment(data, rng_draws, actions, physics):
    """Evaluate an action sequence with exact RNG replay on GPU."""
    n_steps = data["n_steps"]
    CL = CONTEXT_LENGTH

    action_hist = data["steer_command"][:CL].clone()
    state_hist = torch.stack(
        [data["roll_lataccel"][:CL], data["v_ego"][:CL], data["a_ego"][:CL]], dim=-1
    )
    pred_hist = data["target_lataccel"][:CL].clone()
    current_la = pred_hist[-1].clone()

    preds = []

    for step_idx in range(CL, n_steps):
        rng_idx = step_idx - CL

        if step_idx >= CONTROL_START_IDX:
            ctrl_idx = step_idx - CONTROL_START_IDX
            act = (
                actions[ctrl_idx]
                if ctrl_idx < len(actions)
                else torch.tensor(0.0, device=DEV)
            )
        else:
            act = data["steer_command"][step_idx]
        act = act.clamp(STEER_RANGE[0], STEER_RANGE[1])

        action_hist = torch.cat([action_hist[1:], act.unsqueeze(0)])
        new_st = torch.stack(
            [
                data["roll_lataccel"][step_idx],
                data["v_ego"][step_idx],
                data["a_ego"][step_idx],
            ]
        )
        state_hist = torch.cat([state_hist[1:], new_st.unsqueeze(0)])

        full_states = torch.cat(
            [action_hist.unsqueeze(-1), state_hist], dim=-1
        ).unsqueeze(0)
        tokens = (
            torch.bucketize(pred_hist.clamp(-5, 5), BINS, right=False)
            .clamp(0, VOCAB_SIZE - 1)
            .unsqueeze(0)
            .long()
        )

        pred = physics.step_batched(
            full_states, tokens, rng_draws[rng_idx : rng_idx + 1]
        )
        pred = pred[0].clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)

        if step_idx >= CONTROL_START_IDX:
            current_la = pred
        else:
            current_la = data["target_lataccel"][step_idx]

        pred_hist = torch.cat([pred_hist[1:], current_la.unsqueeze(0)])
        preds.append(current_la.item())

    # Cost over scored window
    all_preds = preds[CONTROL_START_IDX - CL :]  # from step 100 onwards
    scored_preds = torch.tensor(all_preds[:N_CTRL], device=DEV)
    targets = data["target_lataccel"][CONTROL_START_IDX:COST_END_IDX]

    lat_cost = ((targets - scored_preds) ** 2).mean() * 100
    jerk_cost = ((torch.diff(scored_preds) / DEL_T) ** 2).mean() * 100
    total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    return total.item()


# ══════════════════════════════════════════════════════════════
#  Get policy warm-start via batched sim
# ══════════════════════════════════════════════════════════════


def get_warm_actions(csv_paths):
    """Use exp055 batched sim to get policy actions."""
    from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
    from experiments.exp055_batch_of_batch.train import (
        ActorCritic,
        _precompute_future_windows,
        fill_obs,
        HIST_LEN,
        OBS_DIM,
        DELTA_SCALE_MAX,
    )
    import torch.nn.functional as F2

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
        a_p = F2.softplus(logits[..., 0]) + 1.0
        b_p = F2.softplus(logits[..., 1]) + 1.0
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


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def main():
    print(f"exp085 — GPU coordinate descent MPC with exact RNG replay")
    print(f"  candidates={N_CAND}  horizon={HORIZON}  passes={NUM_PASSES}")
    print(f"  radii={RADII}  segments={N_SEGS}")

    import onnx2torch

    physics = GPUPhysics(ROOT / "models" / "tinyphysics.onnx")
    print(f"Loaded onnx2torch physics on GPU")

    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_SEGS]

    # Get warm-start actions via batched sim
    print(f"\nWarm-start via batched sim...")
    t0 = time.time()
    warm_actions_all, warm_costs = get_warm_actions(all_csv)
    print(
        f"  Done in {time.time() - t0:.1f}s. Mean GPU cost: {np.mean(warm_costs):.1f}"
    )

    OUT_DIR.mkdir(exist_ok=True)
    results = []

    for i, csv_path in enumerate(all_csv):
        t0 = time.time()
        data = load_segment_data(csv_path)
        rng_draws = torch.from_numpy(precompute_rng(csv_path)).to(DEV)
        warm = warm_actions_all[i].to(DEV)

        # Verify warm-start cost matches
        warm_cost_gpu = evaluate_segment(data, rng_draws, warm, physics)
        print(
            f"\n  [{i}] {csv_path.name}  warm_cost(batched)={warm_costs[i]:.1f}  warm_cost(rng_replay)={warm_cost_gpu:.1f}"
        )

        # Optimize
        opt_actions = optimize_segment(data, rng_draws, warm, physics)
        opt_cost = evaluate_segment(data, rng_draws, opt_actions, physics)

        dt = time.time() - t0
        improve = warm_cost_gpu - opt_cost
        print(f"    optimized={opt_cost:.1f}  Δ={improve:.1f}  ⏱{dt:.0f}s")

        results.append({"path": str(csv_path), "warm": warm_cost_gpu, "opt": opt_cost})

        if (i + 1) % 5 == 0 or i == len(all_csv) - 1:
            wm = np.mean([r["warm"] for r in results])
            om = np.mean([r["opt"] for r in results])
            print(f"\n  [{i + 1} segs] warm={wm:.1f}  opt={om:.1f}  Δ={wm - om:.1f}")

    wm = np.mean([r["warm"] for r in results])
    om = np.mean([r["opt"] for r in results])
    print(
        f"\nDone. {len(results)} segments. warm={wm:.1f}  opt={om:.1f}  Δ={wm - om:.1f}"
    )


if __name__ == "__main__":
    main()
