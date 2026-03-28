# exp084 — Coordinate descent MPC with RNG replay
#
# Per-step optimization using real sim with exact RNG replay.
# Warm-started from exp055 policy actions.
#
# For each segment:
#   1. Roll out policy → get warm-start actions
#   2. For each scored step (100..499), left to right:
#      a. Save sim state + RNG state
#      b. Try N candidate perturbations of current action
#      c. Simulate H steps forward for each candidate
#      d. Keep the best
#   3. 2 passes with decreasing search radius
#   4. Save optimized actions + collect (obs, action) for BC
#
# Based on ksd3's approach (39.35 with PID warm start).
# With our PPO warm start at 42, this should reach ~30.

import numpy as np
import onnxruntime as ort
import pandas as pd
import os, sys, time, copy, json
from hashlib import md5
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ACC_G = 9.81
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
N_SCORED = COST_END_IDX - CONTROL_START_IDX  # 400

# ── Config ────────────────────────────────────────────────────
N_CANDIDATES = int(os.getenv("N_CAND", "40"))
HORIZON = int(os.getenv("HORIZON", "20"))
NUM_PASSES = int(os.getenv("NUM_PASSES", "2"))
SEARCH_RADII = [float(x) for x in os.getenv("RADII", "0.15,0.075").split(",")]
N_SEGMENTS = int(os.getenv("N_SEGS", "5000"))
N_WORKERS = int(os.getenv("WORKERS", "1"))  # CPU parallelism

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "optimized"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)

# ── Policy extraction (runs once to get warm-start actions) ───
DELTA_SCALE = 0.25
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS = 4
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
OBS_DIM = 256


# ══════════════════════════════════════════════════════════════
#  RNG helpers
# ══════════════════════════════════════════════════════════════


def save_rng():
    s = np.random.get_state()
    return (s[0], s[1].copy(), s[2], s[3], s[4])


def restore_rng(s):
    np.random.set_state((s[0], s[1].copy(), s[2], s[3], s[4]))


# ══════════════════════════════════════════════════════════════
#  Physics step (CPU, matches tinyphysics.py exactly)
# ══════════════════════════════════════════════════════════════


class PhysicsSim:
    def __init__(self, model_path="models/tinyphysics.onnx"):
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.ort = ort.InferenceSession(
            str(model_path), options, ["CPUExecutionProvider"]
        )

    def step(self, state_hist, action_hist, lataccel_hist, data_row, step_idx, action):
        """One physics step. Consumes one np.random call. Modifies lists in place."""
        state = (data_row["roll_lataccel"], data_row["v_ego"], data_row["a_ego"])
        state_hist.append(state)
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        action_hist.append(action)

        CL = CONTEXT_LENGTH
        sim_states = state_hist[-CL:]
        sim_actions = action_hist[-CL:]
        past_preds = lataccel_hist[-CL:]

        tokenized = np.digitize(np.clip(past_preds, -5, 5), self.bins, right=True)
        raw = [list(s) for s in sim_states]
        states_arr = np.column_stack([sim_actions, raw]).astype(np.float32)

        res = self.ort.run(
            None,
            {
                "states": states_arr[np.newaxis],
                "tokens": tokenized[np.newaxis].astype(np.int64),
            },
        )[0]

        logits = res[0, -1, :].astype(np.float64) / 0.8
        if np.any(np.isnan(logits)):
            # Fallback: uniform distribution if model produces NaN
            sample = np.random.choice(VOCAB_SIZE)
        else:
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            sample = np.random.choice(VOCAB_SIZE, p=probs)
        pred = self.bins[sample]

        prev_la = lataccel_hist[-1]
        pred = np.clip(pred, prev_la - MAX_ACC_DELTA, prev_la + MAX_ACC_DELTA)

        new_la = pred if step_idx >= CONTROL_START_IDX else data_row["target_lataccel"]
        lataccel_hist.append(new_la)
        return new_la


# ══════════════════════════════════════════════════════════════
#  Get policy warm-start actions for a segment
# ══════════════════════════════════════════════════════════════


def get_all_policy_actions(csv_paths):
    """Run exp055 policy on all segments via batched sim.
    Returns (N, 400) numpy array of actions and (N,) costs.
    Imports exp055's rollout directly to avoid code duplication.
    """
    import torch
    from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
    from experiments.exp055_batch_of_batch.train import (
        ActorCritic,
        rollout as exp055_rollout,
    )

    ac = ActorCritic().to("cuda")
    ckpt = torch.load(BASE_PT, weights_only=False, map_location="cuda")
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", DELTA_SCALE))

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    csv_cache = CSVCache([str(f) for f in csv_paths])

    # Run deterministic rollout — exp055_rollout returns list of costs when deterministic
    # But we also need actions. We'll run via BatchedSimulator directly using exp055's controller logic.
    # The simplest: run exp055_rollout to get costs, then re-run to get actions.
    # Actually, let's just build a thin wrapper that captures action_history.

    from experiments.exp055_batch_of_batch.train import (
        _precompute_future_windows,
        fill_obs,
        HIST_LEN,
        OBS_DIM,
        DELTA_SCALE_MAX,
    )
    import torch.nn.functional as F

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
    actions_np = (
        sim.action_history[:, CONTROL_START_IDX:COST_END_IDX]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return actions_np, costs


# ══════════════════════════════════════════════════════════════
#  Coordinate descent MPC optimizer
# ══════════════════════════════════════════════════════════════


def optimize_segment(args):
    data_path, warm_actions = args
    sim = PhysicsSim(str(ROOT / "models" / "tinyphysics.onnx"))

    df = pd.read_csv(str(data_path))
    data = pd.DataFrame(
        {
            "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
            "v_ego": df["vEgo"].values,
            "a_ego": df["aEgo"].values,
            "target_lataccel": df["targetLateralAcceleration"].values,
            "steer_command": -df["steerCommand"].values,
        }
    )
    n_steps = len(data)
    seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4

    # Working copy of actions
    n_ctrl = n_steps - CONTROL_START_IDX
    actions = np.zeros(n_ctrl)
    n_copy = min(len(warm_actions), n_ctrl)
    actions[:n_copy] = warm_actions[:n_copy]

    cand_rng = np.random.RandomState(seed + 99999)

    for pass_num in range(NUM_PASSES):
        radius = SEARCH_RADII[min(pass_num, len(SEARCH_RADII) - 1)]

        # Init sim from scratch
        np.random.seed(seed)
        sh, ah, lh = [], [], []
        for i in range(CONTEXT_LENGTH):
            row = data.iloc[i]
            sh.append((row["roll_lataccel"], row["v_ego"], row["a_ego"]))
            ah.append(row["steer_command"])
            lh.append(row["target_lataccel"])

        # Warmup
        for step_idx in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            sim.step(
                sh,
                ah,
                lh,
                data.iloc[step_idx],
                step_idx,
                data.iloc[step_idx]["steer_command"],
            )

        # Optimize scored steps
        improved = 0
        for t in range(N_SCORED):
            step_idx = CONTROL_START_IDX + t

            saved_sh = list(sh)
            saved_ah = list(ah)
            saved_lh = list(lh)
            saved_rng = save_rng()

            base = actions[t]
            perturbs = cand_rng.randn(N_CANDIDATES) * radius
            candidates = np.clip(base + perturbs, STEER_RANGE[0], STEER_RANGE[1])
            candidates = np.append(candidates, base)  # always include current

            best_cost = float("inf")
            best_action = base

            for cand in candidates:
                restore_rng(saved_rng)
                # Evaluate H steps forward
                tsh = list(saved_sh)
                tah = list(saved_ah)
                tlh = list(saved_lh)

                cost_lat, cost_jerk = 0.0, 0.0
                n_terms = 0
                prev_la = None

                for h in range(HORIZON):
                    s = step_idx + h
                    if s >= n_steps:
                        break
                    act = (
                        cand
                        if h == 0
                        else (
                            actions[s - CONTROL_START_IDX]
                            if s >= CONTROL_START_IDX
                            and (s - CONTROL_START_IDX) < len(actions)
                            else 0.0
                        )
                    )
                    new_la = sim.step(tsh, tah, tlh, data.iloc[s], s, act)

                    if CONTROL_START_IDX <= s < COST_END_IDX:
                        target = data.iloc[s]["target_lataccel"]
                        cost_lat += (target - new_la) ** 2
                        if prev_la is not None:
                            cost_jerk += ((new_la - prev_la) / DEL_T) ** 2
                        prev_la = new_la
                        n_terms += 1

                if n_terms > 0:
                    cost = (
                        cost_lat / n_terms * 100 * LAT_ACCEL_COST_MULTIPLIER
                        + cost_jerk / max(n_terms - 1, 1) * 100
                    )
                else:
                    cost = 0.0

                if cost < best_cost:
                    best_cost = cost
                    best_action = cand

            if best_action != base:
                actions[t] = best_action
                improved += 1

            # Advance with best action
            sh = saved_sh
            ah = saved_ah
            lh = saved_lh
            restore_rng(saved_rng)
            sim.step(sh, ah, lh, data.iloc[step_idx], step_idx, best_action)

        # Steps after scoring window
        for step_idx in range(COST_END_IDX, n_steps):
            cidx = step_idx - CONTROL_START_IDX
            act = actions[cidx] if cidx < len(actions) else 0.0
            sim.step(sh, ah, lh, data.iloc[step_idx], step_idx, act)

    # Final cost
    final_cost = evaluate_actions(str(data_path), actions)
    return str(data_path), actions, final_cost


# ══════════════════════════════════════════════════════════════
#  RNG replay evaluator (inline)
# ══════════════════════════════════════════════════════════════


def evaluate_actions(data_path, actions):
    sim = PhysicsSim(str(ROOT / "models" / "tinyphysics.onnx"))
    df = pd.read_csv(str(data_path))
    data = pd.DataFrame(
        {
            "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
            "v_ego": df["vEgo"].values,
            "a_ego": df["aEgo"].values,
            "target_lataccel": df["targetLateralAcceleration"].values,
            "steer_command": -df["steerCommand"].values,
        }
    )
    seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

    sh, ah, lh = [], [], []
    for i in range(CONTEXT_LENGTH):
        row = data.iloc[i]
        sh.append((row["roll_lataccel"], row["v_ego"], row["a_ego"]))
        ah.append(row["steer_command"])
        lh.append(row["target_lataccel"])

    for step_idx in range(CONTEXT_LENGTH, len(data)):
        row = data.iloc[step_idx]
        cidx = step_idx - CONTROL_START_IDX
        if step_idx >= CONTROL_START_IDX and 0 <= cidx < len(actions):
            action = actions[cidx]
        else:
            action = row["steer_command"]
        sim.step(sh, ah, lh, row, step_idx, action)

    targets = np.array(
        [
            data.iloc[i]["target_lataccel"]
            for i in range(CONTROL_START_IDX, COST_END_IDX)
        ]
    )
    preds = np.array(lh[CONTROL_START_IDX:COST_END_IDX])
    lat_cost = np.mean((targets - preds) ** 2) * 100
    jerk_cost = np.mean((np.diff(preds) / DEL_T) ** 2) * 100
    return lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def main():
    print(f"exp084 — Coordinate descent MPC with RNG replay")
    print(f"  candidates={N_CANDIDATES}  horizon={HORIZON}  passes={NUM_PASSES}")
    print(f"  radii={SEARCH_RADII}  segments={N_SEGMENTS}  workers={N_WORKERS}")

    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_SEGMENTS]
    OUT_DIR.mkdir(exist_ok=True)

    # Get policy warm-start actions for ALL segments via batched GPU sim
    print(f"\nExtracting policy warm-start actions via batched sim...")
    t0 = time.time()
    all_warm_actions, all_warm_costs_gpu = get_all_policy_actions(all_csv)
    print(
        f"  Done in {time.time() - t0:.1f}s. Mean GPU cost: {np.mean(all_warm_costs_gpu):.1f}"
    )

    # Now optimize each segment with coordinate descent
    print(f"\nOptimizing segments with coordinate descent MPC...")
    all_results = []
    for i, csv_path in enumerate(all_csv):
        t_seg = time.time()
        warm_actions = all_warm_actions[i]
        warm_cost = evaluate_actions(str(csv_path), warm_actions)

        _, opt_actions, opt_cost = optimize_segment((csv_path, warm_actions))
        dt = time.time() - t_seg

        improve = warm_cost - opt_cost
        print(
            f"  [{i:4d}] {csv_path.name}  warm={warm_cost:.1f}  opt={opt_cost:.1f}"
            f"  Δ={improve:.1f}  ⏱{dt:.0f}s"
        )

        all_results.append(
            {
                "path": str(csv_path),
                "warm_cost": warm_cost,
                "opt_cost": opt_cost,
                "actions": opt_actions,
            }
        )

        # Periodic save
        if (i + 1) % 10 == 0 or i == len(all_csv) - 1:
            warm_costs = [r["warm_cost"] for r in all_results]
            opt_costs = [r["opt_cost"] for r in all_results]
            print(
                f"\n  [{i + 1} segs] warm={np.mean(warm_costs):.1f}  opt={np.mean(opt_costs):.1f}"
                f"  Δ={np.mean(warm_costs) - np.mean(opt_costs):.1f}\n"
            )

            np.savez(
                OUT_DIR / "optimized.npz",
                paths=[r["path"] for r in all_results],
                warm_costs=np.array(warm_costs),
                opt_costs=np.array(opt_costs),
                actions=np.stack([r["actions"] for r in all_results]),
            )

    wm = np.mean([r["warm_cost"] for r in all_results])
    om = np.mean([r["opt_cost"] for r in all_results])
    print(f"\nDone. {len(all_results)} segments.")
    print(f"  Warm: {wm:.1f}  Optimized: {om:.1f}  Δ: {wm - om:.1f}")


if __name__ == "__main__":
    main()
