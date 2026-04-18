"""Collect (obs, delta) pairs by replaying MPC actions through the OFFICIAL CPU sim.
Slow but 100% correct — matches the official eval exactly."""

import sys, os, numpy as np, torch, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    DEL_T,
    STEER_RANGE,
    ACC_G,
    FUTURE_PLAN_STEPS,
    State,
    FuturePlan,
)
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
)

N_CTRL = COST_END_IDX - CONTROL_START_IDX
ACTIONS_NPZ = os.getenv(
    "ACTIONS_NPZ", str(ROOT / "experiments/exp110_mpc/checkpoints/actions_5k_final.npz")
)
N_ROUTES = int(os.getenv("N_ROUTES", "5000"))

S_LAT = 5.0
S_STEER = 2.0
S_VEGO = 40.0
S_AEGO = 4.0
S_ROLL = 2.0
S_CURV = 0.02
FUTURE_K = 50


def process_route(args):
    """Replay one route, collect obs and deltas matching exp055's fill_obs."""
    route_path, actions = args

    import numpy as np
    from tinyphysics import (
        TinyPhysicsModel,
        TinyPhysicsSimulator,
        CONTROL_START_IDX,
        COST_END_IDX,
        CONTEXT_LENGTH,
        DEL_T,
        STEER_RANGE,
        ACC_G,
        FUTURE_PLAN_STEPS,
    )

    N_CTRL = COST_END_IDX - CONTROL_START_IDX
    HIST_LEN = 20
    FUTURE_K = 50

    model = TinyPhysicsModel(str(ROOT / "models" / "tinyphysics.onnx"), debug=False)

    # Ring buffers matching exp055
    h_act = np.zeros(HIST_LEN, dtype=np.float64)
    h_act32 = np.zeros(HIST_LEN, dtype=np.float32)
    h_lat = np.zeros(HIST_LEN, dtype=np.float32)
    h_error = np.zeros(HIST_LEN, dtype=np.float32)
    err_sum = np.float32(0.0)
    hist_head = HIST_LEN - 1

    obs_list = []
    delta_list = []

    class CollectController:
        def __init__(self):
            self.step = 0

        def update(self, target_la, current_la, state, future_plan=None):
            nonlocal hist_head, err_sum
            self.step += 1
            step_idx = self.step + CONTEXT_LENGTH - 1

            cur32 = np.float32(current_la)
            tgt32 = np.float32(target_la)
            error = np.float32(tgt32 - cur32)

            next_head = (hist_head + 1) % HIST_LEN
            old_err = h_error[next_head]
            h_error[next_head] = error
            err_sum = err_sum + error - old_err
            ei = err_sum * np.float32(DEL_T / HIST_LEN)

            if step_idx < CONTROL_START_IDX:
                h_act[next_head] = 0.0
                h_act32[next_head] = 0.0
                h_lat[next_head] = cur32
                hist_head = next_head
                return 0.0

            ci = step_idx - CONTROL_START_IDX
            if ci >= N_CTRL:
                return 0.0

            # Build obs matching exp055's fill_obs layout (256 dims):
            # Core features (C=16): target, current, error, roll, v, a,
            #   curvature_tgt, curvature_cur, curvature_err, friction,
            #   error_integral, target_deriv, action_deriv, headroom_lo, headroom_hi, jerk
            roll = np.float32(state.roll_lataccel)
            v = np.float32(state.v_ego)
            a = np.float32(state.a_ego)
            v2 = max(v * v, np.float32(1.0))
            k_tgt = (tgt32 - roll) / v2
            k_cur = (cur32 - roll) / v2
            k_err = k_tgt - k_cur
            fric = np.sqrt(cur32**2 + a**2) / 7.0

            prev_act = np.float32(h_act32[hist_head])
            prev_lat = np.float32(h_lat[hist_head])
            jerk = (cur32 - prev_lat) / DEL_T

            # Build the 16 core features
            C = 16
            core = np.array(
                [
                    tgt32 / S_LAT,
                    cur32 / S_LAT,
                    error / S_LAT,
                    roll / S_ROLL,
                    v / S_VEGO,
                    a / S_AEGO,
                    k_tgt / S_CURV,
                    k_cur / S_CURV,
                    k_err / S_CURV,
                    fric,
                    ei / S_LAT,
                    0.0,  # target_deriv placeholder
                    0.0,  # action_deriv placeholder
                    (STEER_RANGE[1] - prev_act) / S_STEER,  # headroom hi
                    (prev_act - STEER_RANGE[0]) / S_STEER,  # headroom lo
                    jerk / S_LAT,
                ],
                dtype=np.float32,
            )

            # Action history (HIST_LEN=20)
            # Unroll ring buffer
            idx = np.arange(HIST_LEN)
            order = (idx + hist_head + 1) % HIST_LEN
            act_hist = h_act32[order] / S_STEER

            # Lataccel history (HIST_LEN=20)
            lat_hist = h_lat[order] / S_LAT

            # Future plan (FUTURE_K * 4 = 200)
            if future_plan and len(future_plan.lataccel) >= FUTURE_K:
                fut = np.concatenate(
                    [
                        np.array(future_plan.lataccel[:FUTURE_K], dtype=np.float32)
                        / S_LAT,
                        np.array(future_plan.roll_lataccel[:FUTURE_K], dtype=np.float32)
                        / S_ROLL,
                        np.array(future_plan.v_ego[:FUTURE_K], dtype=np.float32)
                        / S_VEGO,
                        np.array(future_plan.a_ego[:FUTURE_K], dtype=np.float32)
                        / S_AEGO,
                    ]
                )
            else:
                fut = np.zeros(FUTURE_K * 4, dtype=np.float32)

            # Full obs: core(16) + act_hist(20) + lat_hist(20) + future(200) = 256
            obs = np.concatenate([core, act_hist, lat_hist, fut]).clip(-5, 5)

            action = float(actions[ci])
            delta = action - float(h_act[hist_head])

            obs_list.append(obs)
            delta_list.append(np.float32(delta))

            h_act[next_head] = action
            h_act32[next_head] = np.float32(action)
            h_lat[next_head] = cur32
            hist_head = next_head
            return action

    ctrl = CollectController()
    sim = TinyPhysicsSimulator(model, str(route_path), ctrl)
    cost = sim.rollout()

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(delta_list, dtype=np.float32),
        cost["total_cost"],
    )


def main():
    print(f"Loading actions from {ACTIONS_NPZ}")
    actions_data = np.load(ACTIONS_NPZ)
    actions_dict = {k: actions_data[k] for k in actions_data.files}
    print(f"  {len(actions_dict)} routes")

    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]

    args_list = [(f, actions_dict[f.name]) for f in all_csv if f.name in actions_dict]
    print(f"  Processing {len(args_list)} routes on CPU (parallel)...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_route, args_list))

    all_obs = np.concatenate([r[0] for r in results])
    all_deltas = np.concatenate([r[1] for r in results])
    costs = [r[2] for r in results]

    print(
        f"  {len(all_obs)} samples, mean cost={np.mean(costs):.1f}, ⏱{time.time() - t0:.0f}s"
    )

    # Save
    save_path = ROOT / "experiments" / "exp110_mpc" / "checkpoints" / "bc_data.npz"
    np.savez(save_path, obs=all_obs, deltas=all_deltas, costs=np.array(costs))
    print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
