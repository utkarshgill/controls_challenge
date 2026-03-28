# exp100 — GPU coordinate descent with RNG replay through real TRT sim
#
# ksd3's approach but on GPU:
# - For each step: try K candidate actions, simulate H=20 steps forward
# - RNG-aware: use known uniform draws for exact stochastic evaluation
# - 2 passes with decreasing search radius (0.15, then 0.075)
# - PPO policy warm start
# - All segments batched in parallel using TRT BatchedPhysicsModel

import numpy as np, os, sys, time, torch, torch.nn.functional as F
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
from tinyphysics_batched import (
    BatchedSimulator,
    BatchedPhysicsModel,
    CSVCache,
    make_ort_session,
)
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
    FUTURE_K,
)

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH
BINS = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)

MPC_K = int(os.getenv("MPC_K", "40"))
MPC_H = int(os.getenv("MPC_H", "20"))
NUM_PASSES = int(os.getenv("NUM_PASSES", "2"))
RADII = [float(x) for x in os.getenv("RADII", "0.15,0.075").split(",")]
N_SEGS = int(os.getenv("N_SEGS", "100"))
SEG_BATCH = int(os.getenv("SEG_BATCH", "5"))
DELTA_SCALE = 0.25

EXP_DIR = Path(__file__).parent
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


def get_warm_start(csv_paths):
    """Get PPO policy warm-start actions via batched sim."""
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


def coord_descent_batch(csv_paths, warm_actions, mdl_path, ort_session, csv_cache):
    """Coordinate descent on a batch of segments using the real TRT sim.

    For each pass, for each step: try K perturbations of the current action,
    simulate H steps forward with RNG-aware TRT physics, pick the best.
    """
    R = len(csv_paths)
    K = MPC_K + 1  # K perturbations + 1 original
    H = MPC_H

    actions = warm_actions.clone()  # (R, N_CTRL)

    # MPC physics model (separate instance for lookahead)
    mpc_phys = BatchedPhysicsModel(str(mdl_path), ort_session=ort_session)

    for pass_num in range(NUM_PASSES):
        radius = RADII[min(pass_num, len(RADII) - 1)]

        # Run full rollout to get histories at each step
        csv_tiled = list(csv_paths)
        data, rng = csv_cache.slice(csv_tiled)
        sim = BatchedSimulator(
            str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
        )
        dg = sim.data_gpu

        # Execute rollout with current actions to build history
        h_act = torch.zeros((R, HIST_LEN), dtype=torch.float64, device="cuda")
        hist_head = HIST_LEN - 1
        improved = 0

        def ctrl_build(step_idx, sim_ref):
            nonlocal hist_head, improved
            if step_idx < CONTROL_START_IDX:
                next_head = (hist_head + 1) % HIST_LEN
                h_act[:, next_head] = 0.0
                hist_head = next_head
                return torch.zeros(R, dtype=torch.float64, device="cuda")
            ci = step_idx - CONTROL_START_IDX
            act = (
                actions[:, ci].double()
                if ci < N_CTRL
                else torch.zeros(R, dtype=torch.float64, device="cuda")
            )
            next_head = (hist_head + 1) % HIST_LEN
            h_act[:, next_head] = act
            hist_head = next_head

            # At each control step, run the H-step lookahead MPC
            if ci < N_CTRL:
                base = actions[:, ci]  # (R,) current action at this step
                perturbs = torch.randn(R, MPC_K, device="cuda") * radius
                candidates = (base.unsqueeze(1) + perturbs).clamp(
                    STEER_RANGE[0], STEER_RANGE[1]
                )
                candidates = torch.cat(
                    [candidates, base.unsqueeze(1)], dim=1
                )  # (R, K) with original

                # Build context from sim histories
                start = max(0, step_idx - CL + 1)
                act_hist = sim_ref.action_history[:, start:step_idx].float()
                state_hist = sim_ref.state_history[:, start : step_idx + 1, :3].float()
                pred_hist = sim_ref.current_lataccel_history[
                    :, max(0, step_idx - CL) : step_idx
                ].float()
                pa = CL - 1 - act_hist.shape[1]
                ps = CL - state_hist.shape[1]
                pp = CL - pred_hist.shape[1]
                if pa > 0:
                    act_hist = F.pad(act_hist, (pa, 0))
                if ps > 0:
                    state_hist = F.pad(state_hist, (0, 0, ps, 0))
                if pp > 0:
                    pred_hist = F.pad(pred_hist, (pp, 0))

                cur_la = sim_ref.current_lataccel.float()

                # Tile for K candidates
                NK = R * K
                act_h = act_hist.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
                st_h = state_hist.unsqueeze(1).expand(-1, K, -1, -1).reshape(NK, CL, 3)
                pr_h = pred_hist.unsqueeze(1).expand(-1, K, -1).reshape(NK, -1)
                cur_la_k = cur_la.unsqueeze(1).expand(-1, K).reshape(NK)
                cand_flat = candidates.reshape(NK)
                prev_la = cur_la_k.clone()

                costs = torch.zeros(NK, device="cuda")

                for h in range(H):
                    s = step_idx + h
                    if s >= dg["target_lataccel"].shape[1]:
                        break

                    # Action: candidate for h=0, stored action for h>0
                    if h == 0:
                        cur_steer = cand_flat
                    else:
                        s_ci = s - CONTROL_START_IDX
                        if 0 <= s_ci < N_CTRL:
                            cur_steer = (
                                actions[:, s_ci]
                                .float()
                                .unsqueeze(1)
                                .expand(-1, K)
                                .reshape(NK)
                            )
                        else:
                            cur_steer = cand_flat * 0

                    a_ctx = torch.cat([act_h, cur_steer.unsqueeze(1)], dim=1)
                    s_idx = min(s, dg["roll_lataccel"].shape[1] - 1)
                    new_st = torch.stack(
                        [
                            dg["roll_lataccel"][:, s_idx]
                            .float()
                            .unsqueeze(1)
                            .expand(-1, K)
                            .reshape(NK),
                            dg["v_ego"][:, s_idx]
                            .float()
                            .unsqueeze(1)
                            .expand(-1, K)
                            .reshape(NK),
                            dg["a_ego"][:, s_idx]
                            .float()
                            .unsqueeze(1)
                            .expand(-1, K)
                            .reshape(NK),
                        ],
                        dim=-1,
                    )
                    s_ctx = torch.cat([st_h[:, 1:], new_st.unsqueeze(1)], dim=1)
                    full_states = torch.cat([a_ctx.unsqueeze(-1), s_ctx], dim=-1)
                    tokens = (
                        torch.bucketize(pr_h.clamp(-5, 5), BINS, right=False)
                        .clamp(0, VOCAB_SIZE - 1)
                        .long()
                    )

                    # RNG-aware physics step
                    rng_idx = s - CL
                    if 0 <= rng_idx < sim_ref._rng_all_gpu.shape[0]:
                        u = sim_ref._rng_all_gpu[rng_idx]
                        u_tiled = u.unsqueeze(1).expand(-1, K).reshape(NK)
                    else:
                        u_tiled = torch.rand(NK, device="cuda", dtype=torch.float64)

                    sampled = mpc_phys._predict_gpu(
                        {"states": full_states, "tokens": tokens},
                        temperature=0.8,
                        rng_u=u_tiled,
                    )
                    sampled_la = BINS[sampled].float()
                    sampled_la = sampled_la.clamp(
                        cur_la_k - MAX_ACC_DELTA, cur_la_k + MAX_ACC_DELTA
                    )

                    if CONTROL_START_IDX <= s < COST_END_IDX:
                        tgt = (
                            dg["target_lataccel"][:, s]
                            .float()
                            .unsqueeze(1)
                            .expand(-1, K)
                            .reshape(NK)
                        )
                        costs += (
                            (tgt - sampled_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
                        )
                        if h > 0:
                            costs += ((sampled_la - prev_la) / DEL_T) ** 2 * 100

                    prev_la = sampled_la
                    act_h = a_ctx[:, 1:]
                    st_h = s_ctx
                    pr_h = torch.cat([pr_h[:, 1:], sampled_la.unsqueeze(1)], dim=1)
                    cur_la_k = sampled_la

                # Pick best per segment
                cost_2d = costs.view(R, K)
                best_idx = cost_2d.argmin(dim=1)
                best_action = candidates[torch.arange(R, device="cuda"), best_idx]

                # Update if improved
                orig_idx = K - 1  # original action is last candidate
                changed = best_idx != orig_idx
                if changed.any():
                    actions[changed, ci] = best_action[changed]
                    improved += changed.sum().item()

            return (
                actions[:, ci].double()
                if ci < N_CTRL
                else torch.zeros(R, dtype=torch.float64, device="cuda")
            )

        sim.rollout(ctrl_build)
        print(
            f"      pass {pass_num}: radius={radius:.3f}  improved={improved}/{N_CTRL}"
        )

    # Final eval with optimized actions
    data2, rng2 = csv_cache.slice(csv_paths)
    sim2 = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data2, cached_rng=rng2
    )

    def ctrl_eval(step_idx, sim_ref):
        if step_idx < CONTROL_START_IDX:
            return torch.zeros(R, dtype=torch.float64, device="cuda")
        ci = step_idx - CONTROL_START_IDX
        return (
            actions[:, ci].double()
            if ci < N_CTRL
            else torch.zeros(R, dtype=torch.float64, device="cuda")
        )

    final_costs = sim2.rollout(ctrl_eval)["total_cost"]

    return final_costs, actions.cpu().numpy()


def main():
    print(f"exp100 — GPU coordinate descent with RNG replay")
    print(f"  K={MPC_K}  H={MPC_H}  passes={NUM_PASSES}  radii={RADII}")
    print(f"  segs={N_SEGS}  batch={SEG_BATCH}")

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_SEGS]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Warm start
    print("Getting PPO warm start...")
    warm_actions, warm_costs = get_warm_start(all_csv)
    print(f"  warm: {np.mean(warm_costs):.1f}")

    EXP_DIR.mkdir(exist_ok=True)
    all_costs = []
    all_actions = []

    for i in range(0, len(all_csv), SEG_BATCH):
        batch_paths = all_csv[i : i + SEG_BATCH]
        batch_warm = warm_actions[i : i + len(batch_paths)]
        print(f"\n  Segments {i}..{i + len(batch_paths) - 1}")
        t0 = time.time()
        costs, opt_act = coord_descent_batch(
            batch_paths, batch_warm, mdl_path, ort_sess, csv_cache
        )
        dt = time.time() - t0
        all_costs.extend(costs.tolist())
        all_actions.append(opt_act)
        print(f"    cost={np.mean(costs):.1f}  ⏱{dt:.0f}s")

    mc = np.mean(all_costs)
    print(f"\nDone. {len(all_csv)} segments. Mean cost: {mc:.1f}")
    print(f"Warm start was: {np.mean(warm_costs):.1f}")

    np.savez(
        EXP_DIR / "optimized_actions.npz",
        actions=np.concatenate(all_actions),
        costs=np.array(all_costs),
        paths=[str(p) for p in all_csv],
    )
    print(f"Saved to {EXP_DIR / 'optimized_actions.npz'}")


if __name__ == "__main__":
    main()
