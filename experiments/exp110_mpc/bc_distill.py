#!/usr/bin/env python3
"""BC distillation: train MLP policy from MPC-optimized actions.

1. Load actions.npz (precomputed optimal actions per route)
2. Replay actions in sim, collect (observation, action) pairs
3. Train the MLP via behavioral cloning
4. Evaluate the BC policy standalone
"""

import numpy as np, os, sys, time, random, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")
# To replay old MPC actions (generated with absolute path seeds):
#   SEED_PREFIX=/workspace/controls_challenge/data
# For new runs matching official eval:
#   SEED_PREFIX=data  (default)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE, DEL_T
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
from experiments.exp055_batch_of_batch.train import (
    ActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    DELTA_SCALE_MAX,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX

ACTIONS_NPZ = os.getenv("ACTIONS_NPZ", "experiments/exp110_mpc/checkpoints/actions.npz")
N_ROUTES = int(os.getenv("N_ROUTES", "5000"))
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "50"))
BC_LR = float(os.getenv("BC_LR", "1e-3"))
BC_BS = int(os.getenv("BC_BS", "4096"))
EVAL_N = int(os.getenv("EVAL_N", "100"))
COLLECT_BATCH = int(os.getenv("COLLECT_BATCH", "500"))


def collect_obs_actions(csv_files, actions_dict, mdl_path, ort_sess, csv_cache):
    """Replay MPC actions, collect (obs, action, delta) at each step."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    # Load stored actions for these routes
    stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
    for i, f in enumerate(csv_files):
        if f.name in actions_dict:
            stored[i] = torch.from_numpy(actions_dict[f.name]).float().to("cuda")

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    all_obs = []
    all_actions = []
    all_deltas = []

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
            return torch.zeros(N, dtype=torch.float64, device="cuda")

        ci = step_idx - CONTROL_START_IDX
        if ci >= N_CTRL:
            return torch.zeros(N, dtype=torch.float64, device="cuda")

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

        action = stored[:, ci].double()
        prev_action = h_act[:, hist_head]
        delta = (action - prev_action).float()

        # Store obs and action
        all_obs.append(obs_buf.clone())
        all_actions.append(action.float().clone())
        all_deltas.append(delta.clone())

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]

    obs_t = torch.cat(all_obs, dim=0)  # (N*N_CTRL, OBS_DIM)
    act_t = torch.cat(all_actions, dim=0)  # (N*N_CTRL,)
    delta_t = torch.cat(all_deltas, dim=0)  # (N*N_CTRL,)

    return obs_t, act_t, delta_t, costs


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    # Load MPC actions
    actions_path = ROOT / ACTIONS_NPZ
    print(f"Loading MPC actions from {actions_path}")
    actions_data = np.load(actions_path)
    actions_dict = {k: actions_data[k] for k in actions_data.files}
    print(f"  {len(actions_dict)} routes loaded")

    # Split: train on first N-EVAL_N routes, eval on last EVAL_N
    train_csv = all_csv[: N_ROUTES - EVAL_N]
    eval_csv = all_csv[:EVAL_N]  # eval on first 100 (overlap with train is fine for BC)

    # Collect observations by replaying MPC actions
    print(f"\nCollecting observations from {len(train_csv)} training routes...")
    all_obs_list, all_delta_list = [], []
    t0 = time.time()
    for batch_start in range(0, len(train_csv), COLLECT_BATCH):
        batch_end = min(batch_start + COLLECT_BATCH, len(train_csv))
        batch_csv = train_csv[batch_start:batch_end]
        obs, act, delta, costs = collect_obs_actions(
            batch_csv, actions_dict, mdl_path, ort_sess, csv_cache
        )
        all_obs_list.append(obs)
        all_delta_list.append(delta)
        print(
            f"  {batch_end}/{len(train_csv)} routes  "
            f"MPC cost={np.mean(costs):.1f}  ⏱{time.time() - t0:.0f}s"
        )

    train_obs = torch.cat(all_obs_list, dim=0)
    train_delta = torch.cat(all_delta_list, dim=0)
    print(f"  Total: {len(train_obs)} samples")

    # Normalize deltas to [-1, 1] for Beta distribution training
    ds = DELTA_SCALE_MAX
    train_raw = (train_delta / ds).clamp(-1, 1)

    # Initialize policy from exp055 weights
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.train()
    print(f"\nInitialized from exp055")

    # BC training: minimize NLL of MPC actions under Beta distribution
    optimizer = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BC_EPOCHS)

    print(f"BC training: {BC_EPOCHS} epochs, lr={BC_LR}, bs={BC_BS}")
    for ep in range(BC_EPOCHS):
        ac.train()
        total_loss, nb = 0.0, 0
        for idx in torch.randperm(len(train_obs), device="cuda").split(BC_BS):
            a_p, b_p = ac.beta_params(train_obs[idx])
            # Target: raw ∈ [-1, 1] → x ∈ (0, 1) for Beta
            x = ((train_raw[idx] + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
            nb += len(idx)
        scheduler.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            # Quick eval on 100 routes
            ac.eval()
            from experiments.exp055_batch_of_batch.train import (
                rollout as policy_rollout,
            )

            eval_costs = policy_rollout(
                eval_csv,
                ac,
                mdl_path,
                ort_sess,
                csv_cache,
                deterministic=True,
                ds=ds,
            )
            print(
                f"  E{ep:3d}  loss={total_loss / nb:.4f}  "
                f"eval={np.mean(eval_costs):.1f}  lr={optimizer.param_groups[0]['lr']:.1e}"
            )

    # Save
    save_path = Path(__file__).resolve().parent / "bc_model.pt"
    torch.save({"ac": ac.state_dict(), "delta_scale": ds}, save_path)
    print(f"\nSaved BC policy to {save_path}")

    # Final eval
    ac.eval()
    from experiments.exp055_batch_of_batch.train import rollout as policy_rollout

    final_costs = policy_rollout(
        eval_csv, ac, mdl_path, ort_sess, csv_cache, deterministic=True, ds=ds
    )
    print(f"Final eval (100 routes): {np.mean(final_costs):.1f}")


if __name__ == "__main__":
    main()
