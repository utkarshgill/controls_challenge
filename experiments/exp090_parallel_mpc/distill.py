# exp090 — BC distillation from MPC data into policy
#
# Loads mpc_data.pt (winning obs/raw + baseline obs/raw from mpc.py)
# Runs NLL toward winners, anchored to baseline, for a few epochs.
# Saves updated policy. Then eval.

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

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
    FUTURE_K,
)

DEV = torch.device("cuda")

BC_LR = float(os.getenv("BC_LR", "3e-5"))
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "5"))
BC_ANCHOR = float(os.getenv("BC_ANCHOR", "0.5"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "0.5"))
MINI_BS = 25000
EVAL_N = 100

EXP_DIR = Path(__file__).parent
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


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


def main():
    # Load policy
    ac = ActorCritic().to(DEV)
    ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
    ac.load_state_dict(ckpt["ac"])
    ds = float(ckpt.get("delta_scale", DELTA_SCALE_MAX))
    print(f"Loaded policy from {BASE_PT} (Δs={ds})")

    # Load MPC shoot data
    mpc = torch.load(
        EXP_DIR / "mpc_shoot_data.pt", weights_only=False, map_location=DEV
    )
    mpc_obs = mpc["mpc_obs"].to(DEV)
    mpc_raw = mpc["mpc_raw"].to(DEV)
    print(f"Loaded MPC data: {mpc_obs.shape[0]} samples")
    print(
        f"  MPC cost: {np.mean(mpc['mpc_costs']):.1f}  baseline: {mpc['baseline']:.1f}"
    )

    # Eval before
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in va_f])

    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
    print(f"\nBefore BC: val={vm:.1f} ± {vs:.1f}")

    # BC: pure NLL toward MPC actions (no anchor — the MPC actions ARE on-distribution
    # because they come from per-step optimization on the policy's own trajectory)
    opt = optim.Adam(ac.actor.parameters(), lr=BC_LR)
    obs = mpc_obs
    raw = mpc_raw
    x_t = ((raw.unsqueeze(-1) + 1) / 2).clamp(1e-6, 1 - 1e-6)

    print(f"\nBC: lr={BC_LR}  epochs={BC_EPOCHS}")
    best_val = vm
    ac.train()
    for epoch in range(BC_EPOCHS):
        nll_sum, n = 0.0, 0
        for idx in torch.randperm(len(obs), device=DEV).split(MINI_BS):
            logits = ac.actor(obs[idx])
            a_p = F.softplus(logits[..., 0]) + 1.0
            b_p = F.softplus(logits[..., 1]) + 1.0
            dist = torch.distributions.Beta(a_p, b_p)
            nll = -dist.log_prob(x_t[idx].squeeze(-1)).mean()
            opt.zero_grad(set_to_none=True)
            nll.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP)
            opt.step()
            nll_sum += nll.item() * idx.numel()
            n += idx.numel()

        ac.eval()
        vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
        mk = ""
        if vm < best_val:
            best_val = vm
            torch.save(
                {"ac": ac.state_dict(), "delta_scale": ds},
                EXP_DIR / "distilled_model.pt",
            )
            mk = " ★"
        print(f"  epoch {epoch}: nll={nll_sum / n:.3f}  val={vm:.1f}±{vs:.1f}{mk}")
        ac.train()

    print(f"\nBest val: {best_val:.1f}")
    print(f"Saved to {EXP_DIR / 'distilled_model.pt'}")


if __name__ == "__main__":
    main()
