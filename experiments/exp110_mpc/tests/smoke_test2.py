"""Run MPC on 1 route, then immediately replay stored actions. Compare costs.
This eliminates ALL possible differences: same process, same GPU, same TRT engine."""

import os, sys, torch, numpy as np

os.environ["CUDA"] = "1"
os.environ["TRT"] = "1"
os.environ["SEED_PREFIX"] = "data"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
N_CTRL = COST_END_IDX - CONTROL_START_IDX

mdl = str(ROOT / "models" / "tinyphysics.onnx")
ort_sess = make_ort_session(mdl)
csvs = sorted((ROOT / "data").glob("*.csv"))[:1]
csv_cache = CSVCache([str(f) for f in csvs])

# ===== RUN 1: MPC-like controller that stores actions =====
data1, rng1 = csv_cache.slice(csvs)
sim1 = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data1, cached_rng=rng1)
N = sim1.N
stored = torch.zeros((N, N_CTRL), dtype=torch.float64, device="cuda")
returned = torch.zeros((N, N_CTRL), dtype=torch.float64, device="cuda")

call_count = [0]


def ctrl1(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    # Simulate what MPC does: compute an action, store it, return it
    # Use a simple heuristic (not zero, so we can see if replay matches)
    target = sim_ref.data_gpu["target_lataccel"][:, step_idx].float()
    current = sim_ref.current_lataccel.float()
    error = target - current
    action = (error * 0.3).double().clamp(-2, 2)  # simple proportional controller
    stored[:, ci] = action
    returned[:, ci] = action
    call_count[0] += 1
    return action


cost1 = sim1.rollout(ctrl1)["total_cost"]
print(f"Run 1 (proportional ctrl): cost={cost1[0]:.4f}  steps={call_count[0]}")

# Verify stored == returned
diff = (stored - returned).abs().max().item()
print(f"stored vs returned max diff: {diff}")

# ===== RUN 2: Replay stored actions =====
data2, rng2 = csv_cache.slice(csvs)
sim2 = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data2, cached_rng=rng2)


def ctrl2(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    return stored[:, ci]


cost2 = sim2.rollout(ctrl2)["total_cost"]
print(f"Run 2 (replay):            cost={cost2[0]:.4f}")

print(f"\nCost match: {np.allclose(cost1, cost2)}")
print(f"Cost diff: {abs(cost1[0] - cost2[0]):.6f}")

# Compare trajectories
la1 = sim1.current_lataccel_history.cpu().numpy()[0]
la2 = sim2.current_lataccel_history.cpu().numpy()[0]
mae = np.mean(np.abs(la1 - la2))
maxdiff = np.max(np.abs(la1 - la2))
print(f"Trajectory MAE: {mae:.10f}  Max: {maxdiff:.10f}")

if maxdiff > 1e-6:
    for i in range(len(la1)):
        if abs(la1[i] - la2[i]) > 1e-6:
            print(f"  First diff at hist index {i}: {la1[i]:.10f} vs {la2[i]:.10f}")
            break

# ===== RUN 3: Replay with float32 stored (what npz does) =====
stored_f32 = stored.float()  # simulate npz save/load
data3, rng3 = csv_cache.slice(csvs)
sim3 = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data3, cached_rng=rng3)


def ctrl3(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    return stored_f32[:, ci].double()


cost3 = sim3.rollout(ctrl3)["total_cost"]
print(f"\nRun 3 (float32 replay):    cost={cost3[0]:.4f}")
print(f"Cost diff from Run 1: {abs(cost1[0] - cost3[0]):.6f}")
