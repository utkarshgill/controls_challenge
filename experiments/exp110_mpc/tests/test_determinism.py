"""Definitive test: run MPC on 1 route, then replay stored actions, compare costs.
Must run on GPU."""

import os, sys, torch, numpy as np

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
N_CTRL = COST_END_IDX - CONTROL_START_IDX

mdl = ROOT / "models" / "tinyphysics.onnx"
ort_sess = make_ort_session(str(mdl))
csvs = sorted((ROOT / "data").glob("*.csv"))[:1]
csv_cache = CSVCache([str(f) for f in csvs])

# Run 1: zero controller, capture actions from sim
data1, rng1 = csv_cache.slice(csvs)
sim1 = BatchedSimulator(
    str(mdl), ort_session=ort_sess, cached_data=data1, cached_rng=rng1
)
stored = torch.zeros((1, N_CTRL), dtype=torch.float64, device="cuda")


def ctrl1(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(1, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(1, dtype=torch.float64, device="cuda")
    act = torch.zeros(1, dtype=torch.float64, device="cuda")  # zero action
    stored[:, ci] = act
    return act


cost1 = sim1.rollout(ctrl1)["total_cost"]
print(f"Run 1 (zero ctrl): cost={cost1[0]:.4f}")

# Run 2: replay stored actions
data2, rng2 = csv_cache.slice(csvs)
sim2 = BatchedSimulator(
    str(mdl), ort_session=ort_sess, cached_data=data2, cached_rng=rng2
)


def ctrl2(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(1, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(1, dtype=torch.float64, device="cuda")
    return stored[:, ci]


cost2 = sim2.rollout(ctrl2)["total_cost"]
print(f"Run 2 (replay):    cost={cost2[0]:.4f}")
print(f"Match: {np.allclose(cost1, cost2)}")

# Compare trajectories
la1 = sim1.current_lataccel_history[0].cpu().numpy()
la2 = sim2.current_lataccel_history[0].cpu().numpy()
mae = np.mean(np.abs(la1 - la2))
print(f"Trajectory MAE: {mae:.10f}")

if mae > 1e-6:
    for i in range(len(la1)):
        if abs(la1[i] - la2[i]) > 1e-6:
            print(f"  First diff at index {i}: {la1[i]:.10f} vs {la2[i]:.10f}")
            break
