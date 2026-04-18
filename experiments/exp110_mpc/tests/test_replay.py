"""Quick test: replay 10 routes from actions.npz and verify cost matches."""

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
mdl_path = "models/tinyphysics.onnx"
ort_sess = make_ort_session(mdl_path)

# Load actions
npz_path = os.getenv(
    "ACTIONS_NPZ", "experiments/exp110_mpc/checkpoints/actions_8xh100.npz"
)
actions_data = np.load(npz_path)
print(f"Loaded {len(actions_data.files)} routes from {npz_path}")
all_csv = sorted((ROOT / "data").glob("*.csv"))[:10]
csv_cache = CSVCache([str(f) for f in all_csv])

# Check if actions exist for these routes
for f in all_csv:
    if f.name in actions_data:
        a = actions_data[f.name]
        print(f"  {f.name}: len={len(a)} range=[{a.min():.3f}, {a.max():.3f}]")
    else:
        print(f"  {f.name}: MISSING")

# Replay
data, rng = csv_cache.slice(all_csv)
sim = BatchedSimulator(
    str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
)
N = sim.N
stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
for i, f in enumerate(all_csv):
    stored[i] = torch.from_numpy(actions_data[f.name]).float().to("cuda")


def ctrl(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    return stored[:, ci].double()


costs = sim.rollout(ctrl)["total_cost"]
print(f"\nReplay costs: mean={np.mean(costs):.1f}")
for i, c in enumerate(costs):
    print(f"  {all_csv[i].name}: {c:.1f}")
