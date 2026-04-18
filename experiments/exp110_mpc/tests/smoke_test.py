"""Smoke test: replay 8xh100 actions with matching seeds, verify cost ~19.5"""

import os, sys, torch, numpy as np

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")
# Match the old run's seed: paths were /workspace/controls_challenge/data/00000.csv
os.environ["SEED_PREFIX"] = "/workspace/controls_challenge/data"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session
N_CTRL = COST_END_IDX - CONTROL_START_IDX

mdl = str(ROOT / "models" / "tinyphysics.onnx")
ort_sess = make_ort_session(mdl)

csvs = sorted((ROOT / "data").glob("*.csv"))[:10]
csv_cache = CSVCache([str(f) for f in csvs])

# Load old actions
npz = np.load(
    ROOT / "experiments" / "exp110_mpc" / "checkpoints" / "8xh100_final_actions.npz"
)
print(f"Loaded {len(npz.files)} routes")

# Replay
data, rng = csv_cache.slice(csvs)
sim = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data, cached_rng=rng)
N = sim.N
stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")
for i, f in enumerate(csvs):
    if f.name in npz:
        stored[i] = torch.from_numpy(npz[f.name]).float().to("cuda")


def ctrl(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    return stored[:, ci].double()


costs = sim.rollout(ctrl)["total_cost"]
print(f"\nReplay costs (should be ~15-20):")
print(f"  mean={np.mean(costs):.1f}")
for i, c in enumerate(costs):
    print(f"  {csvs[i].name}: {c:.1f}")
