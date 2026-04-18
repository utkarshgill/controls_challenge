"""Test: run MPC on 1 route, save actions, load them back, replay, compare."""

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

# Run 1: simple controller, store actions
data1, rng1 = csv_cache.slice(csvs)
sim1 = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data1, cached_rng=rng1)
N = sim1.N
stored = torch.zeros((N, N_CTRL), dtype=torch.float32, device="cuda")


def ctrl1(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    target = sim_ref.data_gpu["target_lataccel"][:, step_idx].float()
    current = sim_ref.current_lataccel.float()
    action = ((target - current) * 0.3).double().clamp(-2, 2)
    stored[:, ci] = action.float()  # store as float32, same as MPC
    return action


cost1 = sim1.rollout(ctrl1)["total_cost"]
print(f"Run 1 cost: {cost1[0]:.4f}")

# Save to npz (exactly as MPC does)
actions_np = stored.cpu().numpy()
np.savez("/tmp/test_actions.npz", **{csvs[0].name: actions_np[0]})

# Load back
loaded = np.load("/tmp/test_actions.npz")
loaded_actions = loaded[csvs[0].name]
print(f"Saved dtype: {actions_np.dtype}, Loaded dtype: {loaded_actions.dtype}")
print(f"Save/load match: {np.array_equal(actions_np[0], loaded_actions)}")
print(f"Max diff: {np.max(np.abs(actions_np[0] - loaded_actions))}")

# Replay loaded actions
data2, rng2 = csv_cache.slice(csvs)
sim2 = BatchedSimulator(mdl, ort_session=ort_sess, cached_data=data2, cached_rng=rng2)
replay_stored = torch.from_numpy(loaded_actions).float().unsqueeze(0).to("cuda")


def ctrl2(step_idx, sim_ref):
    if step_idx < CONTROL_START_IDX:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    ci = step_idx - CONTROL_START_IDX
    if ci >= N_CTRL:
        return torch.zeros(N, dtype=torch.float64, device="cuda")
    return replay_stored[:, ci].double()


cost2 = sim2.rollout(ctrl2)["total_cost"]
print(f"Run 2 cost (replay from npz): {cost2[0]:.4f}")
print(f"Match: {np.allclose(cost1, cost2)}")
print(f"Diff: {abs(cost1[0] - cost2[0]):.6f}")
