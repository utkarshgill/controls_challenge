"""Minimal diagnostic: just call exp055's rollout directly at temp=0.8."""

import os, sys, torch, numpy as np
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics_batched import CSVCache, make_ort_session
from experiments.exp055_batch_of_batch.train import ActorCritic, rollout

DEV = torch.device("cuda")
N_ROUTES = int(os.getenv("N_ROUTES", "10"))

mdl_path = ROOT / "models" / "tinyphysics.onnx"
ort_sess = make_ort_session(mdl_path)
all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
csv_cache = CSVCache([str(f) for f in all_csv])

ac = ActorCritic().to(DEV)
ckpt = torch.load(
    ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
    weights_only=False,
    map_location=DEV,
)
ac.load_state_dict(ckpt["ac"])
ac.eval()
ds = float(ckpt.get("delta_scale", 0.25))

print(f"Running policy (deterministic) on {N_ROUTES} routes...")
costs = rollout(all_csv, ac, mdl_path, ort_sess, csv_cache, deterministic=True, ds=ds)
print(f"Costs: {[f'{c:.1f}' for c in costs]}")
print(f"Mean: {np.mean(costs):.1f}")
