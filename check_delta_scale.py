#!/usr/bin/env python
"""Check steer delta distribution to pick DELTA_SCALE."""
import numpy as np, pandas as pd, glob
from multiprocessing import Pool

def get_deltas(csv_path):
    df = pd.read_csv(csv_path)
    steer = -df['steerCommand'].values[:100]  # only valid range
    deltas = np.diff(steer[20:])  # steps 20→99 = 79 deltas
    return deltas.astype(np.float32)

files = sorted(glob.glob('data/*.csv'))[:100]
print(f"{len(files)} files")

results = [get_deltas(f) for f in files]
all_deltas = np.concatenate(results)
print(f"{len(all_deltas)} total deltas\n")

abs_d = np.abs(all_deltas)
for pct in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
    v = np.percentile(abs_d, pct)
    print(f"  p{pct:5.1f}  |delta| = {v:.6f}")

print(f"\n  mean |delta| = {abs_d.mean():.6f}")
print(f"  std  |delta| = {abs_d.std():.6f}")
print(f"  max  |delta| = {abs_d.max():.6f}")

print("\nClip rates at various DELTA_SCALE:")
for ds in [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]:
    clipped = (abs_d > ds).mean() * 100
    print(f"  DELTA_SCALE={ds:.3f}  clipped={clipped:5.2f}%")
