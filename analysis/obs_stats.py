"""Compute per-feature statistics across the dataset to inform OBS_SCALE."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, CONTEXT_LENGTH
import pandas as pd

FUTURE_K = 50
DATA = ROOT / 'data'
N_FILES = 2000  # sample for speed

files = sorted(DATA.glob('*.csv'))[:N_FILES]

# Accumulators
errors, eis, ediffs = [], [], []
v_egos, a_egos, rolls, kappas = [], [], [], []
fk_all, dk_all = [], []
fv_all, fa_all = [], []

for i, f in enumerate(files):
    df = pd.read_csv(f)
    start = CONTROL_START_IDX
    end = min(len(df), 600)

    tgt = df['targetLateralAcceleration'].values[start:end].astype(np.float32)
    cur = tgt * 0  # current_lataccel not in CSV (sim-generated)
    v   = df['vEgo'].values[start:end].astype(np.float32)
    a   = df['aEgo'].values[start:end].astype(np.float32)
    r   = df['roll'].values[start:end].astype(np.float32)

    e = tgt - cur
    ei = np.cumsum(e)
    ed = np.diff(e, prepend=e[0])

    errors.append(e)
    eis.append(ei)
    ediffs.append(ed)
    v_egos.append(v)
    a_egos.append(a)
    rolls.append(r)

    k = np.clip((tgt - r) / np.maximum(v**2, 25.0), -1.0, 1.0)
    kappas.append(k)

    # Future plan κ (approximate: use target_lataccel as future lataccel)
    fut_lat = df['targetLateralAcceleration'].values.astype(np.float32)
    fut_roll = df['roll'].values.astype(np.float32)
    fut_v = df['vEgo'].values.astype(np.float32)
    fut_a = df['aEgo'].values.astype(np.float32)

    for t in range(start, min(end, len(df) - FUTURE_K)):
        fl = fut_lat[t+1:t+1+FUTURE_K]
        fr = fut_roll[t+1:t+1+FUTURE_K]
        fvs = fut_v[t+1:t+1+FUTURE_K]
        fas = fut_a[t+1:t+1+FUTURE_K]
        if len(fl) == FUTURE_K:
            fk = np.clip((fl - fr) / np.maximum(fvs**2, 25.0), -1.0, 1.0)
            fk_all.append(fk)
            dk_all.append(np.diff(fk))
            fv_all.append(fvs)
            fa_all.append(fas)

    if (i+1) % 500 == 0:
        print(f"  processed {i+1}/{N_FILES}")

print(f"\nAnalyzed {N_FILES} files\n")
print(f"{'Feature':<20} {'mean':>10} {'std':>10} {'min':>10} {'p5':>10} {'p95':>10} {'max':>10}  suggested 1/scale")
print("-" * 110)

def stats(name, arr):
    a = np.concatenate(arr) if isinstance(arr[0], np.ndarray) else np.array(arr)
    a = a.flatten()
    p5, p95 = np.percentile(a, [5, 95])
    rng = max(abs(p5), abs(p95))
    suggested = f"1/{rng:.2f}" if rng > 0.01 else "×100+"
    print(f"{name:<20} {a.mean():10.4f} {a.std():10.4f} {a.min():10.4f} {p5:10.4f} {p95:10.4f} {a.max():10.4f}  {suggested}")

stats("error", errors)
stats("error_integral", eis)
stats("error_diff", ediffs)
stats("v_ego", v_egos)
stats("a_ego", a_egos)
stats("roll_lataccel", rolls)
stats("kappa_now", kappas)
stats("future_kappa", fk_all)
stats("delta_kappa", dk_all)
stats("future_v_ego", fv_all)
stats("future_a_ego", fa_all)
