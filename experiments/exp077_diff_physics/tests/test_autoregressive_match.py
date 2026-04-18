"""Test: does the 1e-5 per-step diff compound over 400 autoregressive steps?
Run both ORT and onnx2torch on the same route with same actions, compare trajectories."""

import os, sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import onnxruntime as ort
import onnx2torch
from tinyphysics import (
    LATACCEL_RANGE,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    MAX_ACC_DELTA,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    ACC_G,
)
from hashlib import md5
import pandas as pd
from pathlib import Path

CL = CONTEXT_LENGTH
BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
BINS_T = torch.from_numpy(BINS)

# Load models
ort_sess = ort.InferenceSession(
    "models/tinyphysics.onnx", providers=["CPUExecutionProvider"]
)
torch_model = onnx2torch.convert("models/tinyphysics.onnx")
torch_model.eval()

# Load route data
df = pd.read_csv("data/00000.csv")
roll_la = (np.sin(df["roll"].values) * ACC_G).astype(np.float32)
v_ego = df["vEgo"].values.astype(np.float32)
a_ego = df["aEgo"].values.astype(np.float32)
target = df["targetLateralAcceleration"].values.astype(np.float32)
steer = (-df["steerCommand"].values).astype(np.float32)

# Seed
seed = int(md5("data/00000.csv".encode()).hexdigest(), 16) % 10**4
rng = np.random.RandomState(seed)
T = len(df)
rng_vals = rng.rand(T - CL)


def softmax(x, temp=0.8):
    x = x / temp
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# Run both models autoregressively
def run_autoregressive(model_fn, label):
    state_hist = np.zeros((T, 3), dtype=np.float32)
    action_hist = np.zeros(T, dtype=np.float32)
    lataccel_hist = np.zeros(T, dtype=np.float32)

    # Initialize
    for i in range(CL):
        state_hist[i] = [roll_la[i], v_ego[i], a_ego[i]]
        action_hist[i] = steer[i]
        lataccel_hist[i] = target[i]

    current_la = target[CL - 1]

    for step in range(CL, T):
        state_hist[step] = [roll_la[step], v_ego[step], a_ego[step]]
        action_hist[step] = 0.0  # zero action

        # Build context
        s = state_hist[step - CL + 1 : step + 1]  # (CL, 3)
        a = action_hist[step - CL + 1 : step + 1]  # (CL,)
        p = lataccel_hist[step - CL : step]  # (CL,)

        states_in = np.column_stack([a, s])[np.newaxis].astype(np.float32)  # (1, CL, 4)
        tokens_in = np.digitize(np.clip(p, -5, 5), BINS, right=True)[np.newaxis].astype(
            np.int64
        )

        # Run model
        logits = model_fn(states_in, tokens_in)  # (1, CL, 1024)
        probs = softmax(logits[0, -1])

        # Sample
        u = rng_vals[step - CL]
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]
        token = np.searchsorted(cdf, u)
        token = min(token, VOCAB_SIZE - 1)

        pred = BINS[token]
        pred = np.clip(pred, current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)

        if step >= CONTROL_START_IDX:
            current_la = pred
        else:
            current_la = target[step]

        lataccel_hist[step] = current_la

    # Compute cost
    ctrl_la = lataccel_hist[CONTROL_START_IDX:COST_END_IDX]
    ctrl_tgt = target[CONTROL_START_IDX:COST_END_IDX]
    lat_cost = np.mean((ctrl_tgt - ctrl_la) ** 2) * 100
    jerk = np.diff(ctrl_la) / DEL_T
    jerk_cost = np.mean(jerk**2) * 100
    total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

    print(f"{label}: total_cost={total:.4f}")
    return lataccel_hist


# ORT model function
def ort_fn(states, tokens):
    return ort_sess.run(None, {"states": states, "tokens": tokens})[0]


# Torch model function
def torch_fn(states, tokens):
    with torch.no_grad():
        return torch_model(torch.from_numpy(states), torch.from_numpy(tokens)).numpy()


la_ort = run_autoregressive(ort_fn, "ORT CPU")
la_torch = run_autoregressive(torch_fn, "onnx2torch")

# Compare trajectories
ctrl_ort = la_ort[CONTROL_START_IDX:COST_END_IDX]
ctrl_torch = la_torch[CONTROL_START_IDX:COST_END_IDX]
mae = np.mean(np.abs(ctrl_ort - ctrl_torch))
maxdiff = np.max(np.abs(ctrl_ort - ctrl_torch))

print(f"\nTrajectory comparison (control window):")
print(f"  MAE: {mae:.8f}")
print(f"  Max diff: {maxdiff:.8f}")
print(f"  Exact match: {np.array_equal(ctrl_ort, ctrl_torch)}")

# Count token mismatches
mismatches = np.sum(la_ort != la_torch)
print(f"  Token mismatches (full trajectory): {mismatches}/{len(la_ort)}")

# Where does first mismatch occur?
for i in range(len(la_ort)):
    if la_ort[i] != la_torch[i]:
        print(
            f"  First mismatch at step {i}: ORT={la_ort[i]:.8f} torch={la_torch[i]:.8f}"
        )
        break
else:
    print("  NO MISMATCHES — trajectories are identical!")
