"""Minimal smoke test for differentiable rollout."""

import numpy as np, os, sys, torch, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
    ACC_G,
)
import pandas as pd
import onnx2torch

DEV = torch.device("cuda")
N_CTRL = COST_END_IDX - CONTROL_START_IDX
CL = CONTEXT_LENGTH
TEMP = 0.8
BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=DEV)

# Load model
print("Loading onnx2torch model...")
model = onnx2torch.convert(str(ROOT / "models" / "tinyphysics.onnx"))
model.to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
print("Done.")

# Load one route
df = pd.read_csv(str(ROOT / "data" / "00000.csv"))
T = len(df)
N = 1

roll_la = (
    torch.from_numpy(np.sin(df["roll"].values) * ACC_G).float().unsqueeze(0).to(DEV)
)
v_ego = torch.from_numpy(df["vEgo"].values).float().unsqueeze(0).to(DEV)
a_ego = torch.from_numpy(df["aEgo"].values).float().unsqueeze(0).to(DEV)
target_la = (
    torch.from_numpy(df["targetLateralAcceleration"].values)
    .float()
    .unsqueeze(0)
    .to(DEV)
)
steer_cmd = torch.from_numpy(-df["steerCommand"].values).float().unsqueeze(0).to(DEV)

# Build full action trajectory: CSV warmup + zeros for control
actions = torch.zeros((1, N_CTRL), device=DEV)
action_full = torch.cat(
    [
        steer_cmd[:, :CONTROL_START_IDX],
        actions,
        torch.zeros((1, T - COST_END_IDX), device=DEV),
    ],
    dim=1,
)

state_full = torch.stack([roll_la[:, :T], v_ego[:, :T], a_ego[:, :T]], dim=-1)

# Init pred history
pred_hist = torch.zeros((1, T), device=DEV)
pred_hist[:, :CL] = target_la[:, :CL]
current_la = pred_hist[:, CL - 1].clone()

# Run warmup (CL to CONTROL_START_IDX)
print(f"Running warmup steps {CL} to {CONTROL_START_IDX}...")
for step in range(CL, CONTROL_START_IDX):
    h = step + 1
    start = h - CL
    states = torch.cat(
        [action_full[:, start:h].unsqueeze(-1), state_full[:, start:h]], dim=-1
    )
    clamped = pred_hist[:, start:h].clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
    tokens = torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()

    with torch.no_grad():
        logits = model(states, tokens)
    probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)
    expected = (probs * BINS.unsqueeze(0)).sum(dim=-1)
    pred = expected.clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)
    current_la = target_la[:, step]
    pred_hist[:, step] = current_la

print(f"After warmup: current_la = {current_la.item():.4f}")

# Run first 10 control steps
print(f"\nFirst 10 control steps:")
prev = current_la.clone()
for step in range(CONTROL_START_IDX, CONTROL_START_IDX + 10):
    h = step + 1
    start = h - CL
    states = torch.cat(
        [action_full[:, start:h].unsqueeze(-1), state_full[:, start:h]], dim=-1
    )
    clamped = pred_hist[:, start:h].clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
    tokens = torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()

    with torch.no_grad():
        logits = model(states, tokens)
    probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)
    expected = (probs * BINS.unsqueeze(0)).sum(dim=-1)
    pred = expected.clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)
    current_la = pred
    pred_hist[:, step] = current_la.detach()

    tgt = target_la[0, step].item()
    err = (tgt - current_la.item()) ** 2
    jerk = ((current_la.item() - prev.item()) / DEL_T) ** 2
    print(
        f"  step {step}: pred={current_la.item():.4f} target={tgt:.4f} err²={err:.4f} jerk²={jerk:.2f}"
    )
    prev = current_la.clone()

# Also compare: what does the REAL sampled sim produce with zero actions?
print("\n--- Compare with official sim (zero actions) ---")
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


class ZeroCtrl:
    def update(self, t, c, s, future_plan=None):
        return 0.0


m = TinyPhysicsModel(str(ROOT / "models" / "tinyphysics.onnx"), debug=False)
sim = TinyPhysicsSimulator(m, str(ROOT / "data" / "00000.csv"), ZeroCtrl())
cost = sim.rollout()
print(f"Zero-action cost on official sim: {cost['total_cost']:.2f}")
