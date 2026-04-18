"""Verify grad_opt's differentiable rollout matches official sim's expected-value path.

Runs both on CPU for route 00000 with zero actions and compares per-step predictions.
"""

import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    LataccelTokenizer,
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

CL = CONTEXT_LENGTH
N_CTRL = COST_END_IDX - CONTROL_START_IDX
TEMP = 0.8
BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


# ── Official sim with expected-value (monkey-patched) ─────────
class ExpValModel(TinyPhysicsModel):
    """Official model but returns expected value instead of sample."""

    def predict(self, input_data, temperature=0.8):
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        # Expected value instead of sample
        bins_np = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
        expected = np.sum(probs[0, -1] * bins_np)
        # Still need to return a token for decode
        token = np.argmin(np.abs(bins_np - expected))
        return token


class ZeroCtrl:
    def update(self, t, c, s, future_plan=None):
        return 0.0


print("Running official sim with expected-value predictions...")
model_official = ExpValModel("models/tinyphysics.onnx", debug=False)
sim = TinyPhysicsSimulator(model_official, "data/00000.csv", ZeroCtrl())
sim.rollout()
official_preds = np.array(sim.current_lataccel_history)
official_acts = np.array(sim.action_history)
official_cost = sim.compute_cost()
print(f"  Official expected-value cost: {official_cost['total_cost']:.2f}")

# ── Our differentiable rollout (on CPU, single route) ─────────
print("\nRunning grad_opt differentiable rollout...")
import onnx2torch

torch_model = onnx2torch.convert("models/tinyphysics.onnx")
torch_model.eval()
for p in torch_model.parameters():
    p.requires_grad_(False)

df = pd.read_csv("data/00000.csv")
T = len(df)

roll_la = torch.from_numpy(np.sin(df["roll"].values) * ACC_G).float().unsqueeze(0)
v_ego = torch.from_numpy(df["vEgo"].values).float().unsqueeze(0)
a_ego = torch.from_numpy(df["aEgo"].values).float().unsqueeze(0)
target_la = (
    torch.from_numpy(df["targetLateralAcceleration"].values).float().unsqueeze(0)
)
steer_cmd = torch.from_numpy(-df["steerCommand"].values).float().unsqueeze(0)

# Actions: zeros for control phase (matching ZeroCtrl)
actions = torch.zeros(1, N_CTRL)
warmup_acts = steer_cmd[:, CL:CONTROL_START_IDX]
n_post = T - COST_END_IDX
post_acts = torch.zeros(1, max(n_post, 0))
action_full = torch.cat([steer_cmd[:, :CL], warmup_acts, actions, post_acts], dim=1)

state_full = torch.stack([roll_la[:, :T], v_ego[:, :T], a_ego[:, :T]], dim=-1)

pred_hist = torch.zeros(1, T)
pred_hist[:, :CL] = target_la[:, :CL]
current_la = pred_hist[:, CL - 1].clone()

our_preds = [0.0] * CL  # will fill from CL onwards

# Warmup
for step_idx in range(CL, CONTROL_START_IDX):
    act_start = step_idx - CL + 1
    tok_start = step_idx - CL
    states = torch.cat(
        [
            action_full[:, act_start : act_start + CL].unsqueeze(-1),
            state_full[:, act_start : act_start + CL],
        ],
        dim=-1,
    )
    clamped = pred_hist[:, tok_start : tok_start + CL].clamp(
        LATACCEL_RANGE[0], LATACCEL_RANGE[1]
    )
    tokens = torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()
    logits = torch_model(states, tokens)
    probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)
    expected_la = (probs * BINS.unsqueeze(0)).sum(dim=-1)
    pred = expected_la.clamp(current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA)
    current_la = target_la[:, step_idx]
    pred_hist[:, step_idx] = current_la
    our_preds.append(current_la.item())

# Control phase
prev_pred = current_la.clone()
lat_cost = 0.0
jerk_cost = 0.0
for step_idx in range(CONTROL_START_IDX, min(T, COST_END_IDX + 50)):
    act_start = step_idx - CL + 1
    tok_start = step_idx - CL
    states = torch.cat(
        [
            action_full[:, act_start : act_start + CL].unsqueeze(-1),
            state_full[:, act_start : act_start + CL],
        ],
        dim=-1,
    )
    clamped = pred_hist[:, tok_start : tok_start + CL].clamp(
        LATACCEL_RANGE[0], LATACCEL_RANGE[1]
    )
    tokens = torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()
    logits = torch_model(states, tokens)
    probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)
    expected_la = (probs * BINS.unsqueeze(0)).sum(dim=-1)
    pred = expected_la.clamp(
        current_la.detach() - MAX_ACC_DELTA, current_la.detach() + MAX_ACC_DELTA
    )
    current_la = pred
    pred_hist[:, step_idx] = current_la.detach()
    our_preds.append(current_la.item())

    ci = step_idx - CONTROL_START_IDX
    if ci < N_CTRL:
        tgt = target_la[0, step_idx].item()
        lat_cost += (tgt - current_la.item()) ** 2 * LAT_ACCEL_COST_MULTIPLIER
        jerk = (current_la.item() - prev_pred.item()) / DEL_T
        jerk_cost += jerk**2

    prev_pred = current_la.clone()

our_total = 100 * (lat_cost / N_CTRL + jerk_cost / N_CTRL)
print(f"  Our expected-value cost: {our_total:.2f}")

# Compare per-step predictions
our_arr = np.array(our_preds[: len(official_preds)])
off_arr = official_preds[: len(our_arr)]
diffs = np.abs(our_arr - off_arr)
print(f"\n  Steps compared: {len(our_arr)}")
print(f"  Max pred diff: {diffs.max():.6f} at step {diffs.argmax()}")
print(f"  Mean pred diff: {diffs.mean():.6f}")
print(
    f"  First diff > 0.001: step {np.argmax(diffs > 0.001) if (diffs > 0.001).any() else 'NONE'}"
)

# Print first 5 control step predictions side by side
print(f"\n  Step |  Official  |    Ours    |   Diff")
for i in range(CONTROL_START_IDX, CONTROL_START_IDX + 10):
    if i < len(our_arr) and i < len(off_arr):
        print(f"  {i:4d} | {off_arr[i]:10.6f} | {our_arr[i]:10.6f} | {diffs[i]:.6f}")
