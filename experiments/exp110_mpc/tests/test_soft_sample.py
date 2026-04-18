"""Verify: differentiable sampling matches official sim exactly in forward pass,
and gradients flow properly in backward pass.

The key: replace searchsorted(CDF, u) with a sigmoid-smoothed version:
  weight_k = sigmoid(tau * (CDF_k - u)) - sigmoid(tau * (CDF_{k-1} - u))
  lataccel = sum(weight_k * bins)

For large tau, forward matches the hard sample. Backward gives gradients
through CDF → cumsum(probs) → softmax(logits/T) → logits → model → actions.
"""

import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from hashlib import md5

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
BINS_NP = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
BINS = torch.from_numpy(BINS_NP).float()
TAU = 1000.0  # sharpness of soft sampling


def soft_sample(probs, u, tau=TAU):
    """Differentiable approximation to searchsorted(CDF, u).

    probs: (N, 1024) — probability distribution
    u: (N,) — pre-generated uniform random value in (0,1)

    Returns: lataccel (N,) — the sampled lataccel value, differentiable through probs
    """
    CDF = torch.cumsum(probs, dim=-1)  # (N, 1024)
    # CDF_shifted = [0, CDF_0, CDF_1, ..., CDF_1022]
    CDF_prev = F.pad(CDF[:, :-1], (1, 0), value=0.0)  # (N, 1024)

    u_expanded = u.unsqueeze(-1)  # (N, 1)

    # Soft indicator: weight_k ≈ 1 if CDF_{k-1} < u <= CDF_k
    weights = torch.sigmoid(tau * (CDF - u_expanded)) - torch.sigmoid(
        tau * (CDF_prev - u_expanded)
    )

    # Weighted sum with bin values
    lataccel = (weights * BINS.unsqueeze(0)).sum(dim=-1)  # (N,)
    return lataccel


# ═══════════════════════════════════════════════════════════
#  Test 1: Does soft_sample match hard sample?
# ═══════════════════════════════════════════════════════════
print("Test 1: Forward-pass match between hard and soft sampling")
print("=" * 60)

np.random.seed(0)
# Create some test probabilities
logits = torch.randn(4, 1024)
probs = F.softmax(logits / TEMP, dim=-1)
u_vals = torch.tensor([0.1, 0.5, 0.9, 0.01])

# Hard sample
CDF_np = probs.numpy().cumsum(axis=-1)
hard_tokens = [np.searchsorted(CDF_np[i], u_vals[i].item()) for i in range(4)]
hard_tokens = [min(t, 1023) for t in hard_tokens]
hard_lataccel = [BINS_NP[t] for t in hard_tokens]

# Soft sample
soft_lataccel = soft_sample(probs, u_vals, tau=TAU)

for i in range(4):
    print(
        f"  u={u_vals[i]:.2f}: hard_token={hard_tokens[i]:4d} "
        f"hard_la={hard_lataccel[i]:.6f} soft_la={soft_lataccel[i].item():.6f} "
        f"diff={abs(hard_lataccel[i] - soft_lataccel[i].item()):.8f}"
    )


# ═══════════════════════════════════════════════════════════
#  Test 2: Do gradients flow?
# ═══════════════════════════════════════════════════════════
print("\nTest 2: Gradient flow")
print("=" * 60)

actions_param = torch.nn.Parameter(torch.zeros(1))
# Fake: action → logits → probs → soft_sample → cost
logits = actions_param * torch.randn(1, 1024)  # action affects logits
probs = F.softmax(logits / TEMP, dim=-1)
u = torch.tensor([0.5])
la = soft_sample(probs, u)
cost = (la - 0.5) ** 2  # target lataccel = 0.5
cost.backward()
print(f"  action grad: {actions_param.grad.item():.6f} (should be nonzero)")
print(f"  grad is nonzero: {actions_param.grad.abs().item() > 1e-10}")


# ═══════════════════════════════════════════════════════════
#  Test 3: Full rollout — soft_sample sim vs official sampled sim
# ═══════════════════════════════════════════════════════════
print("\nTest 3: Full rollout comparison on route 00000")
print("=" * 60)

import onnx2torch

torch_model = onnx2torch.convert("models/tinyphysics.onnx")
torch_model.eval()
for p in torch_model.parameters():
    p.requires_grad_(False)

# Load data
df = pd.read_csv("data/00000.csv")
T = len(df)
roll_la = torch.from_numpy(np.sin(df["roll"].values) * ACC_G).float().unsqueeze(0)
v_ego = torch.from_numpy(df["vEgo"].values).float().unsqueeze(0)
a_ego = torch.from_numpy(df["aEgo"].values).float().unsqueeze(0)
target_la = (
    torch.from_numpy(df["targetLateralAcceleration"].values).float().unsqueeze(0)
)
steer_cmd = torch.from_numpy(-df["steerCommand"].values).float().unsqueeze(0)

# Pre-generate RNG (matching official sim)
seed_str = "data/00000.csv"
seed = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
rng = np.random.RandomState(seed)
n_steps = T - CL
u_all = torch.from_numpy(rng.rand(n_steps).astype(np.float32))  # (n_steps,)

# CEM actions
cem = np.load("experiments/exp110_mpc/checkpoints/actions_5k_v2.npz")
cem_acts = torch.from_numpy(cem["00000.csv"].astype(np.float32)).unsqueeze(
    0
)  # (1, 400)

for label, actions in [("zeros", torch.zeros(1, N_CTRL)), ("CEM", cem_acts)]:
    warmup_acts = steer_cmd[:, CL:CONTROL_START_IDX]
    n_post = T - COST_END_IDX
    action_full = torch.cat(
        [
            steer_cmd[:, :CL],
            warmup_acts,
            actions.clamp(STEER_RANGE[0], STEER_RANGE[1]),
            torch.zeros(1, max(n_post, 0)),
        ],
        dim=1,
    )

    state_full = torch.stack([roll_la[:, :T], v_ego[:, :T], a_ego[:, :T]], dim=-1)

    pred_hist = torch.zeros(1, T)
    pred_hist[:, :CL] = target_la[:, :CL]
    current_la = pred_hist[:, CL - 1].clone()

    with torch.no_grad():
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
            tokens = (
                torch.bucketize(clamped, BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .long()
            )
            logits = torch_model(states, tokens)
            probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)

            rng_idx = step_idx - CL
            u_t = u_all[rng_idx : rng_idx + 1]
            sampled_la = soft_sample(probs, u_t, tau=TAU)

            pred_val = sampled_la.clamp(
                current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA
            )
            # During warmup, use target (like official sim)
            current_la = target_la[:, step_idx]
            pred_hist[:, step_idx] = current_la

        # Control phase
        prev_pred = current_la.clone()
        lat_cost = 0.0
        jerk_cost = 0.0

        for step_idx in range(CONTROL_START_IDX, min(T, COST_END_IDX)):
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
            tokens = (
                torch.bucketize(clamped, BINS, right=False)
                .clamp(0, VOCAB_SIZE - 1)
                .long()
            )
            logits = torch_model(states, tokens)
            probs = F.softmax(logits[:, -1, :] / TEMP, dim=-1)

            rng_idx = step_idx - CL
            u_t = u_all[rng_idx : rng_idx + 1]
            sampled_la = soft_sample(probs, u_t, tau=TAU)

            pred_val = sampled_la.clamp(
                current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA
            )
            current_la = pred_val
            pred_hist[:, step_idx] = current_la.detach()

            ci = step_idx - CONTROL_START_IDX
            tgt = target_la[0, step_idx].item()
            lat_cost += (tgt - current_la.item()) ** 2 * LAT_ACCEL_COST_MULTIPLIER
            jerk = (current_la.item() - prev_pred.item()) / DEL_T
            jerk_cost += jerk**2
            prev_pred = current_la.clone()

    total = 100 * (lat_cost / N_CTRL + jerk_cost / N_CTRL)
    print(f"  {label:5s}: cost = {total:.2f}")

# Official sampled sim for reference
print("\n  (Official sampled sim reference:)")
m_ref = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)


class CemCtrl:
    def __init__(self, acts):
        self.acts = acts
        self.step = 0

    def update(self, t, c, s, future_plan=None):
        self.step += 1
        ci = self.step - (CONTROL_START_IDX - CONTEXT_LENGTH + 1)
        if 0 <= ci < len(self.acts):
            return float(self.acts[ci])
        return 0.0


sim_ref = TinyPhysicsSimulator(m_ref, "data/00000.csv", CemCtrl(cem["00000.csv"]))
c_ref = sim_ref.rollout()
print(f"  CEM on official sim: {c_ref['total_cost']:.2f}")


class Z2:
    def update(self, t, c, s, future_plan=None):
        return 0.0


sim_ref2 = TinyPhysicsSimulator(m_ref, "data/00000.csv", Z2())
c_ref2 = sim_ref2.rollout()
print(f"  Zero on official sim: {c_ref2['total_cost']:.2f}")
