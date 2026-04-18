"""Verify: straight-through estimator for differentiable sampling.

Forward: exact hard sample (matches official sim bit-for-bit)
Backward: gradients through expected value proxy
"""

import sys, numpy as np, torch, torch.nn.functional as F
from hashlib import md5

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
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
import pandas as pd, onnx2torch

CL = CONTEXT_LENGTH
N_CTRL = COST_END_IDX - CONTROL_START_IDX
TEMP = 0.8
BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


def ste_sample(probs, u):
    """Straight-through differentiable sample.

    Forward: returns bins[searchsorted(CDF, u)] — exact hard sample
    Backward: gradients flow through sum(probs * bins) — expected value
    """
    # Hard path (no grad)
    with torch.no_grad():
        CDF = torch.cumsum(probs, dim=-1)
        token = torch.searchsorted(CDF, u.unsqueeze(-1)).squeeze(-1)
        token = token.clamp(0, VOCAB_SIZE - 1)
        hard_la = BINS[token]

    # Soft path (grad flows)
    soft_la = (probs * BINS.unsqueeze(0)).sum(dim=-1)

    # STE: forward = hard_la, backward = d(soft_la)/d(probs)
    return soft_la + (hard_la - soft_la).detach()


# ═══════════════════════════════════════════════════════════
#  Test 1: Forward matches hard sample exactly
# ═══════════════════════════════════════════════════════════
print("Test 1: STE forward == hard sample")
logits = torch.randn(4, 1024)
probs = F.softmax(logits / TEMP, dim=-1)
u_vals = torch.tensor([0.1, 0.5, 0.9, 0.01])

ste_la = ste_sample(probs, u_vals)

CDF_np = probs.numpy().cumsum(axis=-1)
hard_tokens = [np.searchsorted(CDF_np[i], u_vals[i].item()) for i in range(4)]
hard_tokens = [min(t, 1023) for t in hard_tokens]
hard_la = [BINS[t].item() for t in hard_tokens]

for i in range(4):
    diff = abs(ste_la[i].item() - hard_la[i])
    print(
        f"  u={u_vals[i]:.2f}: ste={ste_la[i].item():.6f} hard={hard_la[i]:.6f} diff={diff:.1e}"
    )

# ═══════════════════════════════════════════════════════════
#  Test 2: Gradients flow
# ═══════════════════════════════════════════════════════════
print("\nTest 2: Gradient flow through STE")
act = torch.nn.Parameter(torch.zeros(1))
logits = act * torch.randn(1, 1024)
probs = F.softmax(logits / TEMP, dim=-1)
u = torch.tensor([0.3])
la = ste_sample(probs, u)
loss = (la - 1.0) ** 2
loss.backward()
print(f"  grad = {act.grad.item():.6f} (nonzero: {act.grad.abs().item() > 1e-10})")

# ═══════════════════════════════════════════════════════════
#  Test 3: Full rollout — STE sim vs official sampled sim
# ═══════════════════════════════════════════════════════════
print("\nTest 3: Full rollout on route 00000")
print("=" * 60)

torch_model = onnx2torch.convert("models/tinyphysics.onnx")
torch_model.eval()
for p in torch_model.parameters():
    p.requires_grad_(False)

df = pd.read_csv("data/00000.csv")
T = len(df)
N = 1
roll_la = torch.from_numpy(np.sin(df["roll"].values) * ACC_G).float().unsqueeze(0)
v_ego = torch.from_numpy(df["vEgo"].values).float().unsqueeze(0)
a_ego = torch.from_numpy(df["aEgo"].values).float().unsqueeze(0)
target_la = (
    torch.from_numpy(df["targetLateralAcceleration"].values).float().unsqueeze(0)
)
steer_cmd = torch.from_numpy(-df["steerCommand"].values).float().unsqueeze(0)

# Pre-generate RNG matching official sim
seed_str = "data/00000.csv"
seed = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
rng = np.random.RandomState(seed)
u_all = torch.from_numpy(rng.rand(T - CL).astype(np.float32))

# Load CEM actions
cem = np.load("experiments/exp110_mpc/checkpoints/actions_5k_v2.npz")
cem_acts = torch.from_numpy(cem["00000.csv"].astype(np.float32)).unsqueeze(0)

for label, actions in [("zeros", torch.zeros(1, N_CTRL)), ("CEM", cem_acts)]:
    warmup = steer_cmd[:, CL:CONTROL_START_IDX]
    n_post = T - COST_END_IDX
    action_full = torch.cat(
        [
            steer_cmd[:, :CL],
            warmup,
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
            u_t = u_all[step_idx - CL : step_idx - CL + 1]
            sampled_la = ste_sample(probs, u_t)

            pred_val = sampled_la.clamp(
                current_la - MAX_ACC_DELTA, current_la + MAX_ACC_DELTA
            )
            # Warmup: override with target
            current_la = target_la[:, step_idx]
            pred_hist[:, step_idx] = current_la

        # Control
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
            u_t = u_all[step_idx - CL : step_idx - CL + 1]
            sampled_la = ste_sample(probs, u_t)

            pred_val = sampled_la.clamp(
                current_la.detach() - MAX_ACC_DELTA, current_la.detach() + MAX_ACC_DELTA
            )
            current_la = pred_val
            # Store HARD sample in pred_hist for correct tokenization
            pred_hist[:, step_idx] = current_la.detach()

            tgt = target_la[0, step_idx].item()
            lat_cost += (tgt - current_la.item()) ** 2 * LAT_ACCEL_COST_MULTIPLIER
            jerk = (current_la.item() - prev_pred.item()) / DEL_T
            jerk_cost += jerk**2
            prev_pred = current_la.clone()

    total = 100 * (lat_cost / N_CTRL + jerk_cost / N_CTRL)
    print(f"  {label:5s}: STE cost = {total:.2f}")

# Official reference
print("\n  Official sampled sim reference:")
m_ref = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)


class CemCtrl:
    def __init__(s, a):
        s.a = a
        s.step = 0

    def update(s, t, c, st, future_plan=None):
        s.step += 1
        ci = s.step - (CONTROL_START_IDX - CONTEXT_LENGTH + 1)
        if 0 <= ci < len(s.a):
            return float(s.a[ci])
        return 0.0


sim_ref = TinyPhysicsSimulator(m_ref, "data/00000.csv", CemCtrl(cem["00000.csv"]))
c_ref = sim_ref.rollout()
print(f"  CEM:   {c_ref['total_cost']:.2f}")


class Z:
    def update(s, t, c, st, future_plan=None):
        return 0.0


sim_ref2 = TinyPhysicsSimulator(m_ref, "data/00000.csv", Z())
c_ref2 = sim_ref2.rollout()
print(f"  zeros: {c_ref2['total_cost']:.2f}")
