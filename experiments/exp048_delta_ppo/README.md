# exp048 — Delta-Action PPO

## Hypothesis

PPO fails on the controls challenge for three specific, fixable reasons:

| # | Problem | Why it kills PPO | Fix |
|---|---------|-----------------|-----|
| 1 | **Absolute action space** | Exploration noise cascades through the 20-step autoregressive simulator — one noisy action corrupts context for 20 future predictions | **Delta-action**: policy outputs Δsteer, `action = prev + tanh(Δ)·0.3` — bounds jerk at 3.0, keeps exploration smooth |
| 2 | **No history in state** | POMDP — critic can't model value without seeing the dynamic state the simulator conditions on | **History**: last 10 actions + last 10 lataccels in obs → near-Markov |
| 3 | **PID + residual** | PPO gradient correlates residual with error (easiest local improvement), fighting PID instead of complementing it | **End-to-end**: no PID crutch, BC warm-start from PID demonstrations |

## Architecture

**State** (50 dims):
```
[0]     target_lataccel
[1]     current_lataccel
[2]     v_ego
[3]     a_ego
[4]     roll_lataccel
[5:15]  last 10 actions       (captures what I recently did)
[15:25] last 10 lataccels     (captures dynamics state)
[25:50] future curvatures×25  (2.5s preview for anticipation)
```

**Network**: MLP 50→128→128→128→128→1 (actor, 4 layers), 50→128→128→128→1 (critic, 3 layers).

**Action**: `delta = tanh(network_output) × MAX_DELTA`, `action = prev_action + delta`, clipped to [-2, 2].

## Why MAX_DELTA = 0.3 is the Right Value

- **Bounds jerk** at 3.0 m/s³ per step (vs unbounded in absolute action space)
- **Forces anticipation**: the policy *can't* react instantly — with max Δ=0.3/step, reaching steer=1.5 from 0 takes 5 steps (0.5s). To track well, it MUST use the 2.5s preview to start turning early.
- **Safe exploration**: with σ≈0.135, typical exploration delta ≈ 0.04/step → jerk ≈ 0.4 → jerk_cost contribution ≈ 1.6 (vs PID's 35)

## Training

1. **BC warm-start** (Phase 1): Collect 200 PID trajectories, extract (obs, delta) pairs, supervised MSE on arctanh-transformed targets. Gets policy to ~85 cost.
2. **PPO refinement** (Phase 2): 500 trajectories/epoch, conservative hyperparameters (π_lr=3e-5, clip=0.15, K=10), large effective batch (200K transitions/epoch).

## Run

```bash
cd controls_challenge
.venv/bin/python experiments/exp048_delta_ppo/train.py

# Override defaults:
CSVS=1000 EPOCHS=300 WORKERS=16 .venv/bin/python experiments/exp048_delta_ppo/train.py
```

## Key Differences from exp043

| | exp043 | exp048 |
|---|--------|--------|
| Action | Absolute (tanh → steer) | Delta (tanh → Δsteer, integrated) |
| State | 57d (no history) | 50d (10 actions + 10 lataccels history) |
| Feedback | 10% PID + 90% network | Pure network (end-to-end) |
| π lr | 3e-4 | 3e-5 (10× smaller) |
| K epochs | 20 | 10 |
| Clip | 0.2 | 0.15 |
| CSVs/epoch | 250 | 500 |
| Future | 50 curvatures | 25 curvatures (2.5s, not 5s) |

## Expected Outcome

- BC baseline: ~85 (matching PID)
- After PPO: <60 within 50 epochs, <50 within 150 epochs
- Mechanism: policy learns anticipatory steering using future curvatures, with smooth exploration enabled by delta-action constraint
