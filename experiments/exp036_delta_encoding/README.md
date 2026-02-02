# Experiment 036: Delta-Encoded Preview (Phase-Aware PPO)

## The Core Hypothesis

**Exp034 result**: Linear preview on 3 features → 102 cost (same as PID)

**Diagnosis**: PPO couldn't use preview because observations were **absolute**, but the plant is **autoregressive**.

**The one change**: Encode preview as deltas relative to **current state (t)**, not future baseline.

## What Changed from Exp034

### Exp034 (Future-Relative Encoding)
```python
baseline = future.lataccel[0]  # t+1
Δ[i] = future.lataccel[i] - baseline
```
This hides phase: "How is the future changing relative to itself?"

### Exp036 (Present-Relative Encoding)
```python
baseline = target_lataccel(t)  # current t
Δ[i] = future.lataccel[i] - baseline
```
This reveals phase: "Am I ahead or behind the curve?"

## Why Phase Matters

The learned plant is:
- Autoregressive (depends on last 20 steps)
- Rate-limited (MAX_ACC_DELTA = 0.5 m/s²)
- Stochastic (temperature=0.8 sampling)

PID implicitly operates on error deltas (proportional, derivative).

PPO with absolute preview cannot infer:
- "Am I entering or exiting a curve?"
- "Is my phase aligned with target trajectory?"
- "What is the rate of change?"

Delta-encoding makes these explicit.

## State Representation

**200D delta-encoded preview** (4 signals × 50 steps):

```python
Δlataccel[i] = future.lataccel[i] - target_lataccel(t)   # i ∈ [0..49]
Δroll[i]     = future.roll[i]     - roll_lataccel(t)
Δv[i]        = future.v[i]        - v_ego(t)
Δa[i]        = future.a[i]        - a_ego(t)
```

All deltas normalized by appropriate scales.

## Architecture

```
State: 200D (delta-encoded preview, NO error terms)
Policy: Shallow MLP (200 → 64 → 64 → 1)
Low-pass: α = 0.05 (aggressive)
Residual scale: ε = 0.1
```

Following beautiful_lander.py: simple shallow MLP, no fancy architecture.

## Expected Outcomes (Diagnostic)

| Cost Range | Interpretation |
|------------|----------------|
| **< 80** | ✅ PPO needed phase info, not more power |
| **~80-90** | ✅ Delta-encoding helps, but preview is weak |
| **~101** | ❌ Phase doesn't matter, preview fundamentally limited |
| **> 120** | 🔥 Regression (something broke) |

## Why This Test is Clean

1. **Minimal change**: Only the state encoding changed
2. **Same architecture**: Shallow MLP (like beautiful_lander)
3. **Same hyperparameters**: All PPO settings unchanged
4. **Same evaluation**: Using fixed controller interface
5. **No new information**: Only reshaping what was already given

## The Deep Point

LunarLander worked because:
- State is Markov
- Immediate feedback
- Position-based control

Controls Challenge is different:
- Plant is autoregressive with hidden state
- Delayed effects (action at t affects cost at t+5 to t+20)
- Phase-based control

Absolute observations hide phase.
Delta observations reveal phase.

## Comparison to Winners

Winners achieving <45 likely did:
- Delta-encoding (this experiment)
- OR: BC from better teacher (MPC/optimal control)
- OR: Offline trajectory optimization

If this experiment hits <80, we're on the right path.
If it stays ~101, preview learning from PID trajectories is fundamentally limited.

## Training

```bash
cd experiments/exp036_delta_encoding
python train_ppo.py
```

## Evaluation (After Training)

Create controller and test:

```bash
# Copy checkpoint to controller
# Then evaluate:
python ../../tinyphysics.py --model_path ../../models/tinyphysics.onnx \
    --data_path ../../data --num_segs 100 --controller exp036_delta
```

---

**This is the decisive phase test.**

If PPO can't beat PID with delta-encoded phase information, then:
- Preview alone is insufficient
- Need better teacher than PID
- Or winners used hand-crafted feedforward

Either way, we get a clean answer.
