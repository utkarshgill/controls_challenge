# Experiment Roadmap: 75 → <45

**Goal:** Beat <45 cost through surgical, incremental improvements.

**Current:** exp017 baseline = 75.6 cost (PID recovery)

---

## Phase 1: State Representation (BC only)

### exp017: Baseline ✓
- **State:** `[error, error_integral, error_diff]` (3D)
- **Network:** Linear, no bias
- **Result:** 75.6 cost
- **Status:** DONE - Perfect PID recovery

### exp018: + Velocity
- **State:** `[error, ei, ed, v_ego/30]` (4D)
- **Network:** Linear, no bias
- **Hypothesis:** Speed-adaptive gains → small improvement
- **Expected:** 70-75 cost
- **Why:** Different speeds need different control authority

### exp019: Non-Linear
- **State:** `[error, ei, ed, v_ego/30]` (4D)
- **Network:** 2 hidden layers (64 units), Tanh
- **Hypothesis:** Non-linear dynamics exploitation
- **Expected:** 65-70 cost
- **Why:** Simulator is non-linear, adaptive gains matter

### exp020: + Future Preview
- **State:** `[error, ei, ed, v_ego/30, future_lataccel[0:5]/5]` (9D)
- **Network:** 2 hidden layers (64 units), Tanh
- **Hypothesis:** Proactive control is the big win
- **Expected:** 55-65 cost
- **Why:** Anticipate turns, not react to them

### exp021: Richer State
- **State:** `[error, ei, ed, v_ego/30, a_ego/5, roll, future[0:10]/5]` (16D)
- **Network:** 2-3 hidden layers (128 units), Tanh
- **Hypothesis:** More information helps
- **Expected:** 50-60 cost
- **Why:** Acceleration and road banking are signals

---

## Phase 2: Architecture Tuning (BC only)

### exp022: Hyperparameter Search
- **State:** Best from Phase 1
- **Network:** Grid search over:
  - Layers: [2, 3]
  - Units: [64, 128, 256]
  - Activations: [Tanh, ReLU]
  - Dropout: [0, 0.1]
- **Expected:** 48-55 cost
- **Why:** Architecture matters for capacity

---

## Phase 3: PPO Fine-tuning

### exp023: PPO Setup
- **Init:** Best BC model from exp022
- **Reward:** `-total_cost` (match eval metric)
- **Hyperparams:** Conservative (lr=1e-4, clip=0.15, epochs=4)
- **Expected:** 45-50 cost
- **Why:** BC gets close, PPO optimizes actual objective

### exp024: PPO Tuning
- **Init:** exp023 checkpoint
- **Tune:** Learning rate, clip, entropy, batch size
- **Expected:** <45 cost
- **Why:** Final polish to match winner

---

## Principles

1. **One variable at a time:** State OR architecture OR algorithm
2. **Always compare to previous best:** Track improvement
3. **Evaluate properly:** Train/val/test on shuffled splits
4. **Keep it simple:** Don't add complexity without evidence
5. **Log everything:** Costs, weights, hyperparameters

## Success Criteria

- Each experiment should show clear improvement OR teach us something
- If no improvement after 2 tries, hypothesis is wrong
- Document failures as much as successes
- Goal: <45 cost by exp024

---

## Next Action

Start exp018: Add velocity to state, keep linear network.
Expected time: 10 minutes to code + train + eval.

