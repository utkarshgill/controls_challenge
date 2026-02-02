# Experiment 047: Neural Atom - Growing a Controller

## Goal
Start from a single neuron and incrementally grow a neural controller using supervised learning and RL.

## Progress

### Step 1-3: Behavioral Cloning from PID
- **Single linear neuron** (3 weights: error, error_integral, error_diff)
- Trained on 2000 routes, 1.16M samples
- **Result:** R²=0.937, **Cost=120.8**
- Added tanh activation → Cost=120.8 (no improvement)
- Added roll_lataccel feature → Cost=120.8 (no improvement)

### Step 4: BC from Best Hand-Tuned FF (exp046_v3)
- **50 weights** on future errors: (future_lataccel[i] - target) for i=0..49
- Trained to mimic exp046_v3's FF actions
- **Result:** R²=0.996, **Cost=82.89** (exact match to exp046_v3)
- **Key finding:** Only w[0]=0.267 matters, w[1..49]≈0

### Step 5: Grid Search for Optimal Decay
Hypothesis: Exponential decay w[i] = 0.267 * exp(-i/tau) should beat single-step FF

**Coarse search (tau ∈ [1, 2, 5, 10, 20, 50, 100]):**
- tau=1: **81.59** ✓
- tau=2: 93.69
- tau≥5: Explodes

**Fine search (tau ∈ [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]):**
- **tau=0.9: 81.46** ⭐ OPTIMAL
- tau=1.0: 81.59
- tau=0.7: 81.68
- tau=0.5: 82.15
- tau<0.5 or tau>1.2: Worse

### Optimal Pattern (tau=0.9)
```
w[0] = 0.267  (t=0.0s) - immediate future
w[1] = 0.088  (t=0.1s) - significant
w[2] = 0.029  (t=0.2s) - moderate
w[3] = 0.010  (t=0.3s) - small
w[5] = 0.001  (t=0.5s) - negligible
w[10+] ≈ 0    - ignored
```

**Effective horizon: ~0.3 seconds (3 timesteps)**

## Final Results

| Controller | Method | Cost | vs Baseline |
|------------|--------|------|-------------|
| PID | Fixed | 84.85 | baseline |
| exp046_v3 | Hand-tuned FF | 82.89 | +1.96 |
| **exp047_optimal** | **Learned decay (tau=0.9)** | **81.46** | **+3.39** ⭐ |

## Key Insights

1. **Single neuron is sufficient** for feedforward control
2. **Exponential decay** is the right functional form
3. **Only first 3 timesteps matter** (0.3s lookahead)
4. **Too much future = instability** (jerk explodes)
5. **Optimal tau≈0.9** balances curvature anticipation vs. stability

## Step 6: PPO Training (In Progress)
Goal: Prove PPO can discover tau=0.9 pattern from scratch

**Status:** Environment issues - costs reporting incorrectly in vectorized setting
**Expected:** Should converge to similar pattern as hand-crafted decay

## Next Steps
1. Debug PPO environment cost tracking
2. Verify PPO discovers similar decay pattern
3. Try other functional forms (polynomial, step functions)
4. Push toward competition winner's 35 cost
