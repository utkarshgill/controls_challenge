# MPC Development: Lessons Learned

## Journey Summary

### Attempt 1: Pure Feedforward
- **Cost:** 576
- **Problem:** No integral term → can't track
- **Lesson:** Feedback control needs memory (integral)

### Attempt 2: PID Baseline  
- **Cost:** 103
- **Status:** ✓ Works!
- **Lesson:** This is our target to beat

### Attempt 3: 1-Step Lookahead
- **Cost:** 10,073 (!!)
- **Problem:** No temporal consistency → massive jerk (2341)
- **Lesson:** Single-step optimization without continuity = chaos

### Attempt 4: H=10 Random Shooting
- **Cost:** 1,713
- **Problem:** Random sampling without CEM → jerky, unstable
- **Lesson:** Need iterative refinement (CEM), not just sampling

### Attempt 5: CEM with High Smoothness Weight
- **Cost:** 317 (mpc_simple.py)
- **Problem:** Smoothness weight too high (2500) → can't track
- **Lesson:** Over-penalizing action changes hurts tracking

### Attempt 6: CEM with Moderate Smoothness
- **Cost:** 292 (mpc_final.py v1, smoothness=800)
- **Problem:** Still misaligned! MPC optimizes for `lat + jerk + smooth`, but evaluated on `lat + jerk` only
- **Lesson:** **NEVER add terms to internal cost that aren't in the evaluation!**

### Attempt 7: ALIGNED Cost + Small Std (CURRENT)
- **Params:** smoothness=0, std=0.08, samples=150, iters=4
- **Strategy:** Optimize for EXACT evaluation cost, get smoothness implicitly via small std + warm-start
- **Status:** Running...

---

## Key Insights

### 1. Cost Alignment is CRITICAL
The MPC's internal optimization cost MUST match the evaluation cost exactly. Adding extra terms (like smoothness) creates misalignment and hurts performance.

**Wrong:**
```python
cost = lat_cost + jerk_cost + smoothness_cost  # MPC optimizes for this
# But evaluated on: lat_cost + jerk_cost only
```

**Right:**
```python
cost = lat_cost + jerk_cost  # EXACT match!
# Smoothness comes from small std, warm-start, not cost term
```

### 2. Temporal Consistency via Process, Not Cost
Smooth actions should come from:
- **Small exploration std** (e.g. 0.08 not 0.15)
- **Good warm-start** (shift previous solution)
- **Sufficient CEM iterations** (4+ to converge)

NOT from adding smoothness penalty to cost!

### 3. Horizon Matters
- H=1: Catastrophic (no foresight)
- H=10: Poor (myopic)
- **H=50: Good** (full future visibility)

### 4. Random Shooting vs CEM
- Random shooting: Terrible (1713 cost)
- CEM (iterative refinement): Much better

### 5. PID is Actually Good
PID gets 103 because:
- Integral term provides memory
- Derivative term provides smoothing
- It's tuned well for this task

MPC must leverage its advantage (model + lookahead) to beat it.

---

## What Winners Likely Did

To get <35 cost (vs PID's 103):

1. **Gradient-based optimization** (not sampling)
   - iLQR, DDP, or backprop through ONNX
   - Much more efficient than CEM
   
2. **Heavy compute**
   - GPU acceleration
   - Thousands of samples, not hundreds
   
3. **Better warm-start**
   - Feedforward from future_plan
   - Not just shifted zeros
   
4. **Jerk-aware from start**
   - Optimize for jerk explicitly
   - Not just lataccel tracking

---

## Next Steps if v7 Fails

1. **Try larger std** (0.12-0.15) to escape local minima
2. **More samples** (300-500) with parallel eval
3. **Gradient-based** (iLQR or torch autodiff through model)
4. **Better initialization** (use PID output as seed)
5. **Adaptive std** (anneal over CEM iterations)
