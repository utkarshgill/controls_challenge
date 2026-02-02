# Parallelization Strategy for MPC

## Key Insight: Use Speedup to Afford Better Quality

**Wrong approach:** Parallelize → reduce params → get worse quality but faster
**Right approach:** Parallelize → INCREASE params → get better quality at same time!

---

## Results Comparison

| Version | H | S | W | Std | Parallel | Time | Cost | Quality |
|---------|---|---|---|-----|----------|------|------|---------|
| Baseline | 12 | 150 | 1000 | 0.15 | No | 36min | 157 | ⭐⭐⭐ |
| Speed opt | 15 | 100 | 5000 | 0.08 | No | 18min | ~140* | ⭐⭐⭐⭐ |
| **Parallel bad** | **12** | **80** | **6000** | **0.08** | **Yes** | **6min** | **192** | **⭐⭐** ❌ |
| **Parallel good** | **15** | **120** | **8000** | **0.06** | **Yes** | **~7min** | **TBD** | **⭐⭐⭐⭐⭐?** |

*Projected from partial run

---

## What Changed?

### Version 1: Parallel Bad (192 cost in 6 min)
```python
# Reduced everything for speed
H=12, samples=80, weight=6000
→ Fast but poor quality (worse than non-parallel!)
```

### Version 2: Parallel Good (TBD cost in ~7 min)
```python
# Leveraged parallelization to INCREASE quality
H=15,      # +25% look-ahead (12 → 15)
samples=120,  # +50% exploration (80 → 120)  
weight=8000,  # +33% smoothing (6000 → 8000)
std=0.06      # -25% noise (0.08 → 0.06)

# With 9-worker parallelization:
# Computation: 120 × 15 × 3 = 5,400 evals/step
# Sequential time: ~18-20 min
# Parallel time: ~6-8 min (3x speedup)
```

---

## Why Parallelization Works

**ThreadPoolExecutor with ONNX:**
```python
# Each sequence evaluation:
def _evaluate_sequence(seq):
    for t in range(H):
        lataccel = model.get_current_lataccel(...)  # ← ONNX releases GIL!
    return cost

# Parallel execution:
with ThreadPoolExecutor(9) as pool:
    costs = pool.map(_evaluate_sequence, sequences)
```

**Key factors:**
1. **ONNX Runtime releases Python's GIL** during model inference
2. **Each evaluation is independent** (no shared state)
3. **9 workers on 12 cores** (75% utilization, leave headroom)
4. **Overhead is low** (minimal data passing between threads)

**Observed speedup: 2-3x** (not perfect 9x due to overhead + GIL contention)

---

## The Math

### Without Parallelization:
```
500 steps × 120 samples × 15 horizon × 3 iters = 2.7M model calls
Each call: ~0.001s
Total: 2,700s = 45 min pure compute
With overhead: ~50 min real time
```

### With Parallelization (9 workers):
```
Same 2.7M calls, but divided across 9 threads
Pure compute: 2,700s / 9 = 300s = 5 min
With overhead: ~7-8 min real time
```

**Efficiency: 45min / 7min = 6.4x speedup!**

---

## Why Did V1 Fail?

**Mistake: Reduced too many parameters at once**
```
Baseline:  H=12, S=150 → 12 × 150 = 1,800 evals
Parallel:  H=12, S=80  → 12 × 80 = 960 evals  (-47%!)
```

**Lost too much exploration:**
- Fewer samples (150 → 80) = less search coverage
- Weight increased but couldn't compensate

**Result: Fast but poor tracking (192 cost vs 157 baseline)**

---

## Why Should V2 Succeed?

**Strategy: Use parallelization headroom for quality**
```
Speed opt: H=15, S=100 → 15 × 100 = 1,500 evals (18 min sequential)
Parallel:  H=15, S=120 → 15 × 120 = 1,800 evals (6-8 min parallel)
```

**Improvements:**
- **+20% samples** (100 → 120) = better exploration
- **+60% smoothing** (5000 → 8000) = smoother actions
- **-25% noise** (0.08 → 0.06) = tighter convergence

**Expected: Cost ~120-130 in 7 min** (vs 157 in 36 min baseline)

---

## Lessons Learned

1. **Parallelization ≠ reduce quality**
   - Use speedup to INCREASE params, not decrease them

2. **Threading works for ONNX/NumPy**
   - Despite Python's GIL, ONNX releases it
   - 2-3x speedup achievable with threads

3. **Balance is key**
   - Too few samples: poor coverage
   - Too many workers: diminishing returns
   - Sweet spot: 75% of available cores

4. **Smoothness weight is critical**
   - 6000 was not enough (jerk still 53)
   - 8000 should push jerk down to 30-40
   - Target: lat~50 + jerk~30 = 80 total cost!

---

## Next Optimization: Process Pool

**Current: ThreadPoolExecutor (limited by GIL contention)**
**Future: ProcessPoolExecutor (true parallelism)**

**Challenges:**
- Need to pickle model/data
- Higher overhead (process creation)
- Memory duplication

**Potential gain: 6-9x speedup** (closer to # of cores)

But for now, **ThreadPoolExecutor is good enough!**
