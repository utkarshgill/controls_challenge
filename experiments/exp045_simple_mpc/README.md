# Exp045: Simple Model Predictive Control (MPC)

## Goal
Achieve cost <100 using MPC as a baseline for comparison with learned controllers.

## MPC Architecture

### Core Principle
**"Search-based optimal control"** - No learning, pure optimization at every timestep.

```
At each timestep (500 per trajectory):
1. Receive: 50-step future plan (target path + vehicle dynamics)
2. Sample N action sequences [u₀, u₁, ..., u_H]
3. Roll each forward using TinyPhysics ONNX model
4. Compute cost = lataccel_cost + jerk_cost + action_smoothness
5. Pick best sequence
6. Execute ONLY u₀ (receding horizon principle)
7. Repeat
```

### Key Insight
MPC has no weights/parameters! The "knowledge" is in:
- **Physics model** (`tinyphysics.onnx`) - world dynamics
- **Cost function** - definition of "good driving"
- **Optimizer** (CEM) - search algorithm

## Results Summary

| Config | Horizon | Samples | Weight | Std | Time | Cost | Lataccel | Jerk |
|--------|---------|---------|--------|-----|------|------|----------|------|
| v1     | 8       | 150     | 50     | 0.15| 6min | 1614 | 18.0     | 713  |
| v2     | 8       | 100     | 200    | 0.15| 12min| 357  | 5.3      | 94   |
| v3     | 12      | 150     | 1000   | 0.15| 36min| 157  | 2.2      | 48   |
| **v4** | **15**  | **100** | **5000**|**0.08**|**TBD**|**TBD**|**TBD**|**TBD**|

## Speed Optimizations (v4)

### Changes from v3:
1. **Horizon**: 12 → 15 steps (+25%, but still 3x faster than H=20)
2. **Samples**: 150 → 100 (-33% evals)
3. **Smoothness weight**: 1000 → 5000 (compensate for shorter horizon)
4. **Action std**: 0.15 → 0.08 (less noisy exploration)

### Expected speedup: **2x** (36min → ~18min)

Computation per timestep:
```
v3: 150 samples × 12 steps × 4 iters = 7,200 model calls
v4: 100 samples × 15 steps × 3 iters = 4,500 model calls (-37%)
```

## Cost Function Details

### Simulator Cost (what's reported):
```python
lataccel_cost = mean((pred - target)²) × 100
jerk_cost = mean((Δlat_accel / 0.1s)²) × 100
total_cost = lataccel_cost × 50 + jerk_cost
```

### MPC Internal Cost (for optimization):
```python
# Same as above, PLUS:
action_smooth_cost = mean((Δaction)²) × 5000  # Not in final cost!
```

**Key insight**: Action smoothness is only used INSIDE optimization to guide CEM toward smooth steering sequences. The final reported cost doesn't include it.

## Why MPC?

### Advantages:
- ✅ No training data needed
- ✅ Always optimal given current situation
- ✅ Can handle constraints explicitly
- ✅ Interpretable (we see exactly what it optimizes)
- ✅ Establishes performance ceiling for learned methods

### Disadvantages:
- ❌ **SLOW** (18-36min per trajectory vs <1s for NN)
- ❌ Needs accurate physics model
- ❌ Doesn't improve with experience
- ❌ Can't deploy in real-time (500×slower than needed)

## Next Steps

If v4 achieves <100:
1. Use MPC to generate expert demonstrations
2. Train NN to imitate MPC (behavior cloning)
3. Distill MPC's search into fast feedforward policy
4. Compare: BC-from-MPC vs BC-from-PID

If v4 doesn't reach <100:
- Try H=20-25 with parallelization
- Tune smoothness weight 5000-10000
- Reduce std further 0.08 → 0.05
- Add jerk prediction in internal cost

## Implementation Notes

### CEM (Cross-Entropy Method):
Iterative refinement of action distribution:
```
Init: mean=0, std=action_std
For iter in 1..3:
    Sample 100 sequences from N(mean, std)
    Keep top 25 (elites)
    mean = elites.mean()
    std = elites.std() + ε
Return mean[0]  # First action only
```

### Warm-Starting:
```python
# Previous solution: [u₀, u₁, ..., u_H]
# Warm start: [u₁, u₂, ..., u_H, 0]  # Shift by 1
# Sample around warm_start ± std
```

Provides temporal consistency → smoother actions.

### Running Cost Diagnostics:
Every 50 steps, print:
```
Step 150: running_cost=145.2 (lat=98.5, jerk=46.7)
```
Helps track if cost is stable, improving, or degrading during trajectory.

## References
- `tinyphysics.py`: CONTROL_START_IDX=100, COST_END_IDX=500
- `README.md`: Official cost formula
- `archive/beautiful_lander.py`: PPO example for comparison
