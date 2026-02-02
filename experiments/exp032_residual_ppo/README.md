# Experiment 032: Residual PPO (Nullspace Learning)

**Status**: 🧪 Physics-First Design
**Date**: 2025-01-04

## The Core Insight

> **PPO is not failing to learn. It is optimizing the wrong geometry.**

PID sits at a local optimum in the cost landscape. Most policy perturbations make things worse. PPO from scratch explores in all directions - 99% of which are harmful (increase jerk, destabilize tracking).

**Solution**: Constrain PPO to explore only in the **feedforward nullspace** - the subspace of anticipatory corrections that don't fight the feedback loop.

## Architecture

```
u(t) = u_PID(error, integral, derivative) + ε × π_θ(s_preview)
       ↑                                      ↑
   Frozen baseline                    Learned residual
   (feedback stability)          (anticipatory corrections)
```

### Parameters:
- `ε = 0.1`: Residual is 10% of full action range
- Residual clipped to ±0.5 before scaling
- Low-pass filter (α=0.3) for smoothness
- State emphasizes future curvatures (normalized properly)

## Why This Works

### Physics Constraints:
1. **PID cannot be removed** - provides fundamental feedback stability
2. **Small corrections only** - stay in PID's basin of attraction  
3. **Low-pass filtering** - prevents high-frequency oscillations (jerk penalty)
4. **Feedforward focus** - residual uses future information PID ignores

### Control Theory:
- PID handles **reactive** control (error correction)
- Residual handles **predictive** control (curve anticipation)
- Combined system has:
  - Feedback stability (from PID)
  - Anticipatory performance (from residual)

## Key Fixes from Exp031

### 1. Reward Structure (CRITICAL)
**Fixed**: Only instantaneous step costs, no episode-end double-counting
```python
step_cost = 50 × lat_cost + jerk_cost
reward = -step_cost / 100.0
# NO additional reward at episode termination
```

### 2. State Normalization (CRITICAL)
**Fixed**: Curvatures scaled to ~[-1, 1] like other features
```python
curvature = (lataccel - roll) / v²
curvature_normalized = curvature / 0.01  # Typical scale
```

### 3. Action Space (NEW)
**Changed**: Network outputs small residual, not full control
```python
residual_raw → tanh → clip(±0.5) → lowpass → ×0.1 → residual
action = pid_output + residual
```

## Expected Learning Curve

```
Epoch   0: cost ~80  (PID baseline, ε=0 effectively)
Epoch  20: cost ~75  (residual learning slow improvements)  
Epoch  50: cost ~65  (anticipatory adjustments taking effect)
Epoch 100: cost ~55  (using preview information well)
Epoch 200: cost ~45  (competitive performance)
```

**If cost doesn't improve past 80**:
- Residual is too constrained (increase ε to 0.2)
- Network not using preview (check gradients on curvature features)
- PID baseline is suboptimal (tune PID gains first)

**If cost gets worse (>100)**:
- Residual scale too large (decrease ε to 0.05)
- Filtering insufficient (decrease α to 0.1)
- Exploration too aggressive (decrease entropy_coef)

## Training

```bash
cd experiments/exp032_residual_ppo
python train_ppo.py
```

Expected time: ~1 hour for 200 epochs on CPU

## Evaluation

```bash
# Single route
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller exp032_residual

# Batch evaluation
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller exp032_residual

# Full report
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller exp032_residual --baseline_controller pid
```

## Physics Validation

To verify the residual is learning correctly:

1. **Ablation test**: Set ε=0, should recover PID performance (~80 cost)
2. **Preview test**: Zero out curvature features, performance should degrade
3. **Action analysis**: Residual should be small (~0.1-0.2 magnitude) and smooth
4. **Timing test**: Residual should activate *before* sharp curves (anticipatory)

## Why This is the Right Approach

From the user's analysis:

> "The winner's PPO is not 'stronger PPO'. It is PPO constrained to explore only in the nullspace of PID."

This experiment embodies that principle:
- ✅ PID frozen (feedback stability preserved)
- ✅ Small residual (stay in stable basin)
- ✅ Filtered output (jerk constraint enforced)
- ✅ Preview-focused state (feedforward information emphasized)

The alternative (exp031) tried to learn full control from scratch:
- ❌ Random initialization (starts at cost 400,000)
- ❌ Exploration destroyed smoothness
- ❌ Never found PID's basin of attraction
- ❌ PPO optimizing local noise, not global structure

## Next Steps

If this achieves <50 cost:
1. **Tune ε**: Try 0.15, 0.2 to allow larger improvements
2. **Adaptive ε**: Schedule ε during training (start 0.05, increase to 0.2)
3. **State engineering**: Add velocity-aware features, friction circle awareness
4. **Architecture**: Try attention over future curvatures

If this fails to beat PID:
1. **Verify PID baseline** is correctly implemented
2. **Check preview usage** - ablate curvatures, should see performance drop
3. **Inspect residuals** - are they correlated with upcoming curves?
4. **Consider**: The simulator may not actually reward anticipation

This is the experiment that should prove whether PPO can learn feedforward control in a physically-grounded way.

