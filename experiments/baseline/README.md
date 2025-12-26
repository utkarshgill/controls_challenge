# Experiment 000: Baseline Evaluation

**Date**: 2024-12-25  
**Status**: ✅ Complete  
**Researcher**: engelbart

---

## Hypothesis
Establish baseline performance for PID, BC, and PPO controllers on a consistent test set.

---

## Motivation
Before attempting improvements, we need clean baseline numbers:
- PID: Simple but proven controller
- BC: Behavioral cloning from PID expert
- PPO: Reinforcement learning from BC initialization

This gives us a reference point for all future experiments.

---

## Method

### Controllers Tested
1. **PID**: Hand-tuned proportional-integral-derivative controller
   - Kp=0.3, Ki=0.05, Kd=0.1
   - Anti-windup: ±14 (from training data analysis)

2. **BC (Behavioral Cloning)**:
   - Architecture: 3-layer MLP, 128 hidden units
   - State dim: 56 (without a_ego!)
   - Trained on 5000 files, 50 epochs
   - Anti-windup: ±14

3. **PPO (Proximal Policy Optimization)**:
   - Architecture: Same as BC
   - Initialized from BC weights
   - Trained with parallel envs
   - Anti-windup: ±14

### State Representation (BC & PPO)
```
Features (56D):
  - error (lateral position error)
  - error_diff (derivative)
  - error_integral (with anti-windup ±14)
  - current_lataccel
  - v_ego (ego velocity)
  - curv_now (current curvature)
  - future_curv[50] (50 future curvature points)

OBS_SCALE: [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0]*50

MISSING: a_ego (longitudinal acceleration) ← KEY FINDING!
```

### Evaluation
- Test set: First 100 sorted files (easier subset)
- Metrics: mean, median, std, min, max, failure rate
- Failure definition: cost > 2× median

---

## Results

### Quantitative
| Metric | PID | BC | PPO |
|--------|-----|-----|-----|
| Mean cost | 80.4 | 92.4 | 93.8 |
| Median cost | 67.7 | 69.5 | 69.9 |
| Std dev | 61.7 | 142.4 | 152.1 |
| Min cost | 5.0 | 5.2 | 5.2 |
| Max cost | 374.9 | 1400.8 | 1507.8 |
| Failures (>2× median) | 9/100 | 8/100 | 8/100 |
| Catastrophic (>5× median) | 2/100 | 4/100 | 4/100 |

### Comparison to Target
```
Target:  < 45.0
PID:     80.4  (gap: 35.4 points)
BC:      92.4  (gap: 47.4 points)
PPO:     93.8  (gap: 48.8 points)
```

### Key Observations

1. **Median ≈ Equal**: All three controllers perform similarly on typical files
   - PID median: 67.7
   - BC median: 69.5 (+2.7%)
   - PPO median: 69.9 (+3.2%)

2. **Mean Diverges**: BC/PPO have higher mean due to catastrophic failures
   - PID: 2 catastrophic failures
   - BC/PPO: 4 catastrophic failures each

3. **Worst Case Failures**:
   - File 00069: PID=375, BC=1401, PPO=1508 (4× worse!)
   - File 00015: PID=23, BC=97, PPO=105 (4× worse!)
   - File 00025: PID=5, BC=78, PPO=82 (16× worse!)
   - File 00037: PID=8, BC=20, PPO=22 (2.5× worse!)

4. **BC ≈ PPO**: PPO didn't improve over BC
   - Suggests PPO training needs work OR
   - BC initialization is already near-optimal for this state representation

---

## Analysis

### What Worked
- ✅ Anti-windup (±14) prevents integral runaway on most files
- ✅ BC successfully clones PID on 96% of files
- ✅ State normalization (OBS_SCALE) is reasonable

### What Didn't Work
- ❌ BC/PPO fail catastrophically on 4% of files
- ❌ PPO didn't improve over BC (training issue or state limitation?)
- ❌ All controllers far from target (< 45)

### Surprising Findings

**The `a_ego` Discovery** 🔥

Analyzed failure file 00069:
```
Easy file (00000):  |a_ego| = 0.08 avg, 0.45 max
Hard file (00069):  |a_ego| = 0.43 avg, 3.75 max  ← 5× MORE!
```

**Physics - Friction Circle**:
```
√(a_lat² + a_long²) ≤ μ·g ≈ 9.8 m/s²

When braking at -3.75 m/s²:
  → Only ~9.0 m/s² left for lateral control
  → Same steering input produces LESS effect
  → BC doesn't know car is braking → commands too much steering
```

**Root Cause**: We removed `a_ego` from the state!
- Line 77 in train_bc_pid.py: `# a_ego removed: already in future_plan`
- But a_ego is NOT redundant - it's critical for friction coupling!
- PID doesn't need a_ego (purely reactive)
- BC/PPO NEED a_ego to understand grip limits

---

## Conclusion

**Did the hypothesis hold?** Yes - we established clean baselines.

**Why BC/PPO fail?** Missing `a_ego` feature causes catastrophic failures on files with high longitudinal acceleration.

**Key takeaway**: BC can clone PID on typical cases, but missing physics-critical features causes edge case failures.

---

## Next Steps

Based on these results:

1. **Experiment 001**: Add `a_ego` to BC state (HIGHEST PRIORITY)
   - Hypothesis: Fixes friction circle coupling
   - Expected: 92.4 → 85 (maybe better)
   - File 00069: 1401 → 500?

2. **Experiment 002**: Add explicit friction margin feature
   - After validating a_ego helps
   - `friction_margin = 1 - √(lat² + long²) / (μ·g)`

3. **Experiment 003**: PPO with correct state
   - After BC validates the state design
   - PPO should learn beyond PID

4. **Deep dive**: What did competition winner do?
   - If still far from 45 after above experiments

---

## Reproducibility

### Command to reproduce:
```bash
cd experiments/baseline
python final_evaluation.py
```

### Dependencies:
- Python 3.10
- PyTorch 2.0
- NumPy, Pandas
- ONNX Runtime

### Compute:
- Time: ~10 minutes on M1 Mac
- Resources: Single-threaded evaluation

---

## Artifacts

- Results: `final_results.npz`
- Checkpoints: 
  - BC: `bc_checkpoints/checkpoint_epoch_50.pt`
  - PPO: `ppo_checkpoints/best_model.pt`
- Analysis: `FINDINGS_SUMMARY.md`

