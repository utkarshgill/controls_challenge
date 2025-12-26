# Experiment 001: BC with a_ego (Friction Circle)

**Date**: 2024-12-25  
**Status**: 🏃 Ready to Run  
**Researcher**: engelbart

---

## Hypothesis
Adding `a_ego` (longitudinal acceleration) to the state will fix catastrophic failures on files with high braking/acceleration by enabling the network to understand friction circle coupling.

---

## Motivation

### The Problem
Baseline BC fails catastrophically on 4% of files:
- File 00069: PID=375, BC=1401 (3.7× worse)
- File 00015: PID=23, BC=97 (4.2× worse)
- File 00025: PID=5, BC=78 (15.7× worse)

### The Discovery
File 00069 has **5× more longitudinal acceleration**:
```
Easy file (00000):  |a_ego| = 0.08 avg, 0.45 max
Hard file (00069):  |a_ego| = 0.43 avg, 3.75 max  ← 5× MORE!
```

### The Physics
**Friction Circle**: `√(a_lat² + a_long²) ≤ μ·g ≈ 9.8 m/s²`

When braking at -3.75 m/s²:
- Only √(9.8² - 3.75²) ≈ 9.0 m/s² left for lateral control
- Same steering input produces LESS lateral force
- BC without a_ego doesn't know this → commands too much steering

### Why We Removed It
Line 77 in `train_bc_pid.py`:
```python
# a_ego removed: already in future_plan
```

**But this was wrong!** 
- Current a_ego affects CURRENT friction limits
- Future plan doesn't tell us about NOW
- This is a critical physics coupling

### Why PID Works Without It
- PID is purely reactive (error-based)
- Doesn't try to exploit full friction circle
- Conservative by design → safe but not optimal

### Why BC Needs It
- BC tries to clone PID's actions
- PID implicitly accounts for grip (through error feedback)
- BC needs explicit a_ego to understand when grip is limited

---

## Method

### Model Architecture
- Network: 3-layer MLP, 128 hidden units
- Input dim: 56 → **57** (+1 for a_ego)
- Output: continuous action ∈ [-2, 2]
- Activation: ReLU

### State Representation
```python
Features (57D):
  - error
  - error_diff
  - error_integral (anti-windup ±14)
  - current_lataccel
  - v_ego
  - a_ego  ← NEW!
  - curv_now
  - future_curv[50]

OBS_SCALE: [10.0, 1.0, 0.1, 2.0, 0.03, 20.0, 1000.0] + [1000.0]*50
                                          ^^^^
                                          NEW: a_ego scale
                                          (typical range ±4 m/s²)
```

### Training
- Dataset: 5000 files (same as baseline)
- Expert: PID with anti-windup ±14
- Epochs: 50
- Batch size: 256
- Learning rate: 1e-3
- Optimizer: Adam
- Loss: MSE between BC action and PID action

### Evaluation
- Test set: Same 100 files as baseline
- Metrics: mean, median, std, failures
- Focus: File 00069 (canary for a_ego hypothesis)

---

## Expected Results

### Optimistic
```
Baseline BC: 92.4 mean, 1401 on file 00069
With a_ego:  85.0 mean,  500 on file 00069

Improvement: 7.4 points mean, 2.8× better on hard file
```

### Realistic
```
With a_ego: 88.0 mean, 700 on file 00069

Improvement: 4.4 points mean, 2× better on hard file
```

### Pessimistic (if hypothesis is wrong)
```
With a_ego: 92.0 mean, 1300 on file 00069

Improvement: Minimal - a_ego doesn't help much
```

---

## Success Criteria

**Minimum bar** (hypothesis confirmed):
- Mean cost < 90
- File 00069 cost < 800
- Fewer catastrophic failures (< 4)

**Good result** (strong confirmation):
- Mean cost < 85
- File 00069 cost < 500
- Catastrophic failures ≤ 2

**Excellent result** (exceeds expectations):
- Mean cost < 82
- File 00069 cost < 400
- Catastrophic failures = 0

---

## Implementation Plan

### Step 1: Modify State Builder (30 min)
```python
# In train_bc_pid.py, line 77
# BEFORE:
obs = np.concatenate([
    [error, error_diff, error_integral, current_lataccel, v_ego, curv_now],
    future_curv
])

# AFTER:
obs = np.concatenate([
    [error, error_diff, error_integral, current_lataccel, v_ego, a_ego, curv_now],
    future_curv
])
```

### Step 2: Update OBS_SCALE (5 min)
```python
# Add a_ego scale
OBS_SCALE = np.array([10.0, 1.0, 0.1, 2.0, 0.03, 20.0, 1000.0] + [1000.0] * 50)
```

### Step 3: Update Network Dimensions (5 min)
```python
# In BCNetwork
self.input_dim = 57  # was 56
```

### Step 4: Collect New Expert Data (1 hour)
```bash
python train_bc_pid.py --collect-only
```

### Step 5: Train BC (1 hour)
```bash
python train_bc_pid.py --train
```

### Step 6: Evaluate (10 min)
```bash
python final_evaluation.py
```

### Step 7: Analyze Results (30 min)
- Compare to baseline
- Check file 00069 specifically
- Look at failure patterns

**Total time**: ~3 hours

---

## Analysis Plan

### If It Works (mean < 90)
1. ✅ Hypothesis confirmed: a_ego matters
2. Document improvement on hard files
3. Move to Experiment 002 (friction margin)
4. Move to Experiment 003 (PPO with a_ego)

### If It Partially Works (90 ≤ mean < 92)
1. 🤔 Hypothesis weakly supported
2. Investigate: why only small improvement?
3. Check: is a_ego being used by network? (gradient analysis)
4. Consider: maybe need explicit friction margin feature

### If It Doesn't Work (mean ≥ 92)
1. ❌ Hypothesis rejected
2. Debug: is a_ego in training data correctly?
3. Debug: is normalization correct?
4. Alternative: maybe network capacity is the issue?
5. Alternative: maybe need different architecture (LSTM, attention)?

---

## Risks & Mitigations

**Risk 1**: a_ego doesn't help
- Mitigation: Have backup hypothesis (friction margin, temporal features)

**Risk 2**: Training takes too long
- Mitigation: Use same 5000 files as baseline (proven to work)

**Risk 3**: New OOD issues with a_ego
- Mitigation: Check a_ego distribution in training data first

**Risk 4**: Network doesn't use a_ego
- Mitigation: Gradient analysis, ablation study

---

## Reproducibility

### Command to reproduce:
```bash
cd experiments/exp001_bc_with_a_ego
python run.py --config config.yaml
```

### Modified files:
- `train_bc_pid.py`: Line 77 (add a_ego), line 40 (OBS_SCALE)
- `controllers/bc_pid.py`: Same changes for inference

### Checkpoints:
- Will be saved to `results/checkpoints/`

---

## References

- Baseline experiment: `experiments/baseline/`
- a_ego analysis: `scripts/test_a_ego_hypothesis.py`
- Findings: `docs/FINDINGS_SUMMARY.md`

---

## Notes

This is our **highest priority** experiment. The physics reasoning is sound, the data supports it, and the implementation is straightforward. If this works, it validates our scientific approach and opens the door to further improvements.

