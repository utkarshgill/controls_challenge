# Systematic Experiment Plan

## Current Status
- **PID**: 80.4 (baseline)
- **BC (no a_ego)**: 92.4 
- **PPO (no a_ego)**: 93.8
- **Target**: < 45.0
- **Gap**: 35.4 points

## Key Finding
File 00069 has **5× more longitudinal acceleration** than easy files.
BC/PPO trained without `a_ego` → can't model friction circle coupling → catastrophic failures.

## Experiment Queue

### Experiment 1: BC with a_ego (HIGHEST PRIORITY)
**Hypothesis**: Adding a_ego will fix hard file failures

**Config**:
- State dim: 56 → 57
- Features: [error, error_diff, error_integral, lataccel, v_ego, **a_ego**, curv] + 50 future_curv
- OBS_SCALE: add 20.0 for a_ego (typical range ±4 m/s²)
- Anti-windup: ±14 (keep)

**Steps**:
1. Modify `train_bc_pid.py` line 77: ADD a_ego back
2. Collect expert data (5000 files)
3. Train BC (50 epochs)
4. Evaluate on 100 test files
5. Compare to baseline

**Expected outcome**: 
- Mean cost: 92.4 → 85? (closer to PID)
- File 00069: 1401 → 500? (still worse than PID but improved)

**Success criteria**:
- Mean < 90
- File 00069 < 800

---

### Experiment 2: BC with friction margin (if Exp 1 works)
**Hypothesis**: Explicit friction margin helps network understand limits

**Config**:
- State dim: 57 → 58
- Features: [..., a_ego, **friction_margin**, ...]
- friction_margin = 1 - √(lat² + long²) / (μ·g)
- This gives network explicit "how much grip left" signal

**Expected outcome**:
- Further improvement on hard files
- Mean cost: 85 → 82?

---

### Experiment 3: PPO with a_ego (after BC validates)
**Hypothesis**: PPO with correct state can learn beyond PID

**Config**:
- Same state as BC (with a_ego)
- Train from scratch OR initialize from BC weights
- More parallel envs, longer training

**Expected outcome**:
- PPO should exploit friction circle better than PID
- Mean cost: 82 → 65?

---

### Experiment 4: Advanced features (stretch goal)
If we're still far from 45:
- Future a_ego plan (not just current)
- Temporal features (CNN/LSTM on future plan)
- Explicit MPC-style cost prediction
- Ensemble methods

---

## Experiment Tracking

Each experiment will log:
- Config (state design, hyperparams)
- Training curves
- Evaluation metrics (mean, median, worst 5 files)
- Saved to `experiments/{timestamp}_{name}/`

## Decision Tree

```
Experiment 1 (BC + a_ego)
├─ Success (mean < 90)
│  ├─ Try Experiment 2 (friction margin)
│  └─ Try Experiment 3 (PPO with a_ego)
│
└─ Failure (mean ≥ 90)
   ├─ Debug: why didn't a_ego help?
   ├─ Check: is a_ego in training data correctly?
   └─ Alternative: try different state representations
```

---

## Quick Start

To run Experiment 1:
```bash
# 1. Modify train_bc_pid.py to include a_ego
# 2. Run:
python train_bc_pid.py

# 3. Evaluate:
python final_evaluation.py

# 4. Compare results
```

---

## Notes

- **Anti-windup (±14)** is mandatory - keep in all experiments
- **State normalization** matters - tune OBS_SCALE for new features
- **File 00069** is our canary - if we fix this, we're on the right track
- **Experiment harness** - use it to track all runs systematically

