# Experiment 003: PPO with BC Initialization

**Status**: 🏃 Ready to run  
**Date**: 2024-12-26

---

## Hypothesis

PPO training is unstable because it starts from random initialization (cost=15k).

BC already achieves cost~100. If we **initialize PPO with BC weights**, training should be:
- More stable (start from working policy)
- Faster (skip random exploration)
- Better (can explore improvements from good baseline)

## Changes from exp002

1. ✅ Initialize actor from BC checkpoint
2. ✅ Same hyperparameters (proven to cause learning)
3. ✅ Let PPO improve beyond BC

## Expected Results

```
Baseline:
- exp002 (random): 15k → 500 (unstable)
- BC: ~100 (stable but can't improve)

exp003 (BC init):
- Start: ~100
- After training: < 100 (hopefully < 45!)
```

## Comparison

```
Target:    45
PID:      100
BC:       100
exp002:   497 (random init, unstable)
exp003:   ??? (BC init)
```

## Run

```bash
bash experiments/exp003_ppo_bc_init/run.sh
```

## Results

*TBD after run*

---

## Analysis

*TBD*

