# Experiment 002: PPO Baseline (Fixed Hyperparameters)

**Status**: ✅ Complete (unstable)  
**Date**: 2024-12-26

---

## Hypothesis
With fixed hyperparameters (reward function, lr, exploration), PPO should learn better than PID.

## Changes from Broken Version
1. ✅ Reward: `-(50*lat + jerk)` (was equal weight)
2. ✅ lr: 1e-3 (was 1e-5)
3. ✅ std: 1.0 (was 0.05)
4. ✅ entropy: 0.001 (was 0)
5. ✅ Episode cost tracking: Fixed to match eval

## Results

```
Epoch  0: 15,103 (random init)
Epoch  5:  2,839
Epoch 10:    497 ← Best
Epoch 15: 26,033 (exploded)
```

**Best cost: 497**

## Comparison
```
Target:  45
PID:    100
BC:     100
PPO:    497  ❌ 5× worse than PID
```

## Analysis

**What worked:**
- ✅ Learning happens (15k→500)
- ✅ Network has capacity

**What failed:**
- ❌ Unstable (explodes after epoch 10)
- ❌ 5× worse than PID
- ❌ Never reaches PID level

**Root cause:**
- Training instability
- No warm start (random init)
- Possibly need BC initialization

## Artifacts
- Checkpoint: `results/ppo_parallel_best.pth`
- Log: `results/ppo_training_fixed.log`
- Eval: `results/final_ppo_eval.npz`

## Next Steps
1. Try BC initialization (exp003)
2. Add a_ego to state (exp004)
3. Reduce learning rate if still unstable

