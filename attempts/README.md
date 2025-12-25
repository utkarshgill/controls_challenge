# Attempts Archive

This folder contains all the experimental work and failed attempts that led to the final BC+PPO solution.

## What's Here

**PPO Attempts:**
- `train_ppo_pure.py` - Pure PPO from random init (struggled, cost stuck at 4000-8000)
- `train_ppo_residual.py` - Residual PPO on top of PID baseline (plateaued at 72)
- `train_ppo_residual_v2.py` - V2 with full 55D state (similar results)
- `train_ppo.py` - Early PPO attempt

**BC+PPO Attempts:**
- `train_bc_ppo.py` - Two-stage BC→PPO (had bugs in BC stage, cost ~840+)

**Exploration:**
- `experiment.ipynb` - Jupyter notebook with various experiments
- `debug_bc_sync.py` - Debugging BC data collection

**Model Checkpoints:**
- `ppo_pure_best.pth` - Best from pure PPO (~4623 cost)
- `ppo_residual_best.pth` - Best from residual (~72 cost)
- `bc_ppo_best.pth` - From buggy BC+PPO attempt
- Various other .pth files

**Documentation:**
- `NORMALIZATION_FIX.md` - Early normalization debugging
- `REFACTOR_SUMMARY.md` - Refactoring notes

## Key Lessons Learned

1. **Pure PPO is unstable** without good initialization in autoregressive environments
2. **State must match PID exactly** - error_diff vs error_derivative mismatch killed first BC
3. **Dataset must be shuffled** - files were sorted by difficulty
4. **PID terms help** - giving network error/derivative/integral improves learning
5. **Parallel data collection** is essential for speed
6. **Warm start from BC** is the winning recipe

## What Worked

The final solution is in the parent directory:
- `train_bc_pid.py` - Clean BC implementation (~90 cost)
- Next: Add PPO fine-tuning stage (BC → <45 cost)

