# Second Cleanup - December 25, 2024

This folder contains code that was superseded or found to be non-functional during the second round of development.

## What's Here

### Obsolete Training Scripts
- `train_ppo_from_bc.py` - Sequential PPO training (achieved 200-300 cost, superseded by parallel version)
- `train_bc_with_tracking.py` - Wrapper for BC with experiment tracking
- `experiment_harness.py` - Generic experiment tracking system (unused)
- `compare_experiments.py` - Experiment comparison tool (unused)

### Debug/Test Scripts
- `test_ppo_controller.py` - Early controller testing
- `test_ppo_costs.py` - Cost calculation debugging
- `test_ppo_parallel.py` - Parallel env testing (segfaulted)
- `test_ppo_simple.py` - Simple sequential testing (segfaulted)
- `test_parallel_info.py` - Environment info inspection
- `test_async_speedup.py` - AsyncVectorEnv speedup benchmark (not run)

### Old Weights
- `ppo_best.pth` - Sequential PPO weights (~200 cost, superseded by ppo_parallel_best.pth @ 110 cost)

### Experiment Artifacts
- `experiments/` - Experiment tracking data from harness
- `report.html` - Old evaluation report
- `EXPERIMENT_HARNESS.md` - Documentation for experiment tracking

## Why These Were Removed

1. **Sequential PPO performed worse** than parallel version (200 vs 110 cost)
2. **Debug scripts served their purpose** but are no longer needed
3. **Experiment tracking wasn't used** in final workflow
4. **Multiple segfaults** in test scripts due to NumPy/ONNX multiprocessing issues

## What Works (Kept in Root)

- `train_bc_pid.py` → 84 cost ✅
- `train_ppo_parallel.py` → 110 mean / 82 median cost ✅
- `eval_ppo_simple.py` → Clean evaluation ✅
- `controllers/ppo_parallel.py` → Fixed controller ✅

