# Repository Cleanup Plan

## ✅ **KEEP (Working Code)**

### Core Training & Evaluation
- `train_bc_pid.py` - BC training (84 cost) ✅
- `train_ppo_parallel.py` - Fixed PPO with AsyncVectorEnv (110 mean, 82 median) ✅
- `eval_ppo_simple.py` - Sequential evaluation script ✅
- `controllers/ppo_parallel.py` - Fixed controller (model caching) ✅

### Weights (Keep Best)
- `bc_pid_checkpoint.pth` - BC weights (84 cost) ✅
- `bc_pid_best.pth` - BC weights ✅
- `ppo_parallel_best.pth` - PPO weights (110 mean, 82 median) ✅

### Core Infrastructure
- `tinyphysics.py` - Simulator ✅
- `eval.py` - Official evaluation ✅
- `beautiful_lander.py` - Reference PPO implementation ✅
- `controllers/` - Controller implementations ✅
- `README.md`, `requirements.txt` ✅

### Documentation (Recent)
- `BC_SUMMARY.md` - BC results ✅
- `PROGRESS.md` - Current status ✅
- `PARALLEL_REFACTOR.md` - Parallel implementation notes ✅
- `EXPERIMENT_HARNESS.md` - Experiment tracking docs ✅

## 🗑️ **MOVE TO attempts_2/ (Obsolete/Debug)**

### Obsolete Training Scripts
- `train_ppo_from_bc.py` - Sequential PPO (worse performance, superseded)
- `train_bc_with_tracking.py` - Wrapper script (not needed)
- `experiment_harness.py` - Experiment tracking (unused)
- `compare_experiments.py` - Analysis tool (unused)

### Debug/Test Scripts
- `test_ppo_controller.py` - Debug
- `test_ppo_costs.py` - Debug
- `test_ppo_parallel.py` - Debug (segfault)
- `test_ppo_simple.py` - Debug (segfault)
- `test_parallel_info.py` - Debug
- `test_async_speedup.py` - Debug (not run)

### Old Weights
- `ppo_best.pth` - Old sequential PPO weights (worse than parallel)

### Experiment Artifacts
- `experiments/` - Old experiment tracking data
- `report.html` - Old report

### Old Documentation
- Various markdown files from earlier attempts

## 📋 **Clean Structure After Cleanup**

```
controls_challenge/
├── train_bc_pid.py          # BC training
├── train_ppo_parallel.py    # PPO training (current best)
├── eval_ppo_simple.py       # Simple evaluation
├── tinyphysics.py           # Core simulator
├── eval.py                  # Official eval
├── beautiful_lander.py      # Reference
├── controllers/
│   ├── pid.py
│   ├── ppo_parallel.py      # Our controller
│   └── zero.py
├── bc_pid_checkpoint.pth    # BC weights
├── ppo_parallel_best.pth    # PPO weights
├── README.md
├── requirements.txt
├── BC_SUMMARY.md
├── PROGRESS.md
├── PARALLEL_REFACTOR.md
├── attempts/                # First cleanup
└── attempts_2/              # Second cleanup
```

