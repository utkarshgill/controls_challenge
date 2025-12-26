# Experiment Harness - Quick Start

Simple experiment tracking for systematic BC/PPO development.

## Features

✅ **Track every run** - Parameters, metrics, duration, status
✅ **Compare runs** - See what works, what doesn't  
✅ **No dependencies** - Just Python stdlib + JSON
✅ **Fast** - Minimal overhead (<0.1s per run)

## Quick Start

### 1. Run an experiment

```bash
# BC with 1000 files (baseline)
python train_bc_with_tracking.py --name bc_1k --files 1000

# BC with 5000 files (more data)
python train_bc_with_tracking.py --name bc_5k --files 5000

# BC with different learning rate
python train_bc_with_tracking.py --name bc_lr_small --files 1000 --lr 0.0001
```

### 2. Compare results

```bash
# Show all BC runs
python compare_experiments.py bc

# Compare by validation cost
python compare_experiments.py bc --metric bc_val_cost

# Show best run
python compare_experiments.py bc --metric bc_val_cost --best
```

### 3. Results are saved automatically

```
experiments/
└── bc_experiments/
    ├── runs.jsonl                      # All runs (one per line)
    ├── bc_1k_20251225_120530/          # Run directory
    │   └── run_info.json               # Full run details
    ├── bc_5k_20251225_121045/
    │   └── run_info.json
    └── ...
```

## Example Output

```
================================================================================
Experiment: bc_experiments
Total runs: 3
================================================================================

Run                  Status     Duration     Key Metrics
--------------------------------------------------------------------------------
bc_1k                completed  180.2s       bc_train_cost=114.4, bc_val_cost=398.9
bc_5k                completed  420.5s       bc_train_cost=95.2, bc_val_cost=120.3
bc_lr_small          completed  185.3s       bc_train_cost=125.8, bc_val_cost=410.2
```

## Systematic Testing

**Hypothesis: More training data reduces overfitting**

```bash
# Test 1: Baseline (1000 files)
python train_bc_with_tracking.py --name test_1k --files 1000

# Test 2: 5x more data
python train_bc_with_tracking.py --name test_5k --files 5000

# Test 3: 10x more data
python train_bc_with_tracking.py --name test_10k --files 10000

# Compare
python compare_experiments.py bc --metric bc_val_cost
```

Now you can see:
- Which has lowest val cost?
- Does train/val gap improve?
- Is the extra time worth it?

## Integration with train_bc_pid.py

**Current limitation:** `train_bc_pid.py` doesn't accept parameters yet.

**Workaround:** `train_bc_with_tracking.py` wraps it but can't customize yet.

**TODO:** Refactor `train_bc_pid.py` to:
```python
def train_bc(n_expert_files=1000, n_epochs=50, lr=1e-3, ...):
    # ... training code ...
    return network, metrics
```

Then tracking becomes automatic!

## Why This Helps

**Before:**
- Run experiment, forget parameters
- Can't compare runs easily
- Waste time repeating experiments
- No systematic approach

**After:**
- Every run logged with full context
- Easy comparison: "Did 5000 files help?"
- Build intuition: "Higher LR → worse val"
- Scientific method: hypothesis → test → conclusion

## Files

- `experiment_harness.py` - Core tracking logic
- `train_bc_with_tracking.py` - BC training with tracking
- `compare_experiments.py` - Analysis tool
- `experiments/` - All run data (gitignored)

