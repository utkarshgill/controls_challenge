# Migration Guide - Clean Project Structure

## Goal
Transform messy research repo into clean, organized structure following best practices.

---

## Current State (Messy Root)

```
controls_challenge/
├── 50+ files in root directory ❌
├── Multiple analysis scripts scattered
├── Old experiment folders (exp01, exp02)
├── Duplicate documentation files
├── Mixed concerns (training, eval, analysis, docs)
└── Hard to find anything!
```

---

## Target State (Clean Root)

```
controls_challenge/
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
├── tinyphysics.py              # Core simulator (keep in root for now)
│
├── data/                        # Raw data (unchanged)
├── models/                      # Pretrained models (unchanged)
│
├── src/                         # Core library code
│   ├── controllers/            # Controller implementations
│   ├── networks/               # Neural architectures
│   ├── training/               # Training utilities
│   └── utils/                  # Shared utilities
│
├── experiments/                 # All experiments (organized!)
│   ├── baseline/               # Exp 000: Baselines
│   ├── exp001_bc_with_a_ego/   # Exp 001: Add a_ego
│   └── template/               # Template for new experiments
│
├── scripts/                     # One-off analysis scripts
│   ├── manage_experiments.py   # Experiment management
│   ├── analyze_failures.py
│   └── test_a_ego_hypothesis.py
│
├── notebooks/                   # Jupyter notebooks
│
├── docs/                        # Documentation
│   ├── FINDINGS_SUMMARY.md
│   ├── EXPERIMENT_PLAN.md
│   └── PROJECT_STRUCTURE.md
│
└── archive/                     # Old/deprecated code
    ├── attempts/
    ├── attempts_2/
    └── old_experiments/
```

---

## Migration Steps

### Phase 1: Move Core Library Code ✅ DONE
```bash
mkdir -p src/{controllers,networks,training,utils}
# Controllers already in src/controllers/
```

### Phase 2: Organize Scripts
```bash
# Move analysis scripts
mv analyze_failures.py scripts/
mv test_a_ego_hypothesis.py scripts/
mv check_*.py scripts/
mv diagnose_*.py scripts/
mv evaluate_*.py scripts/
mv verify_*.py scripts/
mv test_*.py scripts/

# Move evaluation scripts
mv eval*.py scripts/
mv baseline.py scripts/
mv final_*.py scripts/
```

### Phase 3: Organize Documentation
```bash
# Move docs
mv FINDINGS_SUMMARY.md docs/
mv EXPERIMENT_PLAN.md docs/
mv PROJECT_STRUCTURE.md docs/
mv MIGRATION_GUIDE.md docs/

# Archive old docs
mv BC_SUMMARY.md archive/docs/
mv PROGRESS.md archive/docs/
mv STATUS.md archive/docs/
mv STRUCTURE.md archive/docs/
mv WHAT_WORKS.md archive/docs/
mv PARALLEL_REFACTOR.md archive/docs/
mv PIPELINE_GUIDE.md archive/docs/
```

### Phase 4: Clean Up Experiments
```bash
# Move old experiment folders to archive
mv exp01_bc_baseline archive/old_experiments/
mv exp02_bc_with_a_ego archive/old_experiments/

# Keep only organized experiments/
# - baseline/
# - exp001_bc_with_a_ego/
# - template/
```

### Phase 5: Archive Old Attempts
```bash
# Already have attempts/ and attempts_2/
# Just move them to archive/
mv attempts archive/attempts_1
mv attempts_2 archive/attempts_2
```

### Phase 6: Clean Up Training Scripts
```bash
# Keep in root for now (actively used):
# - train_bc_pid.py
# - train_ppo_parallel.py
# - train_pipeline.py

# Archive old training scripts
mv back_to_basics.py archive/
```

### Phase 7: Clean Up Checkpoints
```bash
# Move loose checkpoints to appropriate experiment folders
mv bc_pid_best.pth experiments/baseline/results/checkpoints/
mv bc_pid_checkpoint.pth experiments/baseline/results/checkpoints/
mv ppo_parallel_best.pth experiments/baseline/results/checkpoints/

# Move results
mv final_results.npz experiments/baseline/results/
mv baseline_results.npz experiments/baseline/results/
```

### Phase 8: Clean Up Config Files
```bash
# Move experiment config to archive (replaced by YAML configs)
mv experiment_config.py archive/
```

---

## After Migration

### Root Directory (Clean!)
```
controls_challenge/
├── README.md
├── requirements.txt
├── tinyphysics.py
├── train_bc_pid.py          # Active training scripts
├── train_ppo_parallel.py
├── train_pipeline.py
│
├── data/                     # Data
├── models/                   # Models
├── src/                      # Library code
├── experiments/              # Organized experiments
├── scripts/                  # Analysis scripts
├── notebooks/                # Notebooks
├── docs/                     # Documentation
└── archive/                  # Old code
```

**Count**: ~10 items in root (vs 50+ before!)

---

## Usage After Migration

### Run an experiment:
```bash
cd experiments/exp001_bc_with_a_ego
python run.py --config config.yaml
```

### List all experiments:
```bash
python scripts/manage_experiments.py list
```

### Compare results:
```bash
python scripts/manage_experiments.py compare baseline exp001_bc_with_a_ego
```

### Create new experiment:
```bash
python scripts/manage_experiments.py new exp002_my_idea "Test my hypothesis"
```

### Run analysis:
```bash
python scripts/analyze_failures.py
python scripts/test_a_ego_hypothesis.py
```

---

## Benefits

1. **Clean root**: Easy to navigate
2. **Organized experiments**: Each has its own folder
3. **Clear separation**: Code vs experiments vs docs vs archive
4. **Reproducible**: Each experiment is self-contained
5. **Scalable**: Easy to add new experiments
6. **Professional**: Looks like a real research lab!

---

## Rollback Plan

If something breaks:
```bash
# Everything is moved, not deleted
# Can always move back from archive/
```

---

## Next Steps

1. ✅ Create structure (DONE)
2. ⏳ Run migration script (NEXT)
3. ⏳ Test that everything still works
4. ⏳ Update README with new structure
5. ⏳ Run Experiment 001 (BC with a_ego)

