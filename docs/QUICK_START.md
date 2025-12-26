# Quick Start Guide

## For the Impatient Scientist 🚀

### 1. What's the status?
```bash
python scripts/manage_experiments.py status
```

Output:
```
Total experiments: 2
  ✅ Completed: 1 (baseline)
  🏃 Running/Ready: 1 (exp001_bc_with_a_ego)

🏆 Best result: baseline (80.4)
   Target: < 45.0
   Gap: 35.4 points
```

---

### 2. What did we discover?
```bash
cat docs/FINDINGS_SUMMARY.md
```

**TL;DR**: 
- BC works on 96% of files
- Fails catastrophically on 4% (high `a_ego` files)
- **Root cause**: We removed `a_ego` from the state
- **Fix**: Add it back (Experiment 001)

---

### 3. What's next?
```bash
cd experiments/exp001_bc_with_a_ego
cat README.md
```

**Experiment 001**: Add `a_ego` (longitudinal acceleration) to state
- **Hypothesis**: Fixes friction circle coupling
- **Expected**: 92.4 → 85 (or better)
- **Time**: ~3 hours (collect data + train + eval)

---

### 4. How do I run it?

#### Option A: Quick test (30 min)
```bash
# Just modify the eval script to test hypothesis
python scripts/test_a_ego_hypothesis.py
```

#### Option B: Full retrain (3 hours)
```bash
# 1. Modify train_bc_pid.py to include a_ego
# Line 77: Add a_ego to state
# Line 40: Add 20.0 to OBS_SCALE

# 2. Collect new expert data
python train_bc_pid.py --files 5000

# 3. Evaluate
python scripts/final_evaluation.py
```

---

### 5. How do I track experiments?

#### List all experiments
```bash
python scripts/manage_experiments.py list
```

#### Compare results
```bash
python scripts/manage_experiments.py compare baseline exp001_bc_with_a_ego
```

#### Create new experiment
```bash
python scripts/manage_experiments.py new exp002_my_idea "Test friction margin"
```

---

### 6. Project structure at a glance

```
controls_challenge/
├── experiments/          ← All experiments here
│   ├── baseline/        ← Exp 000: Current results
│   ├── exp001_.../      ← Exp 001: Next to run
│   └── template/        ← Copy this for new experiments
│
├── scripts/             ← Analysis tools
│   ├── manage_experiments.py
│   └── final_evaluation.py
│
├── docs/                ← Read these!
│   ├── FINDINGS_SUMMARY.md    ← What we learned
│   ├── EXPERIMENT_PLAN.md     ← Roadmap
│   └── QUICK_START.md         ← You are here
│
└── train_bc_pid.py      ← Main training script
```

---

### 7. Key commands

```bash
# Evaluate current models
python scripts/final_evaluation.py

# Train BC from scratch
python train_bc_pid.py

# Train PPO
python train_ppo_parallel.py

# Analyze failures
python scripts/analyze_failures.py

# Test a_ego hypothesis
python scripts/test_a_ego_hypothesis.py
```

---

### 8. Current results

```
Controller  Mean    Median  Failures
PID         80.4    67.7    9/100
BC          92.4    69.5    8/100
PPO         93.8    69.9    8/100
Target      < 45    -       -
```

**Gap to target**: 35.4 points

---

### 9. What's the plan?

```
Experiment 001: BC + a_ego
├─ Success (< 90)
│  ├─ Exp 002: Add friction margin
│  └─ Exp 003: PPO with a_ego
│
└─ Failure (≥ 90)
   └─ Debug: Why didn't a_ego help?
```

---

### 10. How do I clean up the repo?

```bash
# Preview migration
python scripts/migrate_structure.py --dry-run

# Execute migration (moves files to proper folders)
python scripts/migrate_structure.py --execute
```

This will:
- Move analysis scripts to `scripts/`
- Move docs to `docs/`
- Archive old experiments
- Clean up root directory

---

## FAQ

**Q: Where are the checkpoints?**  
A: `experiments/baseline/results/checkpoints/`

**Q: Where are the results?**  
A: `experiments/baseline/results/final_results.npz`

**Q: How do I add a new feature to the state?**  
A: 
1. Modify `train_bc_pid.py` line 77 (state builder)
2. Update `OBS_SCALE` line 40
3. Update network input dim
4. Retrain

**Q: How long does training take?**  
A: 
- BC: ~1 hour (5000 files, 50 epochs)
- PPO: ~2-3 hours (depends on steps_per_epoch)

**Q: Can I use GPU?**  
A: Yes, but current code is CPU-only. Would need to modify training scripts.

**Q: What if I break something?**  
A: Everything is in git. Just `git status` and `git restore`.

---

## Next Steps

1. ✅ Read `docs/FINDINGS_SUMMARY.md` (understand what we learned)
2. ✅ Read `experiments/exp001_bc_with_a_ego/README.md` (understand next experiment)
3. ⏳ Decide: Quick test or full retrain?
4. ⏳ Run Experiment 001
5. ⏳ Document results
6. ⏳ Plan Experiment 002

---

## Need Help?

- **Scientific findings**: `docs/FINDINGS_SUMMARY.md`
- **Experiment roadmap**: `docs/EXPERIMENT_PLAN.md`
- **Project structure**: `docs/PROJECT_STRUCTURE.md`
- **Migration guide**: `docs/MIGRATION_GUIDE.md`
- **Baseline results**: `experiments/baseline/README.md`

---

**Remember**: We're doing science! Document everything, test hypotheses systematically, and learn from failures. 🔬

