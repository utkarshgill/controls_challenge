# Project Structure

```
controls_challenge/
│
├── README.md                    # Original competition README
├── requirements.txt             # Dependencies
├── tinyphysics.py              # Core simulator
├── controllers.py              # Base controller interface
│
├── data/                       # Training data (CSV files)
├── models/                     # Saved model checkpoints
│   ├── bc_*.pth
│   └── ppo_*.pth
│
├── controllers/                # Controller implementations for evaluation
│   ├── pid.py
│   ├── bc.py
│   └── ppo_parallel.py
│
├── experiments/                # ALL EXPERIMENTS GO HERE
│   │
│   ├── baseline/               # Experiment 0: Establish baselines
│   │   ├── README.md
│   │   ├── run_baseline.py
│   │   └── results/
│   │       └── baseline_results.json
│   │
│   ├── exp01_bc_baseline/      # Experiment 1: BC without a_ego
│   │   ├── README.md           # What, why, hypothesis
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation script
│   │   ├── config.json         # Hyperparameters
│   │   ├── notes.md            # Running notes
│   │   └── results/            # Outputs
│   │       ├── training_log.txt
│   │       ├── eval_results.json
│   │       └── plots/
│   │
│   ├── exp02_bc_with_a_ego/    # Experiment 2: BC with a_ego
│   │   ├── README.md
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── config.json
│   │   ├── notes.md
│   │   └── results/
│   │
│   ├── exp03_bc_friction_margin/  # Experiment 3: Advanced features
│   │   └── ...
│   │
│   └── template/               # Template for new experiments
│       ├── README.md
│       ├── train.py
│       ├── evaluate.py
│       └── config.json
│
├── analysis/                   # One-off analysis scripts
│   ├── check_training_distribution.py
│   ├── analyze_failures.py
│   └── test_a_ego_hypothesis.py
│
├── docs/                       # Documentation
│   ├── FINDINGS_SUMMARY.md
│   ├── EXPERIMENT_PLAN.md
│   └── LESSONS_LEARNED.md
│
└── archive/                    # Old attempts (keep for reference)
    ├── attempts/
    └── attempts_2/
```

## Experiment Naming Convention

`expNN_short_description/`

- `NN` = zero-padded number (01, 02, 03, ...)
- `short_description` = what's being tested (e.g., `bc_with_a_ego`)

## Each Experiment Folder Contains:

### Required:
- `README.md` - What, why, hypothesis, expected outcome
- `config.json` - All hyperparameters
- `results/` - All outputs (logs, models, plots)

### Optional:
- `train.py` - Training script (if differs from template)
- `evaluate.py` - Evaluation script
- `notes.md` - Running commentary, observations

## Workflow

### Starting a new experiment:
```bash
# 1. Copy template
cp -r experiments/template experiments/exp05_my_idea

# 2. Edit README.md with hypothesis
# 3. Update config.json with hyperparameters
# 4. Run training
# 5. Document results in notes.md
```

### Quick reference:
```bash
# See all experiments
ls experiments/

# Latest results
cat experiments/exp02_bc_with_a_ego/results/eval_results.json

# Compare experiments
python analysis/compare_experiments.py exp01 exp02
```

## Root Directory Rules

**Keep in root:**
- Core files: tinyphysics.py, controllers.py
- Entry points: README.md, requirements.txt
- Directories: data/, models/, experiments/, analysis/, docs/

**Move to experiments/:**
- Training scripts (train_*.py)
- Evaluation scripts (evaluate_*.py, test_*.py)
- Experiment-specific code

**Move to analysis/:**
- One-off diagnostic scripts
- Data exploration notebooks
- Hypothesis testing scripts

**Move to archive/:**
- Old attempts that didn't work
- Deprecated code
- Keep for reference, but out of the way

