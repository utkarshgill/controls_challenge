# Project Structure - Research Lab Style

## Philosophy
- **Clean root**: Only essential files visible
- **Experiments isolated**: Each experiment is self-contained
- **Reproducible**: Config + code + results in one place
- **Scalable**: Easy to add new experiments without clutter

---

## Directory Structure

```
controls_challenge/
│
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup (optional)
│
├── data/                        # Raw data (read-only)
│   └── *.csv
│
├── models/                      # Pretrained/reference models
│   └── tinyphysics.onnx
│
├── src/                         # Core library code
│   ├── __init__.py
│   ├── tinyphysics.py          # Simulator
│   ├── controllers/            # Controller implementations
│   │   ├── __init__.py
│   │   ├── pid.py
│   │   ├── bc_pid.py
│   │   └── ppo_parallel.py
│   ├── networks/               # Neural network architectures
│   │   ├── __init__.py
│   │   ├── actor_critic.py
│   │   └── bc_network.py
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── bc_trainer.py
│   │   └── ppo_trainer.py
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── state_builder.py
│       └── evaluation.py
│
├── experiments/                # All experiments go here
│   │
│   ├── baseline/               # Experiment 0: Establish baselines
│   │   ├── README.md           # What/why/results
│   │   ├── config.yaml         # Hyperparameters
│   │   ├── run.py              # Entry point
│   │   ├── results/            # Outputs
│   │   │   ├── metrics.json
│   │   │   ├── plots/
│   │   │   └── checkpoints/
│   │   └── notes.md            # Observations, insights
│   │
│   ├── exp001_bc_with_a_ego/   # Experiment 1: Add a_ego to BC
│   │   ├── README.md
│   │   ├── config.yaml
│   │   ├── run.py
│   │   ├── results/
│   │   └── notes.md
│   │
│   ├── exp002_friction_margin/ # Experiment 2: Explicit friction
│   │   ├── README.md
│   │   ├── config.yaml
│   │   ├── run.py
│   │   ├── results/
│   │   └── notes.md
│   │
│   └── template/               # Template for new experiments
│       ├── README.md
│       ├── config.yaml
│       ├── run.py
│       └── notes.md
│
├── scripts/                    # One-off analysis scripts
│   ├── analyze_failures.py
│   ├── test_a_ego_hypothesis.py
│   └── visualize_trajectories.py
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
│
├── docs/                       # Documentation
│   ├── FINDINGS_SUMMARY.md
│   ├── EXPERIMENT_PLAN.md
│   └── architecture.md
│
└── archive/                    # Old/deprecated code
    └── attempts_1/
```

---

## Experiment Structure (Template)

Each experiment follows this structure:

```
experiments/expXXX_name/
│
├── README.md                   # Experiment card
│   ├── Hypothesis
│   ├── Method
│   ├── Results
│   └── Conclusion
│
├── config.yaml                 # All hyperparameters
│   ├── model: {...}
│   ├── training: {...}
│   └── evaluation: {...}
│
├── run.py                      # Single entry point
│   └── python run.py --config config.yaml
│
├── results/                    # All outputs
│   ├── metrics.json            # Quantitative results
│   ├── checkpoints/            # Model weights
│   │   ├── best.pt
│   │   └── final.pt
│   ├── plots/                  # Visualizations
│   │   ├── training_curve.png
│   │   └── failure_analysis.png
│   └── logs/                   # Training logs
│       └── train.log
│
└── notes.md                    # Lab notebook
    ├── 2024-12-25: Initial run
    ├── 2024-12-26: Tuned LR
    └── Observations: ...
```

---

## Experiment README Template

```markdown
# Experiment XXX: [Name]

**Date**: YYYY-MM-DD  
**Status**: 🏃 Running | ✅ Complete | ❌ Failed  
**Researcher**: [Your name]

## Hypothesis
What are we testing?

## Motivation
Why is this worth trying?

## Method
- Model: [architecture]
- State: [features]
- Training: [dataset, epochs, etc.]

## Results
| Metric | Value |
|--------|-------|
| Mean cost | XX.X |
| Median cost | XX.X |
| Best file | XX.X |
| Worst file | XX.X |

## Comparison to Baseline
- Baseline: 80.4
- This: XX.X
- Improvement: ±X.X%

## Conclusion
Did it work? Why/why not?

## Next Steps
What to try next based on these results?
```

---

## Usage

### Starting a new experiment:
```bash
# 1. Copy template
cp -r experiments/template experiments/exp003_my_idea

# 2. Edit README.md with hypothesis
vim experiments/exp003_my_idea/README.md

# 3. Edit config.yaml with hyperparameters
vim experiments/exp003_my_idea/config.yaml

# 4. Run experiment
cd experiments/exp003_my_idea
python run.py

# 5. Document results in notes.md
vim notes.md
```

### Comparing experiments:
```bash
# All results in one place
ls experiments/*/results/metrics.json

# Generate comparison table
python scripts/compare_experiments.py
```

---

## Benefits

1. **Clean root**: Only 6 top-level folders
2. **Self-contained**: Each experiment has everything it needs
3. **Reproducible**: Config + code + results together
4. **Scalable**: Add experiments without cluttering root
5. **Collaborative**: Easy to share/review specific experiments
6. **Historical**: Old experiments stay organized, not deleted

---

## Migration Plan

Move existing files:
```bash
# Core code → src/
mv tinyphysics.py src/
mv controllers/ src/

# Experiments → experiments/
mkdir -p experiments/baseline
mv baseline.py experiments/baseline/run.py
mv final_evaluation.py experiments/baseline/

# Analysis → scripts/
mv analyze_failures.py scripts/
mv test_a_ego_hypothesis.py scripts/

# Docs → docs/
mv FINDINGS_SUMMARY.md docs/
mv EXPERIMENT_PLAN.md docs/

# Old stuff → archive/
mv attempts_2/ archive/
```

---

## Example: Current Baseline Experiment

```
experiments/baseline/
├── README.md
│   Hypothesis: Establish PID/BC/PPO baselines
│   Results: PID=80.4, BC=92.4, PPO=93.8
│   Conclusion: BC/PPO fail on 4% of files (high a_ego)
│
├── config.yaml
│   state_dim: 56
│   features: [error, error_diff, error_integral, ...]
│   anti_windup: [-14, 14]
│
├── run.py
│   # Runs PID, BC, PPO on 100 files
│
├── results/
│   ├── metrics.json
│   │   {"pid": 80.4, "bc": 92.4, "ppo": 93.8}
│   ├── final_results.npz
│   └── plots/
│       └── cost_distribution.png
│
└── notes.md
    2024-12-25: Discovered a_ego hypothesis
    Key finding: File 00069 has 5× more |a_ego|
```

---

## Next: Experiment 001

```
experiments/exp001_bc_with_a_ego/
├── README.md
│   Hypothesis: Adding a_ego fixes friction circle coupling
│   Expected: 92.4 → 85
│
├── config.yaml
│   state_dim: 57  # +1 for a_ego
│   features: [..., a_ego, ...]
│   obs_scale: [..., 20.0, ...]
│
└── run.py
    # Train BC with a_ego, evaluate on 100 files
```

