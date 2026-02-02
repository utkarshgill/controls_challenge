# How to Run Experiments

## Philosophy

**Each experiment = Self-contained snapshot**
- Training code lives IN the experiment folder
- Can modify one experiment without breaking others
- Everything needed to reproduce in one place
- One command to run

**Root = Minimal**
- Only simulator (`tinyphysics.py`)
- Shared data (`data/`)
- Controllers (`controllers/`)
- No training scripts in root!

---

## Running an Experiment

```bash
# From project root
cd /path/to/controls_challenge

# Run experiment
bash experiments/exp002_ppo_baseline/run.sh

# Or manually:
cd experiments/exp003_ppo_bc_init
python train.py --init-from ../baseline/results/checkpoints/bc_pid_best.pth
```

---

## Evaluating an Experiment

After training, create a controller to evaluate your experiment:

**1. Create a controller in `controllers/` directory**
   - Example: `controllers/exp003_ppo_bc.py`
   - Load your trained model checkpoint
   - Implement `update()` method to predict actions

**2. Test on single route with debug visualization:**
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller exp003_ppo_bc
```

**3. Get batch metrics on multiple routes:**
```bash
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller exp003_ppo_bc
```

**4. Generate a report comparing to baseline:**
```bash
# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller exp003_ppo_bc --baseline_controller pid
```

**Important**: Each experiment needs a corresponding controller in `controllers/` that loads its trained checkpoint. The controller is what gets evaluated by the simulator, not the training code directly.

---

## Creating New Experiment

```bash
# 1. Copy template
cp -r experiments/template experiments/exp003_my_idea

# 2. Edit README.md
#    - What's the hypothesis?
#    - What changes from previous?

# 3. Edit run.sh
#    - What training command?
#    - Where to save results?

# 4. Run it
bash experiments/exp003_my_idea/run.sh

# 5. Create controller in controllers/ for evaluation
#    - Load your trained checkpoint
#    - Implement update() method

# 6. Get batch metrics
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller exp003_my_idea

# 7. Generate comparison report
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller exp003_my_idea --baseline_controller pid

# 8. Document results in README.md
```

---

## Experiment Structure

```
experiments/exp003_ppo_bc_init/
├── train.py               # Training code (self-contained)
├── eval.py                # Evaluation code (if needed)
├── run.sh                 # One command to reproduce
├── README.md              # Hypothesis, results, analysis
└── results/
    ├── checkpoints/
    │   └── ppo_parallel_best.pth
    └── logs/
        └── training.log
```

---

## What Goes Where

### ✅ In experiment folder:
- `train.py` - Training code (full copy, not import)
- `eval.py` - Evaluation code (if needed)
- `run.sh` - One command to run everything
- `README.md` - Hypothesis, results, analysis
- `results/` - Checkpoints, logs, metrics

### ✅ In root:
- `tinyphysics.py` - Simulator only
- `controllers/` - Reusable controllers (create one per experiment for evaluation)
- `data/` - Shared data
- `experiments/` - All experiments

### ❌ Never in root:
- Training scripts (they go IN each experiment)
- Checkpoints
- Logs
- Result files

---

## Comparing Experiments

```bash
# List all experiments
ls experiments/

# Compare results
python scripts/compare_experiments.py exp002_ppo_baseline exp003_bc_init

# Or manually check READMEs
cat experiments/*/README.md | grep "Best cost"
```

---

## Current Experiments

- `baseline/` - PID/BC/PPO baselines (exp 000)
- `exp001_bc_with_a_ego/` - BC with a_ego (failed - wrong approach)
- `exp002_ppo_baseline/` - PPO with fixed hyperparameters (unstable, cost=497)
- `template/` - Copy this for new experiments

---

## Next Experiment Ideas

- exp003: PPO with BC initialization
- exp004: PPO with a_ego in state (57D)
- exp005: PPO with smaller network
- exp006: PPO with reduced future steps (10 instead of 50)
