# Project Structure

## Root Directory (Original Challenge Files)
```
controls_challenge/
├── README.md              # Original challenge README
├── requirements.txt       # Python dependencies
├── tinyphysics.py        # Core simulator
├── eval.py               # Evaluation script
├── controllers/          # Controller implementations
├── models/               # ONNX model files
├── data/                 # CSV route data
└── imgs/                 # Images for documentation
```

## Our Additions

### `/analysis/` - Research & Analysis Scripts
All analysis, research notes, and investigative scripts:
- `BLACKBOARD.md` - Research brainstorming
- `PHYSICS_ANALYSIS.md` - Physics-based analysis
- `SPINNING_UP_LESSONS.md` - PPO insights from OpenAI Spinning Up
- `beautiful_lander.py` - Reference PPO implementation
- `verify_baseline.py` - Baseline verification script
- Other analysis scripts

### `/experiments/` - Experiment Implementations
Organized by experiment number (exp001, exp002, etc.):
- `exp017_baseline/` - 1-neuron PID clone (baseline)
- `exp023_conv/` - Conv1D on curvatures
- `exp030_vehicle_state/` - Vehicle-centric state representation
- Each experiment has its own folder with:
  - `train.py` - Training script
  - `controller.py` - Controller for evaluation
  - `model.pth` - Trained model
  - `README.md` - Experiment notes

### `/archive/` - Old Attempts & Historical Code
Archived code from early attempts, kept for reference

### `/docs/` - Documentation
General project documentation

## Current Best Results (Official Eval)
```
Baseline:
- PID: ~107 total_cost (1000 segs)
- exp023 (Conv BC): ~103 total_cost  ✅ Best BC
- exp030 (Vehicle state): ~112 total_cost

Target: <45 (56% improvement needed via PPO)
```

## Key Files

### Controllers (in `/controllers/`)
- `pid.py` - Baseline PID controller
- `exp023_conv.py` - Best BC controller (Conv1D)
- `exp030_vehicle.py` - Vehicle-centric BC

### Experiments (in `/experiments/`)
- `exp017_baseline/` - Proof that MLP can clone PID
- `exp023_conv/` - Conv1D breakthrough (12% gain over naive BC)
- `exp030_vehicle_state/` - Vehicle-centric representation

## Running Experiments

### Official Evaluation
```bash
# Test a controller
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                     --data_path ./data \
                     --num_segs 1000 \
                     --controller exp023_conv

# Compare against baseline
python eval.py --model_path ./models/tinyphysics.onnx \
              --data_path ./data \
              --num_segs 1000 \
              --test_controller exp023_conv \
              --baseline_controller pid
```

### Training New Experiments
```bash
# Create new experiment folder
mkdir -p experiments/exp031_my_experiment

# Train
python experiments/exp031_my_experiment/train.py

# Test
python tinyphysics.py --controller exp031_my_experiment
```

## Notes
- Keep root clean - match original challenge structure
- All research/analysis goes in `/analysis/`
- All experiments go in `/experiments/expXXX_name/`
- Archive old code in `/archive/`



