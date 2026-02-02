# Experiment 003: Clean BC from PID

**Status**: 🏃 Running  
**Date**: 2024-12-26

---

## Hypothesis

Train a clean BC from PID with carefully designed state representation:
- **State**: 55D = [error, roll_lataccel, v_ego, a_ego, current_lataccel, future_lataccel×50]
- **Key insight**: Include `a_ego` (friction circle) and full future trajectory

Expected: BC should match PID (~85 cost), providing solid baseline for future PPO.

## State Design

```python
State (55D):
├── error (1)             # target - current
├── roll_lataccel (1)     # road effect  
├── v_ego (1)             # velocity
├── a_ego (1)             # longitudinal accel (CRITICAL for friction circle)
├── current_lataccel (1)  # where we are now
└── future_lataccel (50)  # what's coming next 5 seconds
```

**Why this works:**
- `a_ego`: Constraint is √(a_lat² + a_long²) ≤ μg
- `future_lataccel`: Network learns to prepare for upcoming maneuvers
- Simple scalars + trajectory = interpretable and effective

## Network

```
Architecture: 55 → 128 → 128 → 128 → 1
Activation: ReLU + Tanh output
Output: steer in [-2, 2]
```

## Training

```bash
cd experiments/exp003_ppo_bc_init
python train_bc.py
```

- Collect data: 1000 files with PID
- Train: 30 epochs, lr=1e-3, batch=256

## Evaluation

```bash
cd /Users/engelbart/Desktop/stuff/controls_challenge
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller bc
```

## Expected Results

```
Target:    45.0
PID:       84.85
BC:        ~85 (should match PID)
```

## Results

*TBD*

---

## Next Steps

If BC works (~85):
1. Use as initialization for PPO (exp004)
2. PPO can explore beyond BC to reach < 45
