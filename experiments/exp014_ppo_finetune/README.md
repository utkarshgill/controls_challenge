# Experiment 014: PPO Fine-Tuning from BC

## Overview

This experiment fine-tunes the BC policy from exp013 using PPO to optimize the actual cost function directly.

**Key Innovation**: Instead of imitating PID, we let PPO discover better-than-PID strategies by optimizing the true objective.

## Architecture

### Actor-Critic Network

```
Actor (initialized from BC):
  Input: 55D state (normalized)
  Trunk: 3 × (Linear(128) → Tanh)      [from BC]
  Actor Head: Linear(1) → Tanh → ×2.0  [from BC]
  Output: μ ∈ [-2, 2], learnable log_std

Critic (new):
  Critic Head: Linear(128) → Tanh → Linear(1)
  Output: Value estimate V(s)
```

### Why This Works

1. **BC provides strong initialization** - Already learned reactive control (~90 cost)
2. **PPO refines the policy** - Optimizes actual cost, not imitation
3. **Anticipatory control** - Can learn to use future information better than PID
4. **Smooth actions** - Learns to minimize jerk naturally through cost gradient

## State Representation (55D)

Same as BC (exp013):
```python
state = [
  error,              # Target - current lataccel
  error_integral,     # PID I-term
  v_ego,              # Current speed
  a_ego,              # Longitudinal accel
  roll_lataccel,      # Road bank angle effect
  curvatures[50]      # Future road geometry
]
```

## Reward Function

Matches official cost computation exactly:
```python
lat_cost = (target_lataccel - current_lataccel)^2 * 100
jerk_cost = ((current - prev) / 0.1)^2 * 100
total_cost = 50 * lat_cost + jerk_cost

reward = -total_cost / 100.0  # Negative cost (normalized)
```

## Hyperparameters

### PPO
- **Learning rate**: 3e-4 (lower for fine-tuning)
- **Clip epsilon**: 0.2
- **GAE lambda**: 0.95
- **Gamma**: 0.99
- **Entropy coef**: 0.01
- **Value loss coef**: 0.5

### Training
- **Parallel envs**: 8
- **Steps per iteration**: 20,000
- **Batch size**: 2,048
- **PPO epochs**: 10
- **Total iterations**: 200

## Expected Performance

| Method | Cost | Notes |
|--------|------|-------|
| PID (baseline) | 79.62 | Reactive control |
| BC (exp013) | 90.49 | Imitates PID |
| **PPO (exp014)** | **< 70?** | Optimizes directly |
| Competitive | < 50 | Challenge leaderboard |
| Winning | < 30 | Top submissions |

## How to Run

### 1. Train PPO (from BC initialization)

```bash
cd experiments/exp014_ppo_finetune
python train_ppo.py
```

**Expected training time**: 
- ~2-3 hours on MPS (M-series Mac)
- ~1-2 hours on CUDA (GPU)

**Output**:
- `ppo_best.pth` - Best checkpoint (lowest cost)
- `ppo_final.pth` - Final checkpoint

### 2. Evaluate (quick test)

```bash
python evaluate.py
```

This runs 100 routes comparing:
- PID (baseline)
- BC (exp013)
- PPO (exp014)

### 3. Official Evaluation

```bash
cd ../..
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller ppo_exp014 --baseline_controller pid
```

Generates `report.html` with detailed comparison.

### 4. Full Submission (5000 routes)

```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 \
  --test_controller ppo_exp014 --baseline_controller pid
```

## Training Details

### Initialization
- **Actor**: Load from BC (exp013) - already knows reactive control
- **Critic**: Random initialization - learns value function from scratch

### Data Collection
- **Parallel envs**: 8 simultaneous rollouts
- **Routes**: Randomly sampled from 15,000 training files
- **Resampling**: New routes every 10 iterations (diversity)

### PPO Updates
- **GAE advantages**: Temporal difference with λ=0.95
- **Clipped objective**: Prevents large policy updates
- **Value learning**: Critic learns to predict returns
- **Entropy bonus**: Encourages exploration

### Gradient Clipping
- **Max norm**: 0.5 (prevents instability)

## Why PPO > BC

| Aspect | BC | PPO |
|--------|----|----|
| **Objective** | Imitate PID | Minimize cost |
| **Strategy** | Reactive | Anticipatory |
| **Exploration** | None | Through policy gradient |
| **Jerk** | Implicit (from data) | Explicit (in reward) |
| **Ceiling** | ~PID performance | Beyond PID |

## Debugging

### If training is unstable:
1. Lower learning rate: `LEARNING_RATE = 1e-4`
2. Reduce clip epsilon: `CLIP_EPSILON = 0.1`
3. Increase entropy: `ENTROPY_COEF = 0.02`

### If not improving:
1. Train longer: `NUM_ITERATIONS = 500`
2. More parallel envs: `NUM_ENVS = 16`
3. Larger batches: `BATCH_SIZE = 4096`

### If too smooth/conservative:
1. Check BC initialization is good (should be ~90 cost)
2. Increase exploration: `ENTROPY_COEF = 0.02`
3. Lower value loss weight: `VALUE_LOSS_COEF = 0.3`

## Files

- `train_ppo.py` - Main training script
- `evaluate.py` - Evaluation wrapper
- `README.md` - This file
- `../controllers/ppo_exp014.py` - Controller for official evaluation
- `ppo_best.pth` - Best checkpoint (created by training)
- `ppo_final.pth` - Final checkpoint (created by training)

## Next Steps

After getting PPO working:

1. **Tune hyperparameters** - Learning rate, clip epsilon, entropy
2. **More training** - Run for 500-1000 iterations
3. **Larger network** - Try 256 hidden units
4. **Curvature space** - Try 58D state from exp013
5. **Ensemble** - Combine multiple PPO policies
6. **Model-based** - Add dynamics model for planning

## Shannon's Wisdom Applied

> "We know the past but cannot control it. We control the future but cannot know it."

PPO learns to anticipate the future (using curvatures) and control in the present - exactly what's needed to beat reactive PID control.

The key insight: **BC gives us a safe starting point, PPO lets us explore improvements.**



