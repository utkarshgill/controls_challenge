# Experiment 013: Behavioral Cloning from PID (v3 - Fixed)

## Critical Fixes

### 1. Data Collection Bug (Your Discovery)
**Problem**: CSV files contain `steerCommand` for steps 0-99 from the **original logged trajectory**, NOT from PID. The simulator only starts using the controller at step 100 (`CONTROL_START_IDX`).

**Evidence from `tinyphysics.py:146-148`**:
```python
def control_step(self, step_idx: int) -> None:
    action = self.controller.update(...)
    if step_idx < CONTROL_START_IDX:  # CONTROL_START_IDX = 100
        action = self.data['steer_command'].values[step_idx]  # Uses logged data!
```

**Solution**: `collect_data.py` now explicitly skips steps < 100 and only collects (state, action) pairs where action is actually from PID.

### 2. Architecture Improvement (Your Suggestion)
Added **Tanh activation** on network output:
- Network: `Linear → Tanh → Scale by 2.0`
- Naturally bounds actions to `[-2, 2]` (valid steer range)
- More stable than unbounded linear output

### 3. Evaluation Process (Official Challenge Compliance)
**Problem**: Previous evaluation used custom loop, not matching official challenge process.

**Solution**: 
- Created `controllers/bc_exp013.py` with standard `Controller` interface
- Evaluation now uses official `tinyphysics.py` script
- Matches exact process described in challenge README

## State Representation (55D)

```
State:
  - error (target - current lataccel)
  - error_integral (cumulative error for I term)
  - v_ego (vehicle speed)
  - a_ego (vehicle acceleration)
  - roll_lataccel (lateral accel due to road roll)
  - curvatures[50] (future road curvature = (lat_accel - roll) / v²)
```

## Network Architecture

```
Input: 55D state (normalized)
Trunk: 3 layers × 128 hidden units (Tanh activation)
Head: Linear → Tanh → Scale by 2.0
Output: μ ∈ [-2, 2], learnable log_std
Loss: Negative log-likelihood
```

## Training Process

```bash
# 1. Collect clean PID data (only from step >= 100)
python collect_data.py
# Output: pid_trajectories_v3.pkl (~2.7M samples from 5000 routes)

# 2. Train BC network
python train_bc.py
# Output: bc_best.pth

# 3. Evaluate using OFFICIAL process
python evaluate.py
# Uses: tinyphysics.py --controller bc_exp013
```

## Evaluation (Official Process)

See [`EVALUATION.md`](./EVALUATION.md) for detailed evaluation instructions.

**Quick test (100 routes)**:
```bash
cd ../..
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 --controller bc_exp013
```

**Full comparison with report**:
```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller bc_exp013 --baseline_controller pid
```

**Official submission (5000 routes)**:
```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 \
  --test_controller bc_exp013 --baseline_controller pid
```

## Expected Results

With clean data collection and proper evaluation:
- BC should match or slightly beat PID (~99 cost)
- Tanh output provides better stability
- Can be used as initialization for PPO fine-tuning

## Files

- `collect_data.py`: PID data collection (filters step >= 100)
- `train_bc.py`: BC training with tanh output
- `evaluate.py`: Official evaluation wrapper
- `controllers/bc_exp013.py`: Controller implementation
- `EVALUATION.md`: Detailed evaluation guide
- `bc_best.pth`: Trained model checkpoint

## Next Steps

After BC is working:
1. Verify BC matches PID performance
2. Add PPO scaffolding from `beautiful_lander.py`
3. Fine-tune BC with PPO for better performance
4. Target: total_cost < 45 (competitive submission)
