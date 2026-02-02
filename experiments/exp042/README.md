# Experiment 042: Behavioral Cloning Baseline

Clean implementation of behavioral cloning to imitate PID controller.

## Architecture

**MLP**: `55 -> 64 -> 32 -> 1` with tanh activations

**Input features (55 total)**:
- 5 base features:
  - `error`: target_lataccel - current_lataccel
  - `error_integral`: cumulative error
  - `error_diff`: error - prev_error
  - `current_curvature`: (current_lataccel - roll_lataccel) / v_ego²
  - `target_curvature`: (target_lataccel - roll_lataccel) / v_ego²

- 50 future curvatures:
  - `future_curvature[t+1]` through `future_curvature[t+50]`
  - Computed from future_plan (next 5 seconds at 10 Hz)

**Output**: Steering command (single float)

## Files

- `train_bc.py`: Training script for behavioral cloning
- `bc_controller.py`: Controller class using trained MLP
- `eval_bc.py`: Evaluation script comparing BC to PID
- `run.sh`: Helper script to train and evaluate

## Usage

### 1. Train BC model

```bash
cd experiments/exp042

python train_bc.py \
    --model_path ../../models/tinyphysics.onnx \
    --data_dir ../../data \
    --output_dir ./outputs \
    --num_train 800 \
    --num_val 100 \
    --num_test 100 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001
```

This will:
- Collect data from 800 training CSV files by running PID controller
- Train MLP for 50 epochs
- Save best model to `./outputs/best_model.pt`
- Save final model to `./outputs/final_model.pt`
- Save training history to `./outputs/training_history.pt`

### 2. Evaluate BC model

```bash
python eval_bc.py \
    --model_path ../../models/tinyphysics.onnx \
    --data_dir ../../data \
    --bc_checkpoint ./outputs/best_model.pt \
    --num_files 100
```

This will:
- Test BC controller on 100 CSV files
- Compare to PID baseline
- Print cost statistics

### 3. Quick run (train + eval)

```bash
bash run.sh
```

## Expected Results

Based on prior experiments:
- **PID baseline**: ~140 total cost
- **BC model**: ~250 total cost (~1.8x worse)
- BC should be within 2x of PID to be considered successful

## Next Steps

After BC baseline is working:
1. Add PPO fine-tuning (incrementally)
2. Start with ActorCritic architecture
3. Add rollout collection
4. Add PPO update mechanism
5. Train with RL to improve beyond BC

## Notes

- BC imitates PID, so it can't outperform PID directly
- BC is useful as a warm-start for RL fine-tuning
- The 55-feature input (especially future curvatures) allows anticipatory control
