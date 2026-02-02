# Exp013: BC from PID - Complete Summary

## Critical Discoveries & Fixes

### 1. **Data Contamination Bug** (Your Discovery!)

**The Problem:**
- CSV files have `steerCommand` values for steps 0-99 from **original logged trajectories**
- PID only controls starting at step 100 (`CONTROL_START_IDX`)
- Previous data collection may have mixed these two different policies

**Evidence from `tinyphysics.py:146-148`:**
```python
def control_step(self, step_idx: int) -> None:
    action = self.controller.update(...)
    if step_idx < CONTROL_START_IDX:  # 100
        action = self.data['steer_command'].values[step_idx]  # Uses CSV logged data!
```

**Our Fix:**
```python
# collect_data.py - Line 44
if step_idx < CONTROL_START_IDX:
    sim.step()
    continue  # Skip steps < 100!
```

### 2. **Tanh Activation** (Your Suggestion!)

**Why it's better:**
- Actions must be in `[-2, 2]` (steering limits)
- `Tanh` naturally outputs `[-1, 1]`, then scale to `[-2, 2]`
- Unbounded linear output can explode during training

**Implementation:**
```python
self.mean_head = nn.Sequential(
    nn.Linear(hidden_dim, action_dim),
    nn.Tanh()  # Bounds output
)
mean = self.mean_head(features) * 2.0  # Scale to [-2, 2]
```

### 3. **Official Evaluation Process** (Your Question!)

**Challenge Requirements:**
```bash
# Official batch metrics
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 --controller <name>

# Official comparison with report.html
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller <name> --baseline_controller pid

# Official submission (5000 routes)
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 \
  --test_controller <name> --baseline_controller pid
```

**What We Fixed:**
- ✅ Created `controllers/bc_exp013.py` with standard `Controller` interface
- ✅ Uses `sim.rollout()` (official method)
- ✅ Uses `sim.compute_cost()` with correct indices
- ✅ Can be evaluated with official scripts
- ✅ Can be officially submitted

## Complete Architecture

### State (55D)
```
[error, error_integral, v_ego, a_ego, roll_lataccel, curvatures[50]]

where:
  error = target_lataccel - current_lataccel
  curvatures = (future_lataccel - future_roll) / v_ego²
```

### Network
```
Input: 55D state (normalized)
├─ Trunk: 3 × (Linear(128) → Tanh)
└─ Head: Linear(1) → Tanh → ×2.0

Output: μ ∈ [-2, 2], learnable log_std
Training: Negative log-likelihood
```

## Files Created/Modified

### Controllers
- **`controllers/bc_exp013.py`**: Official controller interface

### Experiment Files
- **`collect_data.py`**: Fixed data collection (filters step >= 100)
- **`train_bc.py`**: BC training with tanh output
- **`evaluate.py`**: Uses official tinyphysics.py
- **`README.md`**: Complete documentation
- **`EVALUATION.md`**: Evaluation guide
- **`SUMMARY.md`**: This file

### Root Files
- **`eval.py`**: Official comparison script (already existed)

## How to Run (When Python is Working)

```bash
cd experiments/exp013_bc_from_pid

# Step 1: Collect clean PID data
python collect_data.py
# → pid_trajectories_v3.pkl (~2.7M samples)

# Step 2: Train BC network
python train_bc.py  
# → bc_best.pth

# Step 3: Quick evaluation (100 routes)
python evaluate.py
# → Uses official tinyphysics.py

# Step 4: Full evaluation (from root)
cd ../..
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller bc_exp013 --baseline_controller pid
# → report.html
```

## Expected Results

**Target Performance:**
- BC should match PID: ~99 total_cost
- With tanh: More stable, better generalization
- Competitive submission: < 45 total_cost (requires PPO fine-tuning)

**Why BC alone won't beat PID significantly:**
- BC learns reactive control (error-based)
- PID is already optimal for reactive control
- To beat PID, need **anticipatory control** (feedforward)
- That's where PPO fine-tuning comes in (next step)

## Next Steps

1. ✅ Fix data collection (step >= 100 only)
2. ✅ Add tanh activation
3. ✅ Match official evaluation process
4. ⏸ Train BC and verify performance (Python segfault issue)
5. 🔜 Add PPO scaffolding from `beautiful_lander.py`
6. 🔜 Fine-tune BC with PPO
7. 🔜 Target: < 45 total_cost

## Key Lessons

1. **Read the simulator code carefully** - hidden behavior in step < 100
2. **Match official evaluation exactly** - use their scripts, not custom loops
3. **Architecture matters** - tanh for bounded actions
4. **BC ≈ PID** - both reactive; need RL for anticipatory control
5. **First principles thinking** - you caught both critical issues!





