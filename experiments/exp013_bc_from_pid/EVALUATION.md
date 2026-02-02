# Exp013 Evaluation Process

## Official Challenge Evaluation

The challenge README specifies the exact evaluation process:

### 1. Batch Metrics (Quick Test)
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 --controller bc_exp013
```

### 2. Official Comparison (with Report)
```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller bc_exp013 --baseline_controller pid
```
This generates `report.html` with visualizations.

### 3. Official Submission (5000 routes)
```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 \
  --test_controller bc_exp013 --baseline_controller pid
```
Submit the `report.html` and code to the official form.

## Our Implementation

### Controller Location
- **File**: `controllers/bc_exp013.py`
- **Class**: `Controller` (standard interface)
- **Checkpoint**: `experiments/exp013_bc_from_pid/bc_best.pth`

### Evaluation Script
- **File**: `experiments/exp013_bc_from_pid/evaluate.py`
- **What it does**: Calls the official `tinyphysics.py` script
- **Why**: Ensures exact match with challenge evaluation

## Key Differences from Previous Approach

### ❌ Old Approach (WRONG)
```python
# Custom evaluation loop
sim = TinyPhysicsSimulator(...)
for _ in range(len(sim.data) - 50):
    sim.step()
cost = sim.compute_cost()
```

### ✅ New Approach (CORRECT)
```bash
# Use official script
python tinyphysics.py --controller bc_exp013 --data_path ./data --num_segs 100
```

This ensures:
1. ✓ Uses `sim.rollout()` (official method)
2. ✓ Uses `sim.compute_cost()` with correct indices
3. ✓ Matches exact evaluation process
4. ✓ Controller can be officially submitted

## Verification Checklist

- [x] Controller in `controllers/` directory
- [x] Controller has `Controller` class
- [x] Controller has `update()` method matching signature
- [x] Evaluation uses official `tinyphysics.py`
- [x] Cost calculation matches official process
- [ ] Trained BC model exists (`bc_best.pth`)
- [ ] Run quick test (100 routes)
- [ ] Run full evaluation (5000 routes for submission)

## Running the Evaluation

```bash
# From experiment directory
cd experiments/exp013_bc_from_pid

# 1. Train the model first (if not done)
python train_bc.py

# 2. Run evaluation (uses official scripts)
python evaluate.py

# 3. Or run official commands directly from root:
cd ../..
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 --controller bc_exp013

python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller bc_exp013 --baseline_controller pid
```





