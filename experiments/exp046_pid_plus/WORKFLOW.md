# CORRECT EVALUATION WORKFLOW

## DO NOT FUCK THIS UP

### Step 1: Develop
Work in `exp046_pid_plus/controller.py`
- Iterate quickly
- Single trajectory tests are for DEBUGGING only
- DO NOT trust single-trajectory scores

### Step 2: Deploy
When ready to evaluate:
```bash
cp controller.py ../../controllers/exp046_controller.py
```

### Step 3: Evaluate (THE ONLY WAY)
```bash
cd ../..
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 --controller exp046_controller
```

### Step 4: Record
Document the batch average score.
That is the ONLY number that matters.

## Baseline
```
PID (100 routes):
  lat_cost:   1.272
  jerk_cost:  21.27
  total:      84.85  ← BEAT THIS
```
