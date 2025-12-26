# Experiment 005: PPO with Data Augmentation

**Status**: 🏃 Training  
**Date**: 2024-12-26

---

## Problem from exp004

PPO learned asymmetric responses:
```
+1.0 error → +0.96 action  ✅ Strong response
-1.0 error → -0.14 action  ❌ Weak response (7× weaker!)

Result: 7.73× worse than PID on average
```

**Root cause**: Training data has slight directional bias (+0.05 mean action)

---

## Solution: Data Augmentation

Train on **original + horizontally flipped** data with 50% probability:

### Flipping Logic:
```python
If flip == True:
  error = -error
  current_lataccel = -current_lataccel  
  target_lataccel = -target_lataccel (implicit in error)
  future_lataccel = -future_lataccel
  roll_lataccel = -roll_lataccel
  action = -action (when applying)

Keep unchanged:
  v_ego (speed is scalar)
  a_ego (longitudinal accel)
```

This forces the network to learn **symmetric responses**.

---

## Expected Improvement

Without augmentation (exp004):
```
File 00000:  2.72× worse (correlation 0.94, but stuck on others)
File 00001: 21.39× worse (stuck at -0.11)
Average:     7.73× worse
```

With augmentation (exp005):
```
Expected: Symmetric responses
  +1.0 error → ~+1.0 action
  -1.0 error → ~-1.0 action

Target: < 2× worse than PID (ideally match or beat)
```

---

## Hyperparameters

Same as exp004 (proven architecture):
```
lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
entropy_coef = 0.001
K_epochs = 10
batch_size = 10,000
num_envs = 8
steps_per_epoch = 10,000
max_epochs = 100
```

---

## Run

```bash
cd experiments/exp005_ppo_augmented
source ../../.venv/bin/activate
python train.py
```

## Evaluation

```bash
cd /Users/engelbart/Desktop/stuff/controls_challenge
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller ppo_aug
```

---

## Results

*TBD after training*

---

## Notes

- Data augmentation is standard in RL for symmetric problems
- 50% flip probability doubles effective dataset size
- Should fix the 7× asymmetry issue
- If this works, we're on path to beating PID

