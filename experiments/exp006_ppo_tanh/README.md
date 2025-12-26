# Experiment 006: PPO with Tanh Activations

**Status**: 🏃 Training  
**Date**: 2024-12-26

---

## Problem from exp004 & exp005

PPO learned asymmetric responses even with data augmentation:

```
exp004 (ReLU, no aug):
  +1.0 error → +0.96 action  ✅
  -1.0 error → -0.14 action  ❌ (7× weaker!)

exp005 (ReLU, with aug):
  +1.0 error → +0.69 action  
  -1.0 error → -0.15 action  ❌ (still 4.6× weaker!)
```

**Root cause**: ReLU is asymmetric!
```python
ReLU(x) = max(0, x)  # Only passes positive values
                     # Zeros out negative values
```

---

## Solution: Symmetric Activation

Replace ReLU with **Tanh**:

```python
# OLD (asymmetric):
trunk = [Linear, ReLU, Linear, ReLU, ...]

# NEW (symmetric):
trunk = [Linear, Tanh, Linear, Tanh, ...]

where Tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

**Why Tanh is symmetric:**
```
Tanh(+1) ≈ +0.76
Tanh(-1) ≈ -0.76  ← Perfect symmetry!
```

**Why ReLU breaks symmetry:**
```
If Linear maps:
  +error → +features → ReLU passes → network learns ✅
  -error → -features → ReLU zeros → network can't learn ❌
```

---

## Architecture Changes

```python
ActorCritic:
  Trunk:  state → Linear → Tanh → Linear → Tanh
  Actor:  trunk → Linear → Tanh × 3 → Linear → action_mean
  Critic: trunk → Linear → Tanh × 3 → Linear → value
```

**Everything else unchanged** (same as exp005):
- Data augmentation: 50% flip
- State: 55D normalized
- Hyperparameters: proven from beautiful_lander.py

---

## Expected Results

```
PID:     84.85
exp004: 407.9  (ReLU, no aug, 7× asymmetry)
exp005: 387.9  (ReLU, aug, 4.6× asymmetry)
exp006:  ???   (Tanh, aug, symmetric!)
```

**If Tanh fixes it:**
- Symmetric responses: +1.0 → +X, -1.0 → -X
- Better performance (closer to PID)
- Path to < 45 target

---

## Run

```bash
cd experiments/exp006_ppo_tanh
source ../../.venv/bin/activate
python train.py
```

## Evaluation

```bash
cd /Users/engelbart/Desktop/stuff/controls_challenge
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller ppo_tanh
```

---

## Notes

- Tanh is standard for symmetric control problems
- Slower to train than ReLU (saturates at ±1)
- But correctness > speed
- If this doesn't fix it, problem is elsewhere (state representation, reward, etc.)

