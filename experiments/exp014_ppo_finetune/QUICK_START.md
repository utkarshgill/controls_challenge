# Quick Start: Experiment 014

## What is this?

**PPO fine-tuning** of the BC policy from exp013. This should beat both PID and BC by optimizing the actual cost function directly.

## Prerequisites

1. ✅ BC checkpoint from exp013: `../exp013_bc_from_pid/bc_best.pth`
2. ✅ Virtual environment activated: `source ../../.venv/bin/activate`
3. ✅ Data files in: `../../data/*.csv`

## Quick Run

```bash
# From this directory
source ../../.venv/bin/activate
python train_ppo.py
```

**Training time**: ~2-3 hours

## What to Expect

### During Training

You'll see output like:
```
Iter   0 | Cost:   95.23 | Policy Loss: 0.0234 | Value Loss: 0.1234 | Entropy: 0.0123
Iter  10 | Cost:   89.45 | Policy Loss: 0.0198 | Value Loss: 0.0987 | Entropy: 0.0115
  Eval Cost: 87.32
Iter  20 | Cost:   82.67 | Policy Loss: 0.0165 | Value Loss: 0.0765 | Entropy: 0.0108
...
```

**Good signs**:
- Cost is decreasing over time
- Policy loss stabilizes (not exploding)
- Entropy stays > 0.005 (still exploring)

**Bad signs**:
- Cost increases consistently
- Policy loss > 0.1 (gradient explosion)
- Entropy < 0.001 (collapsed to deterministic)

### After Training

Check results:
```bash
python evaluate.py
```

Expected progression:
- **PID**: ~79.6 (baseline)
- **BC**: ~90.5 (starting point)
- **PPO** (early): ~85 (learning)
- **PPO** (converged): **< 70** (target)

## Troubleshooting

### "BC checkpoint not found"
```bash
cd ../exp013_bc_from_pid
python train_bc.py  # Train BC first
```

### Training crashes (segfault)
This might be PyTorch/MPS issue. Try:
```python
# In train_ppo.py, line 28:
device = torch.device('cpu')  # Force CPU instead of MPS
```

### Cost not improving
- Check BC baseline is good (~90)
- Try lower learning rate: `LEARNING_RATE = 1e-4`
- Train longer: `NUM_ITERATIONS = 500`

### Cost exploding
- Lower learning rate: `LEARNING_RATE = 1e-5`
- Reduce clip: `CLIP_EPSILON = 0.1`
- Increase grad clipping: `MAX_GRAD_NORM = 0.3`

## Files Created

After training:
- `ppo_best.pth` - Best checkpoint (use this for evaluation)
- `ppo_final.pth` - Final checkpoint

## Evaluation

### Quick test (100 routes):
```bash
cd ../..
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller ppo_exp014 --baseline_controller pid
```

### Full submission (5000 routes):
```bash
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 \
  --test_controller ppo_exp014 --baseline_controller pid
```

## Understanding the Output

### During Training

**Cost**: Episode total cost (lower is better)
- Starts at ~90 (BC level)
- Should decrease to ~70 or below

**Policy Loss**: PPO clipped objective
- Should stabilize around 0.01-0.03
- If > 0.1, reduce learning rate

**Value Loss**: Critic MSE
- Decreases as critic learns to predict returns
- Should be < 0.1 after convergence

**Entropy**: Policy exploration
- Starts at ~0.01-0.02
- Gradually decreases (policy becomes more certain)
- If < 0.001, increase `ENTROPY_COEF`

## Theory

### Why PPO > BC?

**BC learns**: "What would PID do?"
**PPO learns**: "What minimizes cost?"

PID is reactive (error-based), but the optimal controller is anticipatory (future-aware). PPO can discover this.

### Key Innovation

We have 50 future curvatures in the state. PID-style BC uses them for consistency but is fundamentally reactive. PPO can learn to actually use them predictively.

Example:
```
Curvatures: [0, 0, 0.01, 0.02, 0.03, ...]  # Sharp turn ahead

PID/BC: React to current error
PPO: Start turning early to minimize jerk
```

## Next Steps

If cost < 70: 🎉 **Success!**
If cost 70-80: Good progress, tune hyperparameters
If cost > 80: Debug (see troubleshooting)

**Target**: < 50 for competitive submission
**Stretch**: < 30 for leaderboard top

## Questions?

See `README.md` for full details and `train_ppo.py` for implementation.



