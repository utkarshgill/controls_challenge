# Training Notes

## Quick Start

```bash
# Test everything works
python test_arch.py

# Train (will take 1-2 hours)
python train.py

# Or use the script
./run.sh

# Evaluate after training
python evaluate.py
```

## What to Watch During Training

**eval_cost:** Should start around 75-80 (PID baseline)
- If it drops below 75: **FF is helping!** 🎯
- If it stays at 75-80: FF not being used (investigate)
- If it goes above 80: Something wrong (check architecture)

**σ (exploration std):** Controls how much PPO explores
- Starts high (more exploration)
- Decreases as policy improves
- Should converge to 0.1-0.3

**Time per epoch:** ~30-60 seconds typical
- rollout: CPU simulation time
- update: GPU training time

## Architecture Summary

```
Input: Future Plan [4 × 50]
  ├─ lataccel[50]  - target trajectory
  ├─ v_ego[50]     - speed profile
  ├─ a_ego[50]     - acceleration profile
  └─ roll_lataccel[50] - road banking

     ↓ 1D Conv Tower
     
FF Network Output: [1]
  - Anticipatory steering

     +
     
PID Controller (fixed):
  - Kp=0.195, Ki=0.100, Kd=-0.053
  - Reactive correction

     =
     
Total Action: [-2, 2]
```

## Key Design Decisions

1. **Why fix PID?**
   - Proven stable baseline
   - Smaller learning problem
   - Safe exploration (FF=0 → pure PID)

2. **Why 1D conv?**
   - Temporal patterns in future trajectory
   - Translation invariant (same pattern, different times)
   - Hierarchical (local → global)

3. **Why tanh output?**
   - Bounded action space
   - Smooth gradients
   - Matches LunarLander success

4. **Why no BC init?**
   - BC learns teacher's policy (PID doesn't use future)
   - Cold start lets PPO discover preview from scratch
   - Avoid local optima from imitation

## Debugging Checklist

If training isn't working:

- [ ] Architecture test passes: `python test_arch.py`
- [ ] Data exists: `ls ../../data/*.csv | wc -l` (should be ~20k)
- [ ] Model exists: `ls ../../models/tinyphysics.onnx`
- [ ] Initial cost ≈ 75-80 (PID baseline)
- [ ] FF actions non-zero after a few epochs
- [ ] Gradient flow (no NaN, no explosion)
- [ ] Eval cost changes (not stuck)

## Expected Timeline

- **Epochs 0-20:** Exploration, cost ~75-80
- **Epochs 20-50:** Learning preview, cost drops to 70-75
- **Epochs 50-100:** Fine-tuning, cost 65-70
- **Epochs 100+:** Convergence, best cost achieved

If cost doesn't drop by epoch 50, something's wrong.

## Hyperparameter Tuning

If needed, adjust in `train.py`:

**Exploration:**
- `entropy_coef`: Higher = more exploration (try 0.02)
- Initial `log_std`: Higher = more random (try init to 0.5)

**Learning rate:**
- `pi_lr`: Actor learning rate (try 1e-4 or 5e-4)
- `vf_lr`: Critic learning rate (usually 2-3x actor)

**Batch size:**
- Larger = more stable, slower (try 2048 or 8192)
- Smaller = faster, less stable

**Parallelism:**
- `NUM_ENVS`: More = faster, more memory (try 24 or 32)

## Success Criteria

**Minimal success:** eval_cost < 75 (beat PID)
**Good success:** eval_cost < 60 (20% improvement)
**Great success:** eval_cost < 45 (winner level)

## What We'll Learn

If this works, we'll have proof that:
1. Preview helps (FF with future_plan beats PID)
2. PPO can learn controls (without BC bootstrap)
3. Hybrid architecture is effective (FF+FB structure)

If this doesn't work, we'll learn:
1. Where the bottleneck is (architecture? training? reward?)
2. What to try next (BC? offline RL? hand-engineering?)
3. More about the problem structure

Either way: progress! 🚀

