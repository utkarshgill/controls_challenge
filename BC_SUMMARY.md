# Behavioral Cloning Summary

## Architecture
```
State: 57D vector
├── PID terms (3): error, error_diff, error_integral
├── Current state (4): current_lataccel, v_ego, a_ego, curv_now
└── Future plan (50): 50 curvature values

Network: MLP
├── Trunk: Linear(57, 128) + ReLU
├── Actor head: 3x [Linear(128, 128) + ReLU]
└── Output: Linear(128, 1) → tanh → scale to [-2, 2]
```

## Critical Insights

### 1. State Must Match PID Exactly
```python
# PID computes:
error_diff = error - prev_error        # NO /dt
error_integral += error                # NO *dt
# No clipping on integral

# We must match this exactly, or BC learns wrong mapping
```

### 2. Dataset Must Be Shuffled
- Files are sorted by difficulty (easy → hard)
- Without shuffle: train on easy, validate on hard → huge gap
- With shuffle: both sets have mixed difficulty → good generalization

### 3. Feature Scaling Matters
```python
OBS_SCALE = [10, 1, 1, 2, 0.03, 20, 1000] + [1000]*50
# error: ÷10, error_diff: ÷1, error_integral: ÷1, ...
```

## Results

**Typical performance:**
- PID baseline: 100-150 (varies by file sampling)
- BC train: 80-100
- BC val: 100-130
- MSE loss: ~0.003-0.005

**What this means:**
- ✅ Network can learn the control task
- ✅ Architecture is sound
- ✅ Ready for PPO fine-tuning

## Next Steps: PPO Fine-Tuning

BC provides:
1. **Warm start** (~100 cost baseline vs random ~10,000)
2. **Validated architecture** (network CAN learn steering)
3. **Saved checkpoint** (bc_pid_checkpoint.pth)

PPO will:
1. Load BC weights as initialization
2. Optimize for actual cost function (not just MSE to PID)
3. Learn to handle autoregressive compounding errors
4. Target: <45 cost (competition winner achieved this)

## Key Files

- `train_bc_pid.py`: BC training script
- `bc_pid_best.pth`: Trained weights
- `bc_pid_checkpoint.pth`: Full config + weights + performance metrics
- `controllers/pid.py`: Reference PID implementation

## Lessons Learned

1. **Feature engineering matters**: PID terms (error, derivative, integral) give network temporal context
2. **State matching is critical**: Train/test mismatch kills performance
3. **Data diversity is essential**: Shuffle before split, use 1000+ files
4. **Multiprocessing helps**: 8-10× speedup with pool initialization
5. **Relative thresholds work better**: Compare to PID baseline, not absolute values

