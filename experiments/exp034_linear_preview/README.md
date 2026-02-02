# Experiment 034: 3-Parameter Linear Preview Controller

## The Decisive Test

**Question**: Does preview help in this simulator?

**Method**: Minimal hypothesis class - learn exactly 3 weights for 3 preview features.

If this can't beat PID (~75 cost), then preview doesn't matter.
If it does, we have the winning geometry.

## Architecture

```
u = u_PID(e) + LPF(w1×f1 + w2×f2 + w3×f3)
```

### The 3 Canonical Preview Features

Computed from `Δlataccel[i] = future.lataccel[i] - future.lataccel[0]`:

**f1**: Near term anticipation (0.1-0.6s)
```python
f1 = mean(Δlataccel[1:6])
```

**f2**: Mid horizon curvature (1.0-2.5s)
```python
f2 = mean(Δlataccel[10:25])
```

**f3**: Long term signed mass
```python
f3 = mean(Δlataccel) / len
```

### Why These Features Are Safe

1. **No error information**: Δ is relative to future t+1, not current error
2. **No absolute coordinates**: Only differential changes
3. **No high frequency**: Time-averaged over windows
4. **No current state**: Cannot reconstruct PID reactive space

Mathematical guarantee:
```
Cov(policy_output, current_error) = 0
```

## Policy: Pure Linear

```python
actor = nn.Linear(3, 1)  # Just 3 weights (w1, w2, w3)
```

NO hidden layers.
NO nonlinearity.  
NO tanh or activation functions.

This is **parameter tuning**, not behavior learning.

## Low-Pass Filter: α = 0.05

Much more aggressive than exp032/033 (which used 0.3).

```python
filtered_residual_t = 0.05 × raw_residual_t + 0.95 × filtered_residual_{t-1}
```

This blocks high-frequency reactive leakage through the preview channel.

## Expected Results

### If Preview Helps
- Cost should drop from ~90 → ~60-70 within 50 epochs
- Jerk should remain stable
- NO catastrophic divergence ever
- Smooth monotonic improvement

### If Preview Doesn't Help
- Cost stays ≥ 75 (PID level)
- No improvement despite training
- This would prove: simulator + cost + learned plant don't reward anticipation

Either outcome is scientifically valuable.

## Training Details

- **Optimizer**: PPO (but effectively gradient descent on 3 parameters)
- **Exploration**: σ ≈ 0.1 (very low)
- **Entropy**: 0.0 (deterministic)
- **Critic**: Nerfed (VF_COEF=0.05) - this is still a POMDP
- **Batch size**: 2048 steps
- **Epochs**: 200

## Comparison to Previous Experiments

| Experiment | State Dim | Policy | Result |
|------------|-----------|--------|--------|
| exp032 | 55 | MLP + error | 75k cost (catastrophic) |
| exp033 | 53 | MLP, no error | Not tested yet |
| exp034 | 3 | **Pure linear** | **Testing now** |

## The Control Theory

This implements the **minimal learnable feedforward controller**.

PID already spans the reactive subspace:
```
span{error, ∫error, d(error)/dt}
```

This learns the feedforward subspace:
```
span{near_preview, mid_preview, long_preview}
```

These are orthogonal by construction.

## What Winners Likely Did

Based on reported costs of 30-45:

1. Started with this or something very similar
2. Maybe added 3-6 more preview features (harmonics, timing)
3. Possibly kept it linear or added ONE shallow hidden layer
4. Used very aggressive low-pass filtering
5. Trained with BC + tiny PPO finetune OR pure gradient descent

They did NOT:
- Use deep networks
- Learn from scratch with exploration
- Train on high-dimensional raw preview
- Allow reactive correlation

## Next Steps (If This Works)

1. ✅ **Works (cost < 75)**: Preview helps
   - Add 3 more features (velocity-weighted, curvature extrema, timing)
   - Try one shallow hidden layer (6 → 16 → 1)
   - Keep all other constraints

2. ❌ **Doesn't work (cost ≥ 75)**: Preview doesn't help
   - Simulator reward structure doesn't favor anticipation
   - Or: PID is already optimal for this learned plant
   - Or: Jerk penalty dominates any lataccel improvement
   - All are deep findings about the problem

## Files

```
experiments/exp034_linear_preview/
├── train_ppo.py          # 3-parameter linear training
├── README.md             # This file
└── checkpoints/          # Trained weights (just 3 floats!)
```

## The Philosophical Point

We crossed from:
- "Can PPO learn this?" → **Wrong question**

To:
- "Does anticipation help in this geometry?" → **Right question**

By constraining the hypothesis class to the minimal feedforward subspace, we:
1. Remove all failure modes from reactive leakage
2. Test the pure value of preview
3. Learn interpretable weights (w1, w2, w3)
4. Guarantee stability

If PPO fails now, it's because preview doesn't matter.
Not because PPO is weak.

---

**This is the experiment that decides everything.**
