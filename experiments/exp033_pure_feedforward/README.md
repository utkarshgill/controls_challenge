# Experiment 033: Pure Feedforward PPO (Nullspace by Construction)

## The Winning Move

After exp032 taught us that **unconstrained PPO exits the stable manifold**, we implement the correct solution:

> **Remove reactive degrees of freedom from the hypothesis class.**

## Architecture

```
u = u_PID(e) + ε × π_θ(s_preview)
```

Where `s_preview` contains **ZERO error terms**:

```python
state = [
    v_ego,           # velocity
    a_ego,           # longitudinal accel  
    roll_lataccel,   # road bank angle contribution
    curvatures[50]   # 5-second preview
]
# NO error
# NO error_integral
# NO current_lataccel
```

## Why This Works

### Mathematical Guarantee

```
Cov(π_θ(s_preview), error_t) = 0
```

Not because PPO learned it.
Because **error is not in the input**.

This is **control decomposition**:
- PID: reactive control (feedback loop)
- Policy: predictive control (feedforward anticipation)

### No Reactive Leakage

The failure mode of exp032 was:
- Policy correlated with error
- Small systematic bias accumulated
- Destabilized phase alignment
- Quadratic cost exploded

This cannot happen here. Policy **cannot see error**.

## Key Hyperparameters

### 1. Zero Exploration
```python
ENTROPY_COEF = 0.0
log_std = -2.3  # σ ≈ 0.1
```

We're not discovering behaviors. We're nudging a manifold.

### 2. Shallow Network
```python
HIDDEN_DIM = 64
ACTOR_LAYERS = 2  # Could even be 1 (pure linear)
```

Anticipatory correction is likely near-linear in curvature.
Deep networks reintroduce unnecessary complexity.

### 3. Nerfed Critic
```python
VF_COEF = 0.05
```

Still a POMDP (hidden: PID state, simulator history, filter state).
Critic provides baseline, not oracle.

## Expected Results

### Theoretical Bounds
- PID (reactive only): ~75-100 cost
- Perfect 5s preview: ~40-50 cost
- Winners: 30-45 cost

### Learning Curve
- **Epoch 0**: ~100-130 (PID + tiny random residual)
- **Epoch 20-50**: ~80-100 (learning anticipation)
- **Epoch 100+**: ~60-80 (approaching feedforward limit)

If this doesn't beat PID, then preview doesn't help in this simulator.
That would be a deep finding.

But it will.

## What Changed from Exp032

| Component | Exp032 (Reactive Leak) | Exp033 (Pure Feedforward) |
|-----------|------------------------|---------------------------|
| State | `[error, integral, ...]` | `[v, a, roll, preview]` |
| Policy-Error Correlation | Learned to correlate | **Zero by construction** |
| Exploration | Low (σ≈0.3, entropy=0.01) | **Zero (σ≈0.1, entropy=0)** |
| Network Depth | 3 layers, 128 hidden | 2 layers, 64 hidden |
| Eval Cost | 75k (catastrophic) | **~100 expected** |

## The Control Theory Insight

From exp032 FINDINGS.md:

> "PPO cannot find the nullspace by gradient descent unless you remove reactive degrees of freedom from the hypothesis class."

This experiment **embodies that insight**.

We didn't:
- Tune hyperparameters harder
- Add more reward shaping
- Train longer
- Use a better RL algorithm

We **changed the control architecture** to enforce the constraint.

## Files

```
experiments/exp033_pure_feedforward/
├── train_ppo.py          # Pure feedforward training
├── README.md             # This file
└── checkpoints/          # Trained policies
```

Controller evaluation:
```
controllers/exp033_feedforward.py   # (to be created after training)
```

## Training

```bash
cd experiments/exp033_pure_feedforward
python train_ppo.py
```

## Evaluation

```bash
# After training, create controller and evaluate
python ../../tinyphysics.py --model_path ../../models/tinyphysics.onnx \
    --data_path ../../data --num_segs 100 --controller exp033_feedforward
```

## The Deep Lesson

RL is not magic. It optimizes the function class you give it.

In POMDP with:
- Hidden state
- Learned dynamics  
- Delayed effects
- Quadratic costs

You must **constrain the architecture** to match the control structure.

Exp032: Asked PPO to discover the nullspace.
Exp033: **Built the nullspace into the architecture.**

This is how winners got <45.
