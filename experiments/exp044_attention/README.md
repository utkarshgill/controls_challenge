# Exp044: Simple Cross-Attention for Feedforward Preview

## Motivation

Previous experiments (exp043) plateaued at Val ~100-250 using a simple MLP that processes all inputs uniformly. The hypothesis is that **feedforward (FF) and feedback (FB) have different information geometry**:

- **Feedforward**: Sequential temporal structure (future_κ[50]) → needs selective attention
- **Feedback**: Point-in-time state (current[7]) → needs dense processing

## Architecture

### Attention-Based Separation

```
Input: [current_state(7), future_κ(50)] = 57 dims

┌─────────────────────────────────────┐
│ Feedforward Path (Attention)        │
│                                     │
│ Query: current_state → "What to    │
│        look for in future?"         │
│                                     │
│ Keys/Values: future_κ + position →  │
│             "What's coming?"        │
│                                     │
│ Attention: Dynamic weighting based  │
│           on current state          │
│                                     │
│ Output: ff_features [32]            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Feedback Path (MLP)                 │
│                                     │
│ Input: current_state                │
│                                     │
│ MLP: Dense processing               │
│                                     │
│ Output: fb_features [64]            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Combine                             │
│                                     │
│ concat(ff_features, fb_features)    │
│         ↓                           │
│    Actor / Critic                   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ PID Residual Stream (α=0.1)         │
│                                     │
│ final_action = 0.1 * PID +          │
│                0.9 * network        │
└─────────────────────────────────────┘
```

## Key Features

### 1. Dynamic Preview Horizon
Attention allows the network to:
- Focus on **near future** at low speed or high error (reactive)
- Focus on **far future** at high speed or low error (anticipatory)
- Learn context-dependent preview weights

### 2. Positional Encoding
Each future_κ[i] is augmented with normalized position `i/50`:
```python
future_κ_pos[i] = [κ[i], i/50]
```
This tells the network "when" each curvature occurs.

### 3. Explicit FF+FB Separation
Instead of asking a single MLP to discover both:
- Attention specializes in temporal preview (FF)
- MLP specializes in state feedback (FB)
- Both paths are learned end-to-end

### 4. PID Baseline
10% PID feedback provides stability baseline while network learns anticipatory control.

## Expected Advantages Over Exp043

| Exp043 (MLP) | Exp044 (Attention) |
|--------------|-------------------|
| Uniform processing | Selective attention |
| Fixed receptive field | Dynamic preview horizon |
| Must discover FF+FB | Explicit separation |
| All features mixed | Geometry-aware |

## Training

### Stage 1: BC
```bash
python train_bc.py
```
- Collects PID demonstrations
- Network learns residual over PID baseline
- Saves `bc_init.pt`

### Stage 2: PPO
```bash
python train.py
```
- Loads BC warm-start
- Fine-tunes with PPO
- Target: <45 cost

## Interpretability

Attention weights show **what the network is looking at**:
```python
# During inference
ff_features, attn_weights = self.attention(current, future_κ)

# attn_weights[i] = importance of future_κ[i]
# Can visualize: "At high speed, it attends to timesteps 30-50"
```

This is a major advantage for debugging and understanding learned policies.

## Hypothesis

If this works better than exp043, it validates that:
1. Information geometry matters (temporal vs point-in-time)
2. Architectural inductive biases help learning
3. Attention can learn speed-dependent preview strategies

If it doesn't improve, it suggests:
1. The MLP was already learning appropriate features
2. The bottleneck is elsewhere (hyperparameters, reward shaping, etc.)
3. More sophisticated architectures won't help until we fix fundamentals
