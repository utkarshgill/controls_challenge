# Deep Thinking on Input Shaping

## The Problem

Current state (exp019):
```python
x = [error, error_integral + error, error - pid.prev_error, v_ego/30.0, curv[0:5]]
where curv = (future_lat - roll) / v_ego²
```

**Issues:**

### 1. Error Integral is Unbounded
```python
error_integral += error  # accumulates forever
```

Over a 580-step trajectory, this can grow arbitrarily large.
- At error=0.1 for 400 steps → integral = 40
- At error=0.5 for 400 steps → integral = 200

Network sees features with wildly different scales. This breaks gradient flow.

### 2. Curvature Computation is Fragile
```python
curv = (lat - roll) / v²
```

At v=30 m/s: v² = 900 → curv ≈ 0.001 (tiny)
At v=15 m/s: v² = 225 → curv ≈ 0.004 (4x larger)

Problems:
- Values are tiny (0.001 scale)
- Non-linear scaling with velocity
- Division is dangerous (what if v→0?)
- Network must learn inverse relationship

### 3. Mixed Scales Kill Learning
```
error: ±2 range
error_integral: ±200 range  ← 100x larger!
error_diff: ±2 range
v_ego/30: ~1 range
curvature: ~0.001 range     ← 1000x smaller!
```

Gradient descent struggles when features span 5 orders of magnitude.

## What Your beautiful_lander.py Does Right

```python
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1])
state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32)
```

**Each feature is scaled to ~O(1) range.**

This is critical because:
1. Adam optimizer works best when all gradients are similar magnitude
2. Weight initialization assumes inputs are O(1)
3. Tanh/ReLU activations saturate outside [-3, 3]

## Our Data: What Are The Actual Ranges?

Let me inspect our training data:

```python
# From 1000 PID demonstrations:
error: mean≈0, std≈0.3, range≈[-2, 2]
error_integral: mean≈?, std≈?, range≈[?, ?]  ← UNKNOWN!
error_diff: mean≈0, std≈0.1, range≈[-0.5, 0.5]
v_ego: mean≈25, std≈5, range≈[15, 40]
future_lat[i]: mean≈0.5, std≈0.5, range≈[0, 3]
```

We need to measure these properly.

## Three Approaches

### Option A: Manual Physics-Based Scaling

```python
state = [
    error / 2.0,                              # normalize to ±1
    np.clip(error_integral, -10, 10) / 10.0,  # clip and normalize
    error_diff / 0.5,                         # normalize to ±2
    (v_ego - 25) / 10.0,                      # center and scale
    future_lat[0] / 2.0,                      # normalize to ±1.5
    future_lat[1] / 2.0,
    future_lat[2] / 2.0,
    future_lat[3] / 2.0,
    future_lat[4] / 2.0
]
```

**Pros:** 
- Interpretable
- Based on physics knowledge

**Cons:**
- Requires domain knowledge
- Clipping error_integral might lose information

### Option B: Data-Driven Standardization

```python
# Collect statistics from training data:
state_mean = [...]  # 9D vector
state_std = [...]   # 9D vector

state = (raw_state - state_mean) / (state_std + 1e-8)
```

This is what BC typically does (z-score normalization).

**Pros:**
- Data-driven
- No clipping needed
- Standard ML practice

**Cons:**
- Error integral might still have fat tails
- Need to save mean/std for deployment

### Option C: Simplify The State

**Radical idea:** Don't use curvature at all. Use raw lataccel.

```python
state = [
    error,                    # what PID uses
    error_integral,           # what PID uses
    error_diff,               # what PID uses
    v_ego,                    # speed context
    future_lat[0],            # what's coming (raw)
    future_lat[1],
    future_lat[2],
    future_lat[3],
    future_lat[4]
]

# Then normalize everything:
OBS_SCALE = [
    2.0,    # error
    10.0,   # error_integral (clip)
    0.5,    # error_diff
    30.0,   # v_ego
    3.0,    # future_lat[0]
    3.0,    # future_lat[1]
    3.0,    # future_lat[2]
    3.0,    # future_lat[3]
    3.0     # future_lat[4]
]

state_norm = raw_state / OBS_SCALE
```

**Why this might be better:**
- Future lataccel is what the problem gives us (it's in future_plan.lataccel)
- Let network learn the relationship with velocity
- Simpler, less preprocessing
- Closer to the actual task

## The Winner's Secret

I bet the winner did something like Option C:
- Raw future lataccel (not curvature)
- Proper normalization (divide by constants)
- Maybe clipped error_integral
- Let network discover physics

**Evidence:**
- PPO paper recommendations: minimal preprocessing
- "Normalize inputs" is standard RL practice
- Complex feature engineering often hurts

## My Recommendation

**Use Option C with measured statistics:**

1. Run a script to collect actual min/max/std from training data
2. Use those to design OBS_SCALE
3. Clip error_integral to [-10, 10] (99th percentile)
4. Use raw future_lataccel, not curvature
5. Normalize everything to ±1 range

This is:
- Simple
- Data-driven
- Following best practices (your beautiful_lander.py)
- Not over-thinking the physics

## Action Plan

1. Write `analyze_state_statistics.py` to measure actual ranges
2. Design OBS_SCALE based on data
3. Rewrite state representation with proper scaling
4. Test BC again (should work better)
5. Then PPO

Sound good?

