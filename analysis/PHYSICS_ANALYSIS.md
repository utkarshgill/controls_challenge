# Physics Analysis: What Law Are We Learning?

## The Problem (Bicycle Model)

**Dynamics:**
```
lateral_accel = v² × curvature
curvature = steering_angle / wheelbase (simplified)
```

**Cost Function:**
```
total_cost = 50 × (Σ(actual_lat - target_lat)²) + (Σ(jerk)²)
```

**Time scales:**
- dt = 0.1s
- 49 future steps = 4.9 seconds ahead
- At v=30 m/s: looking 147 meters ahead!

## What is the Optimal Control Law?

### 1. **Error Correction (PID terms)**
- `P`: Proportional to current error → immediate response
- `I`: Integral of error → eliminate steady-state offset
- `D`: Derivative of error → predict and dampen oscillations

### 2. **Speed Dependency**
At 2× speed, same curvature needs:
- 4× more lateral acceleration (v² relationship)
- But LESS steering angle per meter traveled
- **Key**: Control gains must be speed-dependent!

### 3. **Predictive Component** 
With 49 future steps, we can:
- See upcoming turns 5 seconds ahead
- Pre-position the car
- Plan smooth trajectories (minimize jerk)

### 4. **Jerk Minimization**
Jerk = d(lataccel)/dt

For smooth control:
- Action should be close to prev_action
- Large changes only when necessary
- This is a **soft constraint** on action smoothness

## The "Natural" State Space

### Current Approach (exp025):
```python
state = [error, error_i, error_d, v_ego, prev_action] + future_curvatures[49]
```

### Physics-Motivated Alternative:

**1. Normalized Error (dimensionless)**
```python
error_norm = error / (v_ego² × dt)  # "How many time constants behind?"
```

**2. Speed-Normalized Integral**
```python
error_i_norm = error_i / v_ego  # Distance-based, not time-based
```

**3. Predictive Lookahead (distance-weighted)**
```python
# Weight future curvatures by "time to reach"
# Closer events matter more
weights = exp(-t / tau) where tau ~ 1-2 seconds

near_curv = Σ(curv[0:10] × weights)   # Next 1 second
mid_curv = Σ(curv[10:30] × weights)   # 1-3 seconds
far_curv = Σ(curv[30:49] × weights)   # 3-5 seconds
```

**4. Speed-Normalized Action**
```python
action_norm = prev_action / v_ego²  # Steering effectiveness scales with v²
```

## The Key Insight: MPC Perspective

The winner likely framed this as **Model Predictive Control**:

1. **Use the future trajectory** to solve a finite-horizon optimization
2. **The network learns the solution** to this optimization
3. This is much more powerful than reactive PID

**What should the network learn?**
```
optimal_action = f(
    current_state,           # Where am I?
    future_trajectory,       # Where do I need to go?
    prev_action,            # What was I doing?
    dynamics_model          # How does the car respond?
)
```

## Hypothesis: Multi-Timescale State

Instead of raw 49 curvatures, compress to multiple timescales:

```python
# Temporal convolution, but interpret as:
immediate = avg(curv[0:5])      # Next 0.5s - emergency response
tactical = avg(curv[5:20])      # 0.5-2s - active maneuvering  
strategic = avg(curv[20:49])    # 2-5s - path planning
```

This gives the network:
- **What to do NOW** (immediate)
- **What's coming SOON** (tactical)
- **What to prepare for** (strategic)

## Why PPO Works for the Winner

PPO can learn:
1. **Non-linear speed dependencies** (v² in dynamics)
2. **Trajectory optimization** (looking ahead optimally)
3. **Risk-aware control** (conservative on hard routes, aggressive on easy)

But it needs:
- **Right state representation** (physics-motivated)
- **Right reward shaping** (cost-to-go, not instantaneous)
- **Enough training** (100+ epochs)

## Action Items

Test these representations:
1. Speed-normalized state
2. Multi-timescale future compression
3. MPC-inspired "cost-to-go" features

