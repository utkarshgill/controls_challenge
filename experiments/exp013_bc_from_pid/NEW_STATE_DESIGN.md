# New State Representation (58D)

## Improvements

### 1. Add Error Derivative (PID D-term)
- Helps with damping and stability
- Shows rate of change of tracking error

### 2. Explicit Current and Target Curvatures
- Instead of just `error`, show both what we ARE doing and what we SHOULD do
- More geometric, physically interpretable
- Speed-invariant representation

### 3. Friction Circle Term
- Lateral and longitudinal forces share available tire grip
- If braking hard → less lateral grip available
- Helps network understand dynamic limits

## New State Vector (58D)

```python
state = [
    # [0] current_curvature - The path we're currently on
    current_curvature = current_lataccel / v_ego²
    
    # [1] target_curvature - The path we should be on
    target_curvature = target_lataccel / v_ego²
    
    # [2] curvature_error - The gap (what PID tries to minimize)
    curvature_error = target_curvature - current_curvature
    
    # [3] curvature_error_integral - PID I-term
    curvature_error_integral = Σ(curvature_error)
    
    # [4] curvature_error_derivative - PID D-term (NEW!)
    curvature_error_derivative = (curvature_error - prev_curvature_error) / dt
    
    # [5] v_ego - Current speed
    v_ego
    
    # [6] a_ego - Longitudinal acceleration
    a_ego
    
    # [7] friction_available - Available lateral grip (NEW!)
    # Friction circle: sqrt((μg)² - a_long²) / (μg)
    friction_available = sqrt(1 - (a_ego / 10.0)²)
    
    # [8:57] future_curvatures[50] - Future road geometry
    future_curvatures[0:50]
]
```

## Physics Intuition

### Curvature Space
```
Old: error = target_lataccel - current_lataccel (speed-dependent)
New: curvature_error = target_curvature - current_curvature (geometric)

Same corner at different speeds:
  v=20: lataccel_error = 4.0 m/s²    (different)
  v=30: lataccel_error = 9.0 m/s²
  
  v=20: curvature_error = 0.01 /m    (same!)
  v=30: curvature_error = 0.01 /m
```

### Friction Circle
```
Total available grip: ~10 m/s² (combined lateral + longitudinal)

If a_ego = 0 (cruising):
  friction_available = 1.0 (full lateral capacity)

If a_ego = 5 (braking):
  friction_available = sqrt(1 - 0.25) = 0.87 (87% lateral capacity)

If a_ego = 8.66 (hard braking):
  friction_available = sqrt(1 - 0.75) = 0.5 (50% lateral capacity)

Network learns: "When braking hard, I have less steering authority"
```

### Error Derivative
```
curvature_error_derivative = rate of change of tracking error

Positive: Error is growing → Need more aggressive control
Negative: Error is shrinking → Can ease off
Zero: Error is stable → Maintain current action

Classic PID D-term for damping oscillations
```

## Expected Benefits

1. **Better generalization** - Curvature is speed-invariant
2. **More learnable** - Explicit current/target makes goal clear
3. **Physics-aware** - Friction circle respects vehicle limits
4. **Better control** - Error derivative helps damping
5. **Consistent units** - Everything related to curvature in same space

## Comparison

| Dimension | Old (55D) | New (58D) |
|-----------|-----------|-----------|
| Error representation | lataccel | curvature (speed-invariant) |
| Current state | implicit | explicit (current_curvature) |
| Derivative term | ❌ | ✅ (curvature_error_derivative) |
| Friction awareness | implicit (only a_ego) | explicit (friction_available) |
| Future info | curvatures[50] | curvatures[50] (same) |

## Implementation Priority

**Phase 1: Curvature space (56D)**
- current_curvature, target_curvature
- curvature_error, curvature_error_integral, curvature_error_derivative
- v_ego, a_ego
- future_curvatures[50]
- **Skip friction term initially** (simpler)

**Phase 2: Add friction (58D)**
- Add friction_available term
- See if it helps

Start with Phase 1, evaluate, then add Phase 2 if needed.





