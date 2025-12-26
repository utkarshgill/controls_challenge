# Critical Fix: State Normalization

## The Problem
State features had wildly different magnitudes:
```
error:      ~0.1
lataccel:   ~0.5
v_ego:      ~33     ← 100x larger!
a_ego:      ~0.05
curvatures: ~0.0001 ← nearly invisible!
```

**Result:** 
- v_ego dominates gradients → network ignores other features
- Curvatures too small → network can't learn future anticipation
- Slow learning, stuck at 3000+ cost after 20 epochs

## The Fix
Added `OBS_SCALE` normalization (like LunarLander):
```python
OBS_SCALE = [10, 2, 0.03, 20, 1000, 1000, 1000, 1000, 1000, 1000]
state_normalized = state * OBS_SCALE
```

**Result:** All features now in ~[-1, 1] range
- Equal gradient contribution
- Curvatures now visible to network
- Should learn much faster

## Other Changes
- Increased epochs: 100 → 500 (more time to converge)
- Initial σ remains 0.3 (low exploration)

## Expected Performance
- **Before:** 3200 cost @ epoch 15 (45x worse than PID)
- **After:** Should drop below 500 within 50 epochs
- **Target:** < 70 (beat PID), stretch: < 45 (beat competition)

