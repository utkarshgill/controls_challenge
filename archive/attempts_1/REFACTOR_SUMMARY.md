# PPO Refactor Summary

## Changes Made

### 1. State Representation: 5D → 10D Smart Summary
**Old (5D):** `[error, current_lataccel, v_ego, a_ego, current_curv]`

**New (10D):** 
```
[error, current_lataccel, v_ego, a_ego, current_curv,
 next_curv_1, next_curv_2, next_curv_3,
 future_max, future_mean]
```

**Rationale:**
- Near-term lookahead (3 steps = 0.3s) for anticipation
- Max/mean statistics summarize full horizon difficulty
- Low-dimensional like LunarLander (8D) for faster learning
- Avoids 50D curse while keeping future information

### 2. Initial Exploration: σ = 1.0 → 0.3
**Old:** `log_std = 0.0` → σ = 1.0
**New:** `log_std = -1.2` → σ = 0.3

**Rationale:**
- Random exploration is catastrophic in autoregressive control
- LunarLander tolerates high σ, TinyPhysics doesn't
- PPO will still increase σ if needed via gradient descent

### 3. Performance Baseline Check
- **PID**: ~70 cost (range: 6-142)
- **Random**: ~10,000 cost
- **Old PPO (σ=1.0)**: ~10,000-19,000 cost ← basically random!
- **Old PPO eval (deterministic)**: 300-500 ← learning something, but handicapped

## Expected Improvement
With lower exploration, PPO should:
1. Start closer to reasonable control (~500 instead of ~10,000)
2. Actually learn from gradient descent instead of random walk
3. Converge toward PID baseline (~70) and hopefully beat it

## Next Steps
Run `train_ppo.py` and monitor:
- Initial costs should be << 10,000 (if not, σ still too high)
- Training costs should decrease steadily
- Target: < 70 (beat PID), stretch goal: < 45 (beat competition winner)

