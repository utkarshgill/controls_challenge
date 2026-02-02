# Experiment 037: NNFF-Style Temporal Probes

## The Breakthrough Insight

**Previous experiments failed because**:
- Exp034: 3 aggregated features → lost temporal resolution (cost: 102)
- Exp036: 200D delta-encoded → too many correlated dimensions (hung)

**The NNFF lesson**:
Winners didn't use 50 raw future values. They used **6-8 carefully chosen temporal probes**.

NNFF isn't "better ML." It's **better feature engineering**.

## The 7 Temporal Probes

Stolen directly from NNFF's structure (adapted for Controls Challenge):

### Anticipation Probes (4)
```python
Δlat_03 = future.lataccel[3]  - target_lataccel(t)  # 0.3s ahead
Δlat_06 = future.lataccel[6]  - target_lataccel(t)  # 0.6s ahead
Δlat_10 = future.lataccel[10] - target_lataccel(t)  # 1.0s ahead
Δlat_15 = future.lataccel[15] - target_lataccel(t)  # 1.5s ahead
```

These answer: "How is the curve changing at specific horizons?"

### Roll Compensation Probes (2)
```python
Δroll_03 = future.roll[3]  - roll_lataccel(t)
Δroll_10 = future.roll[10] - roll_lataccel(t)
```

These mirror NNFF's roll and roll-rate effects.

### Speed Conditioning (1)
```python
v_ego (normalized)
```

## Why This is Different

| Approach | State Representation | Problem |
|----------|---------------------|---------|
| Exp034 | 3 aggregated means | Lost timing information |
| Exp036 | 200D raw deltas | High-dim, correlated, explosion |
| **Exp037** | **7 orthogonal probes** | **NNFF-style basis** |

NNFF didn't win with deeper networks or smarter PPO.
It won by asking the **right 7 questions** instead of dumping 50 correlated values.

## Policy: Pure Linear

```python
actor = nn.Linear(7, 1)  # Just 7 weights
```

PPO is not "learning physics."
PPO is **tuning gains on physically meaningful basis functions**.

This is how NNFF worked.
This is how comma's ML Controls was meant to be used.
This is the only way PPO survives quadratic jerk penalties.

## Expected Outcomes

| Cost Range | Interpretation |
|------------|----------------|
| **< 75** | ✅✅✅ Temporal structure was the missing piece! |
| **75-90** | ✅ Helps but not enough (need better probes or nonlinear) |
| **~102** | ❌ Preview fundamentally limited by PID teacher |
| **> 120** | 🔥 Something broke |

## The Deep Point

This experiment tests the NNFF hypothesis directly:

> Winners got <45 not because they had better RL, but because they fed PPO **orthogonal physical questions** instead of **raw correlated sequences**.

If this hits <75, we've found the recipe.
If it stays ~102, then winners used something else (better teacher, offline optimization, hand-tuned feedforward).

## Training

```bash
cd experiments/exp037_temporal_probes
python train_ppo.py
```

Expected: Fast convergence (7 parameters only), stable training, clear signal.

## Comparison to Winners

Winners (comma NNFF specifically) likely:
1. Started with temporal probes like these
2. Added 2-3 more (curvature rate, roll rate, maybe error @ -0.3s)
3. Used shallow MLP (1 hidden layer, 16-32 units max)
4. Kept exploration minimal
5. Applied aggressive low-pass filtering

Total parameter count for winner: probably **< 500 weights**.

This is not deep learning. This is **gain tuning on a physical basis**.

---

**This is the experiment that decides if PPO can work at all.**

If 7 orthogonal temporal probes don't beat PID, then:
- Preview learning from PID is fundamentally limited
- Winners used better teachers (MPC, optimal control)
- Or winners hand-crafted feedforward

Either way, clean answer.
