# Next Big Run — Changes Checklist

All changes for a clean from-scratch training run.
Reference: Google "What Matters" 250k-agent study, AAAI 2024 colored noise paper,
"Implementation Matters" (Engstrom et al.), PPO-BR, Beta distribution papers.

---

## 1. Observation Normalization (running mean/std)
**Priority: HIGH** — study says "always use observation normalization"

- Add `RunningMeanStd` for observations (separate from the reward one)
- During rollout, normalize `obs = (obs - obs_rms.mean) / obs_rms.std`
- Clip normalized obs to `[-5, 5]` (keep the clip, but apply AFTER normalization)
- Must save/load `obs_rms` state with checkpoint for inference consistency
- Controller (`exp050_rich.py`) must load and apply the same normalization at eval time
- Replaces the current hand-tuned `S_LAT`, `S_STEER`, etc. scaling constants

## 2. Remove Huber Loss + Value Clipping
**Priority: HIGH** — study says "use NEITHER Huber loss nor PPO-style value clipping"

Current:
```python
v_clipped = old_val[idx] + (val - old_val[idx]).clamp(-10.0, 10.0)
vf_loss = torch.max(
    F.huber_loss(val, ret_t[idx], delta=10.0, reduction='none'),
    F.huber_loss(v_clipped, ret_t[idx], delta=10.0, reduction='none'),
).mean()
```

Replace with plain MSE:
```python
vf_loss = F.mse_loss(val, ret_t[idx])
```

## 3. Colored (Temporally Correlated) Noise for Exploration
**Priority: MEDIUM** — AAAI 2024 paper, most impactful early in training

- Replace i.i.d. Beta sampling with temporally correlated noise
- Implementation: sample pink noise (1/f spectrum) sequence per episode,
  add to the Beta mean instead of sampling from the Beta directly
- Or: use an AR(1) process on the raw action: `noise_t = ρ * noise_{t-1} + √(1-ρ²) * ε`
  where `ε ~ Beta_sample - Beta_mean`, and `ρ ≈ 0.5` (tune this)
- Produces coherent steering exploration maneuvers instead of per-step jitter
- More useful when σ is still large (early training)

## 4. Adam eps = 1e-5 ✅ (already wired in)
Prevents huge parameter updates when Adam's variance estimate is small.

## 5. Reward Normalization via RunningMeanStd ✅ (already wired in)
Divides rewards by running std before GAE computation.
Stabilizes value targets across varying reward magnitudes.

## 6. Per-Minibatch Advantage Normalization ✅ (already wired in)
Normalize advantages within each minibatch, not globally.
Keeps gradient magnitudes stable across all K_EPOCHS passes.

## 7. Recompute Advantages Each Data Pass
**Priority: LOW** — study recommends this

Currently we compute advantages once and reuse across all K_EPOCHS.
Better: recompute GAE after each full pass over the data using updated value estimates.
Trade-off: adds compute (K extra value forward passes over all data).

## 8. Last-Layer Init (already correct ✅)
Actor last layer: `orthogonal_(gain=0.01)` — matches "100x smaller" recommendation.
Critic last layer: `orthogonal_(gain=1.0)` — standard.

## 9. Discount Factor
**Status: γ=0.95 (validated)**
Study says γ is one of the most important hyperparameters, tune per-environment.
We tested 0.9, 0.95, 0.97. 0.95 is our sweet spot.

## 10. GAE λ=0.9 ✅ (matches study recommendation exactly)

---

## Training Config for Next Run

```bash
# Fresh from scratch — no RESUME
DECAY_LR=1 REMOTE_HOSTS=169.254.159.243 FRAC=1.25:1 \
  CSVS=1000 EPOCHS=400 WORKERS=10 \
  .venv/bin/python experiments/exp050_rich_ppo/train.py
```

- 400 epochs (longer runway, no restart needed with stable training)
- No RESUME — clean start with all changes baked in from epoch 0
- All normalizations active from step 1

---

## Inference-Time (exp050_rich.py) Changes Needed

- Load `obs_rms` (mean/std) from checkpoint and apply at inference
- Remove hand-tuned `S_LAT`, `S_STEER` etc. if obs normalization replaces them
- Keep LPF and Newton correction as-is (they're inference-only enhancements)

---

## NOT Doing (decided against)

- MPC (purged — too myopic, fights the policy)
- DBIAS/innovation tracking (purged — tracks white noise)
- Horizon optimal target / tridiagonal solver (purged — made things worse)
- Learnable DELTA_SCALE (Beta already self-limits, 0.25 is correct)
