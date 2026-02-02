# Bug Fixes for Exp039

## Critical Bugs Fixed

### 1. **Action Mismatch Bug** (The main killer)

**Problem:**
```python
# collect_episodes() sampled action A and stored it
_, raw_ff = actor_critic.act(...)
episode_actions.append(raw_ff)  # Store A

# Then sim.step() → controller.update() sampled AGAIN → action B
# PPO trained on (state, A, reward) but simulator executed (state, B)
# Result: NO LEARNING - wrong state-action pairs
```

**Fix:**
- Added `self.presampled_ff` to `HybridController`
- During training: sample once, pass to controller via `controller.presampled_ff = ff_action`
- During eval: controller samples internally (presampled_ff is None)
- Now PPO trains on the EXACT actions that were executed

**Files changed:**
- `HybridController.__init__`: Added `self.presampled_ff = None`
- `HybridController.update()`: Use presampled if available, otherwise sample
- `collect_episodes()`: Set `controller.presampled_ff` before stepping

---

### 2. **Dead Network from Bad Initialization**

**Problem:**
```python
# Applied gain=0.01 to ALL 6 layers
for m in actor.modules():
    nn.init.orthogonal_(m.weight, gain=0.01)

# Result: (0.01)^6 = 1e-12 → network outputs literally zero
```

**Fix:**
```python
# Use normal gain (sqrt(2)) for internal ReLU layers
for m in actor.modules():
    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

# Only make the FINAL output layer small
nn.init.orthogonal_(actor.fc3.weight, gain=0.01)
```

**Why it works:**
- Internal layers can learn features at normal scale
- Only the final layer starts near zero (so FF ≈ 0 initially)
- Network can actually produce gradients and learn

---

## Expected Behavior After Fixes

**Before:**
- FF actions: constant 0.0
- Eval cost: stuck at ~110 (worse than PID)
- No learning after 68+ epochs

**After:**
- FF actions: should show variation (non-zero std)
- Eval cost: should decrease below PID baseline (~75)
- Learning: gradients flow, network updates, performance improves

---

## Testing

Restart training with:
```bash
cd /Users/engelbart/Desktop/stuff/controls_challenge/experiments/exp039_pid_plus_ff
python train.py
```

Watch for:
1. ✓ FF actions with non-zero std (e.g., std=0.02-0.10)
2. ✓ Eval cost decreasing over epochs
3. ✓ Best cost < 75 (beating PID)

