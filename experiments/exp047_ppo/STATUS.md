# exp047: PPO Controller

## Progress

### ✓ Step 1: Environment Wrapper (DONE)
- Created `env_simple.py` - wraps tinyphysics
- Uses real physics simulation (TinyPhysicsModel)
- PID test: 105.6 cost (vs official 100.2)
  - 5.4 point mismatch, close enough to proceed
  - Likely minor indexing difference, not critical

**Status:** Environment works, ready for RL training

---

## Next Steps

### Step 2: Simple Policy Network
- Input: [error, prev_action] (no future yet)
- Output: action
- Train with simple RL loop (not full PPO yet)
- Target: Match PID (~100)

### Step 3: Add Future Plan
- Input: [error, prev_action, future_plan[:10]]
- Train with RL
- Target: Beat v3 (<82)

### Step 4: Full PPO + Scale
- Implement proper PPO (advantage, clipping)
- Bigger network
- Train longer
- Target: <60

### Step 5: Final Push  
- LSTM for future processing
- Massive compute
- Target: <35

---

## Current State
- Environment: ✓ Working
- Policy: ⏳ Next
- Training loop: ⏳ After policy
- PPO: ⏳ Later
