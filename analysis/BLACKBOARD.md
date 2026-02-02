# BLACKBOARD

## Problem
Control simulated car lateral dynamics. Track target lateral acceleration with minimal error and jerk.

## Physics Model (tinyphysics.onnx)
- Autoregressive ML model mimicking bicycle dynamics
- Inputs: `v_ego`, `a_ego`, `road_lataccel`, `current_lataccel`, `steer_action`
- Output: next `lataccel`
- Trained on real openpilot driving data
- Non-linear, realistic noise

## Cost Function
```
lataccel_cost = mean((target - actual)²) × 100
jerk_cost = mean((Δlataccel / Δt)²) × 100
total_cost = lataccel_cost × 50 + jerk_cost
```
- Only eval steps 100:500 (400 steps)
- Δt = 0.01s

## Data
- 20k routes from real driving
- Split: 15k train / 2.5k val / 2.5k test
- Must shuffle to avoid bias

## Baseline: PID
- Coefficients: P=0.195, I=0.100, D=-0.053
- Cost: ~75-80
- Simple, stateful, linear

## What We Proved (exp017)
**1 neuron perfectly recovers PID from 1k demonstrations**
- Input: [error, error_integral, error_diff]
- Output: action
- No bias, linear layer
- Exact coefficient match after 5k epochs
- Test cost: 75.61

## Key Insight
PID is linear in its state representation. Neural net can learn this exactly.

## Next Question
**Can we beat PID?**

Options:
1. Non-linear control (add hidden layers)
2. Better state representation (use future plan, v_ego, curvature)
3. Learn from better teacher (optimal control?)
4. End-to-end: state → action directly

## Constraints
- Action bounded: [-2, 2]
- Must be causal (no future info beyond plan)
- Autoregressive errors accumulate
- Jerk penalty limits aggressive control

## Hypothesis
PID is suboptimal because:
1. Fixed gains (not adaptive to speed/curvature)
2. Linear (can't exploit non-linear dynamics)
3. Hand-tuned (not optimized for actual cost)

Winner got <45 cost. That's 40% better than PID.
There's signal here.

## Winner's Recipe (PPO, <45 cost)

### What They Probably Did

**1. Richer State**
Not just [error, error_integral, error_diff]. Likely:
- Current error
- Velocity (v_ego) - control should adapt to speed
- Future plan lookahead (next 5-10 steps of target curvature)
- Maybe: a_ego, road_lataccel
- Maybe: recent history (last few errors/actions)

Why: Different speeds need different control. Curvature preview enables proactive control.

**2. Non-Linear Network**
Not 1 neuron. Probably:
- 2-3 hidden layers, 64-128 units each
- Tanh activations (bounded, smooth)
- Small network (fast inference, less overfitting)

Why: Exploit non-linear dynamics, adaptive gains.

**3. BC Initialization**
Critical insight: Start from BC on PID demonstrations.
- BC gets you to ~80-90 cost immediately
- PPO fine-tunes from there
- Pure RL from scratch probably fails (sparse reward, exploration)

Why: Warm start is everything in controls. PID is a good prior.

**4. PPO Training**
- Reward: `-cost` per episode (or maybe `-step_cost` per step)
- Baseline: Value function to reduce variance
- Small learning rate (1e-4 to 3e-5)
- Small clip epsilon (0.1-0.2)
- Many rollouts per update (to get stable gradients)
- Train on same simulator (tinyphysics.onnx)

Critical: Reward shaping. Probably weighted lataccel vs jerk to match cost function.

**5. The Edge**
What gets <45 vs ~80:
- **Proactive control**: Use future plan to start turning early
- **Speed-adaptive gains**: Lighter touch at high speed
- **Jerk awareness**: Smooth actions (PPO optimizes total cost directly)
- **Non-linear dynamics**: Exploit simulator quirks PID can't

### What They Didn't Do
- Fancy RL (PPO is simple, works)
- Huge networks (controls needs fast, robust)
- Model-free exploration (BC init is key)
- Complicated state (keep it simple, causal)

### Our Path
1. ✅ Prove concept: 1 neuron = PID
2. Add hidden layers + richer state
3. BC train to ~80-90
4. PPO fine-tune to <45

The winner didn't do magic. They did good ML engineering:
- Right state representation
- BC + PPO curriculum
- Careful hyperparameter tuning
- Optimize what you measure

---

## exp021 PPO BREAKTHROUGH (Shannon Analysis)

### The Signal Flow Bug

Standard PPO failed due to train/eval mismatch:
```
Train:  policy → mean+noise → execute → cost=bad → "avoid"
Eval:   policy → mean → execute → learned wrong thing!
```

Reward from NOISY execution blamed the MEAN (the signal).

### Fix: Decouple Exploration from Credit

```
1. Policy outputs mean
2. Execute mean+noise (explore)
3. Reward AS IF executed mean only
4. Update policy based on what IT did, not noise
```

**Results:**
- BC: 74.88
- Broken PPO: 74.88 → 110+ (degrading)  
- Fixed PPO: 85.9 → 84.6 → 82.5 → 79.3 (IMPROVING!)

### Shannon Insight

Exploration noise = channel noise  
Policy mean = signal  
**Don't blame signal for channel errors**

Training now...


## CTF STATUS (exp023-025)

**Current Best: 65.02** (Conv BC + action history)
**Target: <45**  
**Gap: 20 points (30%)**

### Key Findings:
1. **Conv architecture = 12% gain** (temporal patterns in curvature)
2. **Action history = 1% gain** (jerk minimization)
3. **PPO DEGRADES performance** (65 → 73) - BROKEN!

### Route Variance:
- Highway (easy): v=33, curv=0.0001, cost~40
- City (hard): v=15, curv=0.099 (100x!), cost~100
- Winner must excel at BOTH

### Next Steps:
1. Debug PPO (currently makes things worse)
2. Try different RL (SAC/TD3/offline)
3. Or completely different approach?

