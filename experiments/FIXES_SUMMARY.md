# Critical Fixes Applied

## Experiment 040: PID Cloning

### Problem
BC was training on raw PID outputs but evaluation used `tanh()`, creating a fundamental mismatch:
- Training: Learn f(x) = Px + Ix_i + Dx_d (linear, unbounded)
- Inference: Apply tanh(f(x)) (nonlinear, bounded to [-1, 1])
- Result: Network learned distorted coefficients to compensate for tanh

### Evidence
```
Learned: P=0.2211, I=0.0993, D=-0.1386
Target:  P=0.1950, I=0.1000, D=-0.0530
Error:   0.112444

BC Eval: 182.2 (way worse than PID baseline ~75)
```

The D coefficient was especially wrong (-0.1386 vs -0.0530) because derivatives can spike beyond [-1, 1].

### Fix Applied
Removed `tanh()` from:
1. `OneNeuronActor.act()` - no longer applies tanh to actions
2. PPO loss computation - changed `tanh_log_prob()` to `compute_log_prob()` (regular Gaussian)

### Expected After Fix
```
Learned coefficients should match PID almost exactly:
P ≈ 0.195, I ≈ 0.100, D ≈ -0.053
Error < 0.01
BC Eval ≈ 75 (matching PID baseline)
```

---

## Experiment 039: PID + Feedforward

### Problems Fixed

#### 1. Action Mismatch (Critical - prevented all learning)
**Problem:**
```python
# In collect_episodes(): sampled action A
_, raw_ff = actor_critic.act(...)
episode_actions.append(raw_ff)  # Store A

# Then sim.step() → controller.update() sampled AGAIN → action B
# PPO trained on (state, A, reward) but executed (state, B)
```

**Fix:** Controller now accepts pre-sampled actions
- Added `controller.presampled_ff` attribute
- Training: sample once, pass to controller
- Eval: controller samples internally

#### 2. Dead Network Initialization
**Problem:**
```python
# Applied gain=0.01 to ALL 6 layers
for m in actor.modules():
    nn.init.orthogonal_(m.weight, gain=0.01)
# Result: (0.01)^6 = 1e-12 → outputs literally zero
```

**Fix:** Normal gain for internal layers, small gain only on final output
```python
# Internal layers: gain = sqrt(2) for ReLU
# Final layer: gain = 0.01 (so FF starts near zero)
```

#### 3. Critic Initialization Bug
**Problem:** Tried to access `self.ac_device.critic.modules()` but critic layers are direct attributes, not a sub-module

**Fix:** Initialize critic layers explicitly by name

### Expected After Fix
- FF actions: non-zero std (e.g., 0.02-0.10)
- Learning: cost decreases over epochs
- Best cost: < 75 (beating PID baseline)

---

## Action Items

1. **Exp 040**: Delete cache file and restart
```bash
rm /path/to/exp040_pid_tuning/pid_demonstrations_1000.pkl
python train.py
```
Should now recover PID perfectly.

2. **Exp 039**: Restart training (old one was running broken code)
```bash
# Kill old training (Ctrl+C)
python train.py
```
Should now show learning and FF actions with variation.

---

## Physics Insight

**Exp 040**: You cannot learn a linear transformation through a nonlinear lens. The tanh distorts the function space.

**Exp 039**: 
- Action mismatch = training on counterfactuals (wrong causal link)
- Dead initialization = no gradient signal can flow
Both are fatal to learning.

