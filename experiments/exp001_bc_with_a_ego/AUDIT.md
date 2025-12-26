# Deep Audit: PPO Implementation & Learnability Analysis

## 1️⃣ COST CALCULATION ✅

**Finding**: Costs are calculated from ACTUAL trajectories, not future_plan.

```python
# tinyphysics.py:183-190
def compute_cost(self) -> Dict[str, float]:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    
    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
```

✅ **CORRECT**: Uses actual executed trajectory (current_lataccel_history vs target_lataccel_history)

---

## 2️⃣ PPO IMPLEMENTATION COMPARISON

### A. Architecture ✅ MATCHES

| Component | beautiful_lander.py | train_ppo_parallel.py | Status |
|-----------|---------------------|------------------------|--------|
| Trunk | 1 layer, 128 hidden | 1 layer, 128 hidden | ✅ Match |
| Actor head | 3 layers, 128 hidden | 3 layers, 128 hidden | ✅ Match |
| Critic head | 3 layers, 128 hidden | 3 layers, 128 hidden | ✅ Match |
| log_std | Parameter | Parameter | ✅ Match |
| tanh squashing | Yes | Yes | ✅ Match |
| OBS_SCALE multiplication | Yes | Yes | ✅ Match |

### B. PPO Algorithm ✅ MATCHES

| Component | beautiful_lander.py | train_ppo_parallel.py | Status |
|-----------|---------------------|------------------------|--------|
| GAE computation | Line 110-120 | Line 248-259 | ✅ Match |
| Advantage normalization | Yes | Yes | ✅ Match |
| PPO-clip loss | Yes | Yes | ✅ Match |
| Critic MSE loss | Yes | Yes | ✅ Match |
| Entropy bonus | Yes | Yes | ✅ Match |
| Gradient clipping | 0.5 | 0.1 | ⚠️ Different |
| tanh_log_prob | Yes | Yes | ✅ Match |

### C. Rollout Pattern ✅ MATCHES

Both use `AsyncVectorEnv` with identical rollout logic:
- Store (state_tensor, raw_action) during rollout
- Accumulate rewards/dones
- PPO.update() with batched data

### D. Key Differences ⚠️

| Parameter | beautiful_lander.py | train_ppo_parallel.py | Impact |
|-----------|---------------------|------------------------|---------|
| lr | 1e-3 | 1e-5 | 🔴 100× slower learning |
| eps_clip | 0.2 | 0.1 | ⚠️ More conservative |
| entropy_coef | 0.001 | 0.0 | ⚠️ No exploration bonus |
| log_std init | zeros (1.0) | log(0.05) | 🔴 20× less exploration |
| grad_clip | 0.5 | 0.1 | ⚠️ Tighter clipping |
| K_epochs | 10 | 4 | ⚠️ Less optimization |
| batch_size | 10k | 2k | ⚠️ Smaller batches |

**🚨 CRITICAL ISSUES:**
1. **Learning rate 100× too small** (1e-5 vs 1e-3)
2. **Exploration 20× too conservative** (log_std = log(0.05) vs 0)
3. **No entropy bonus** (0.0 vs 0.001)

---

## 3️⃣ REWARD FUNCTION AUDIT

### Our Current Reward:
```python
# train_ppo_parallel.py:139-143
lat_cost = (current_lataccel - target_lataccel) ** 2
jerk_cost = (current_lataccel - self.prev_lataccel) ** 2
reward = -(lat_cost + jerk_cost) / 100.0
```

### Evaluation Cost:
```python
# tinyphysics.py:184-189
lat_accel_cost = np.mean((target - pred)**2) * 100
jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
total_cost = (lat_accel_cost * 50) + jerk_cost
```

### 🚨 MISMATCHES:

1. **Jerk calculation different**:
   - Reward: `(current - prev)²`
   - Eval: `(Δcurrent / dt)²` (same numerically but conceptually dt=0.1)

2. **Weighting completely wrong**:
   - Reward: `lat + jerk` (equal weight)
   - Eval: `50×lat + jerk` (lat is 50× more important!)

3. **Scaling mismatch**:
   - Reward: divide by 100
   - Eval: multiply by 100

### ✅ **CORRECT reward should be:**
```python
lat_cost = (current_lataccel - target_lataccel) ** 2
jerk_cost = ((current_lataccel - self.prev_lataccel) / DEL_T) ** 2
reward = -(50 * lat_cost + jerk_cost)  # Match eval exactly!
```

---

## 4️⃣ SIGNAL LEARNABILITY ANALYSIS

### A. Input Signal Quality ✅ GOOD

**State (56D):**
```
- error, error_diff, error_integral (3D) ← PID terms
- current_lataccel, v_ego, curv_now (3D) ← current physics
- future_curv[50] (50D) ← predictive info
```

**What's available:**
- ✅ Current error (immediate feedback)
- ✅ Error dynamics (derivative & integral)
- ✅ Current speed & acceleration demand
- ✅ 5 seconds of future path (50 × 0.1s)

**What's MISSING:**
- ❌ a_ego (longitudinal acceleration) ← FRICTION CIRCLE!
- ❌ Explicit friction margin
- ❌ Vehicle dynamics (tire slip, etc.)

### B. Output Signal Feasibility ✅ REASONABLE

**Action space:** `steer ∈ [-2, 2]`
- Continuous control
- Direct actuation (no discretization)
- tanh squashing ensures bounds

**Dynamics:** `current_lataccel = f(steer, v_ego, ...)`
- Smooth, differentiable
- Learnable mapping

### C. Learnability Assessment

**Theoretical difficulty:** ⭐⭐⭐☆☆ (Medium)

Why learnable:
1. ✅ Markov: Current state has enough info
2. ✅ Smooth dynamics: Small steer changes → small lataccel changes
3. ✅ Future plan: 50-step lookahead enables prediction
4. ✅ Dense reward: Get signal every step
5. ✅ Bounded: Finite state/action spaces

Why hard:
1. ❌ Delayed feedback: Steering → tire force → lataccel (lag)
2. ❌ Speed-dependent: Same steer has different effects at different speeds
3. ❌ Jerk-tracking tradeoff: Can't optimize both perfectly
4. ❌ Missing physics: a_ego not in state

**Winner got < 45 with PPO → Definitely learnable!**

---

## 5️⃣ PPO ARRANGEMENT ANALYSIS

### Current Setup:
```
State (56D) → ActorCritic(128 hidden, 4 layers) → Action (1D)
├─ Actor: state → (μ, σ) → steer ∈ [-2, 2]
└─ Critic: state → V(s)

Training:
- 8 parallel envs
- 10k steps/epoch (≈25 episodes)
- GAE advantages
- PPO-clip updates
```

### Issues with Current Arrangement:

**1. State Representation ❌**
- Missing a_ego (friction circle)
- Missing explicit dynamics
- 50 future curvatures might be redundant (high dim)

**2. Network Architecture ❓**
- 4 layers × 128 = small for 56D input
- No attention over future plan
- Treats all 50 future steps equally (should decay?)

**3. Reward Function 🔴 WRONG**
- Doesn't match evaluation cost
- Equal weight to lat/jerk (should be 50:1)
- This is THE critical bug!

**4. Hyperparameters 🔴 TOO CONSERVATIVE**
- lr = 1e-5 (100× too small)
- exploration = 0.05 (20× too small)
- entropy = 0 (no exploration bonus)

### Better Arrangement Options:

**Option A: Fix Current (Immediate)**
```python
# 1. Fix reward to match eval
reward = -(50 * lat_cost + jerk_cost)

# 2. Increase learning rate
lr = 1e-3  # Match beautiful_lander

# 3. Increase exploration
log_std = 0.0  # Start at 1.0 like reference

# 4. Add entropy bonus
entropy_coef = 0.001
```

**Option B: Add a_ego (Physics-Informed)**
```python
# State: 56D → 57D
state = [..., v_ego, a_ego, curv_now, ...future_curv]

# Explicit friction awareness
```

**Option C: Temporal Architecture (Advanced)**
```python
# CNN/LSTM over future plan
trunk → LSTM(future_curv[50]) → concat(current_state) → policy
```

**Option D: Hierarchical (Overkill)**
```python
# High-level: path planning (slower)
# Low-level: tracking (faster)
```

---

## 6️⃣ HYPOTHESES: WHAT'S BLOCKING LEARNING

### Hypothesis 1: REWARD MISMATCH 🔥 **MOST LIKELY**
**Evidence:**
- Reward weights lat/jerk equally
- Eval weights 50:1
- Network optimizes wrong objective
- This explains why PPO ≈ BC ≈ random (all fail equally)

**Test**: Fix reward, retrain
**Expected**: Huge improvement (could reach < 60)

### Hypothesis 2: LEARNING TOO SLOW 🔥 **VERY LIKELY**
**Evidence:**
- lr = 1e-5 (100× smaller than reference)
- log_std = -3.0 (20× less exploration)
- entropy = 0 (no exploration bonus)

**Test**: Increase lr to 1e-3, log_std to 0, entropy to 0.001
**Expected**: Faster convergence

### Hypothesis 3: MISSING a_ego ⚠️ **POSSIBLE**
**Evidence:**
- Physics: friction circle depends on a_ego
- File 00069: 5× more |a_ego|, BC fails catastrophically
- PID doesn't need it (reactive), but feedforward does

**Test**: Add a_ego to state
**Expected**: Better on hard files with braking

### Hypothesis 4: EXPLORATION INSUFFICIENT ⚠️ **POSSIBLE**
**Evidence:**
- std = 0.05 very tight
- No entropy bonus
- Network might not explore OOD states

**Test**: Increase std, add entropy
**Expected**: Better generalization

### Hypothesis 5: NETWORK TOO SMALL ❓ **UNLIKELY**
**Evidence:**
- 128 hidden × 4 layers = 66k params
- Reference uses same size and works
- Problem isn't that complex

**Test**: Double network size
**Expected**: Minimal improvement

### Hypothesis 6: SAMPLE EFFICIENCY ❓ **UNLIKELY**
**Evidence:**
- 10k steps/epoch × 100 epochs = 1M steps
- Reference solves LunarLander in < 1M
- Should be enough data

**Test**: Train longer
**Expected**: Minimal improvement if other bugs exist

---

## 7️⃣ ROOT CAUSE ANALYSIS

**Primary suspects (in order):**

1. **🔴 REWARD FUNCTION MISMATCH** (90% confidence)
   - Wrong objective → network learns wrong behavior
   - This is a BLUNDER (user's word)
   - Must fix immediately

2. **🔴 HYPERPARAMETERS TOO CONSERVATIVE** (80% confidence)
   - 100× slower learning
   - 20× less exploration
   - Compounds with reward mismatch

3. **🟡 MISSING a_ego** (50% confidence)
   - Physics-critical for friction
   - But PID works without it (reactive vs feedforward)
   - Might only matter for hard files

4. **🟡 EXPLORATION** (30% confidence)
   - Tight std might limit OOD discovery
   - But should converge eventually

5. **⚪ ARCHITECTURE** (10% confidence)
   - Current arch matches reference
   - Unlikely bottleneck

---

## 8️⃣ RECOMMENDED FIX PRIORITY

### 🔴 P0: CRITICAL BUGS (Fix immediately)
1. **Reward function**: Match eval cost (50:1 weighting)
2. **Learning rate**: 1e-5 → 1e-3
3. **Exploration**: log_std from -3.0 → 0.0

### 🟡 P1: IMPORTANT (Fix after P0)
4. **Entropy bonus**: 0.0 → 0.001
5. **K_epochs**: 4 → 10
6. **Batch size**: 2k → 10k

### 🟢 P2: NICE TO HAVE (Try after P0/P1 work)
7. **Add a_ego**: 56D → 57D state
8. **Gradient clip**: 0.1 → 0.5
9. **Anti-windup**: Apply to gym env (already done)

---

## 9️⃣ EXPECTED OUTCOMES

### After fixing P0 bugs:
```
Current: PPO ≈ 100 (doesn't learn)
After P0: PPO ≈ 60-70 (learns but not optimal)
```

### After fixing P1:
```
After P1: PPO ≈ 50-60 (good learning)
```

### After P2 (a_ego):
```
After P2: PPO ≈ 45-50 (near-optimal)
Target: < 45
```

### If still > 45 after all fixes:
- Need MPC or better state representation
- Or target is near-theoretical optimum
- Or winner used ensembles/tricks

---

## 🎯 CONFIDENCE LEVELS

**Can we reach < 45 with PPO?**

With current code: ❌ No (broken reward)
With P0 fixes: ⚠️ Maybe (60-70 expected)
With P0+P1+P2: ✅ Likely (45-55 expected)

**Winner got < 45 → It's possible!**

But need to:
1. Fix reward ASAP
2. Match reference hyperparams
3. Add physics (a_ego)
4. Possibly tune more

---

## 📋 NEXT STEPS (When user says "go")

1. Create exp002_ppo_fixed/
2. Fix reward function (50:1 weighting)
3. Match beautiful_lander hyperparams exactly
4. Train 100 epochs
5. Evaluate
6. If < 60: Add a_ego (exp003)
7. If still > 60: Debug more

**DO NOT IMPLEMENT YET - Waiting for user confirmation**

