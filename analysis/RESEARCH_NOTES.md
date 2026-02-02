# Research Notes: The BC Ceiling Problem

## The Puzzle

We have a paradox:

**exp019 results:**
- Training loss: 0.000008 (near perfect)
- Test cost: 75.72 (same as PID)

The network can fit the data perfectly. It has:
- Non-linear capacity (2x64 hidden units)
- Rich state (9D: PID state + velocity + 5 future curvatures)
- Enough samples (5000 demonstrations)

Yet it achieves the **exact same cost as PID**.

## What's Actually Happening

### The Teacher-Student Problem

PID is our teacher. PID's policy is:
```
action = P*error + I*error_integral + D*error_diff
```

PID **never looks at:**
- velocity (v_ego)
- future plan (future_lataccel)

So in the training data, the mapping from full 9D state → action is:
```
action = f(error, ei, ed)
       ≈ ignore v_ego
       ≈ ignore future[0:5]
```

### What The Network Learns

When we train with MSE loss on PID demonstrations:
```
min ||network(state) - PID(state)||²
```

The network learns to **imitate PID**, not to **control well**.

Perfect imitation → Learn to ignore the extra features.

### Why This Is Fundamental

This isn't a bug. This is behavioral cloning working correctly.

BC learns: "What would PID do?"
We want: "What should I do?"

These are different questions.

## The Information Theory View

Shannon would say:

**Mutual Information:**
```
I(action; future | error, ei, ed) = 0
```

Given PID state, the future plan provides **zero bits** about PID's action.
Because PID doesn't use it.

The network optimally learns this. Adding future to state doesn't change the entropy of the teacher's policy.

## Why The Winner Used PPO

This is why BC + PPO is the recipe:

**BC Phase:**
- Input: Rich state (including future)
- Output: Imitate PID
- Result: Stable initialization, ~75 cost
- Problem: Can't beat teacher

**PPO Phase:**
- Objective: Minimize actual cost (not imitate)
- Gradient: Comes from cost function, not teacher
- Discovery: "Wait, I can use this future info!"
- Result: Learn proactive control, <45 cost

PPO's value: It gets gradient signal from **outcome** not **imitation**.

## Alternative Hypothesis: Maybe Feature Engineering Is Wrong?

Wait. Let me question my features.

**Current:** future curvature = (lat - roll) / v²

This is physics-based. Curvature is the "right" representation.

But... what if it's not what the network needs?

### What Does The Simulator Actually Use?

Looking at tinyphysics.py:
```python
Inputs: v_ego, a_ego, road_lataccel, current_lataccel, steer_action
Output: next current_lataccel
```

The simulator doesn't know about "curvature". It knows about **lataccel**.

### Maybe We're Over-Processing

**Current state:**
```
future_curv = (future_lataccel - roll) / v²
```

**Problems:**
1. Division by v² → unbounded when v→0
2. Subtracting roll → assumes we know the "road curvature"
3. Physics interpretation ≠ what simulator needs

**Alternative:**
```
state = [error, ei, ed, v_ego/30, future_lataccel[0:5]/5]
```

Just normalize the raw lataccel. Let the network figure out the physics.

### But Would This Help?

Probably not. The problem isn't feature engineering.

The problem is: PID doesn't use future_lataccel, so the training signal (imitate PID) says "ignore it."

No amount of feature engineering fixes this.

## The Fundamental Limit

**BC learns the teacher's policy ceiling.**

If teacher cost = 75, BC cost ≥ 75.

We can't beat 75 with BC alone because:
```
L_BC = E[||π_student - π_teacher||²]
```

Minimizing imitation error ≠ minimizing control cost.

## What Would Work?

### Option 1: Better Teacher

Don't clone PID. Clone something better.

**Ideas:**
- MPC (Model Predictive Control) - uses future, gets ~60 cost?
- Optimal control solution - theoretical minimum
- Human demonstrator - but we don't have this

**Problem:** We don't have a better teacher.

### Option 2: Data Augmentation

What if we perturb PID's actions and label with actual outcomes?

```
For each state:
  try action = PID(state) + noise
  rollout and measure cost
  if cost < PID_cost:
    label = improved_action
```

This is basically offline RL / dataset distillation.

**Problem:** 
- Computationally expensive
- Still limited by regions PID explores
- Basically reinventing PPO the hard way

### Option 3: PPO (The Right Answer)

Just do PPO. This is what it's for.

**Why PPO works:**
1. Starts from BC init (safe, stable)
2. Explores small variations
3. Gets gradient from **cost**, not imitation
4. Discovers: "future info helps!"
5. Learns: proactive control

The winner used this because it's the right tool.

## My Recommendation

**Don't do more BC experiments.**

We've proven:
- ✅ BC can recover PID (exp017)
- ✅ BC can fit complex functions (exp019)
- ✅ BC hits ceiling at teacher performance

**Next: Jump to PPO (exp020)**

Start from exp019 checkpoint (good init) and fine-tune with PPO.

Expected result: 60-65 cost quickly, then grind to <45.

## One More Thing...

Actually, there's ONE BC experiment worth trying:

**What if we clone PID but ONLY on the future-dependent error?**

No. That's overthinking.

The issue is clear: BC ceiling is real. PPO is the path.

## Decision

Skip remaining BC experiments (exp020-022 in roadmap).

Go straight to exp023: PPO fine-tuning from exp019 init.

This is what the winner did. This is what we should do.

---

*"Don't fight the problem. Fight the right problem."* - Every researcher who's wasted time eventually

