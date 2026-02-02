# Approach: Physics-Informed Hybrid Control

## The Physics Problem

**Rate Limit:** Lateral acceleration can only change by 0.5 m/s² per 0.1s timestep.
- Maximum jerk: 5 m/s³
- To change 2 m/s²: need 0.4 seconds minimum
- This makes reactive control fundamentally limited

**PID is Reactive:**
```
error = target - current
action = Kp*error + Ki*∫error + Kd*derror/dt
```
- Sees error → responds
- But rate limit means: by the time you see error, too late to fix it
- Always playing catch-up

**The Gap:** PID gets ~75 cost. Winner got <45 cost. That's 40% improvement.

## The Solution: Feedforward + Feedback

**Total Control = Feedback (PID) + Feedforward (Learned)**

### Why This Structure?

1. **Feedback (PID):** Corrects for disturbances and model error
   - Proven stable
   - Handles unexpected perturbations
   - Eliminates steady-state bias
   
2. **Feedforward (Learned):** Anticipates future needs
   - Uses future_plan to predict steering needed
   - Starts steering BEFORE error appears
   - Compensates for rate limit

### Why Learn FF, Not Full Controller?

**Smaller problem:**
- Not learning 8D control (error, integral, derivative, future...)
- Only learning: "given future plan, how much to anticipate?"
- PID provides safety net

**Better gradient signal:**
- FF helps → cost drops → clear reward
- FF hurts → cost rises → clear penalty
- No confusion about which component to credit

**Safe initialization:**
- Start with FF ≈ 0 → pure PID (known stable)
- Explore small FF adjustments
- Can't catastrophically destabilize

## The Architecture

### Input: Future Plan as Bicycle Model Trajectory

The future_plan contains:
- `lataccel[50]` - where we need to be (0.1s to 5.0s ahead)
- `v_ego[50]` - speed affects turning radius
- `a_ego[50]` - weight transfer affects grip
- `roll_lataccel[50]` - road banking bias

These are the bicycle model parameters! At each future timestep k:
```
steer_needed[k] ≈ (lataccel[k] - roll[k]) / v[k]²
```

The network learns:
- Which future horizons matter (preview timing)
- How to weight them (temporal patterns)
- How to ramp smoothly (avoid jerk)
- How to adapt to state (speed-dependent gains)

### Network: 1D CNN Over Time

```python
Input: [lataccel, v_ego, a_ego, roll_lataccel] × 50 timesteps
       ↓
Conv1d(4→16, k=5) - learn local temporal patterns
       ↓
Conv1d(16→32, k=5, stride=2) - downsample, longer patterns
       ↓
Conv1d(32→64, k=3, stride=2) - high-level trajectory shape
       ↓
Flatten + MLP(128→64→1) - map to single FF action
       ↓
Output: FF steering ∈ [-1, 1]
```

**Why 1D CNN?**
- Translation invariant: same pattern at 0.5s or 1.0s recognized
- Hierarchical: local → global temporal structure
- Fewer parameters than fully connected
- Inductive bias: nearby timesteps correlated

## PPO Training

**From beautiful_lander.py (battle-tested):**
- Dual models: CPU rollout, GPU update (avoid transfer latency)
- Vectorized environments (16 parallel routes)
- Standard PPO: clip=0.2, γ=0.99, λ=0.95
- 100k steps per epoch, 5k batch size, 20 inner epochs

**Key difference from previous attempts:**
- Not learning full controller (smaller problem)
- Not BC initialization (no teacher bias)
- Not end-to-end (structured FF+FB)

**Reward = negative cost:**
- Shaped by GAE over trajectory
- Directly optimizes what we measure
- No manual reward engineering needed

## What We Expect to Learn

**Preview timing:**
- Optimal horizon ≈ 0.5-1.0s (rate limit / typical Δlataccel)
- Early layers: immediate future (0.1-0.3s)
- Later layers: longer trends (0.5-2.0s)

**Speed adaptation:**
- High speed → gentler FF (more momentum)
- Low speed → stronger FF (more responsive)
- Learned via v_ego channel

**Smooth ramping:**
- Can't output sharp FF changes (increases jerk)
- Conv learns smooth temporal patterns
- Entropy regularization encourages smooth policies

**Coordination with PID:**
- FF handles predictable trajectory following
- PID handles unpredictable disturbances
- Together: better than either alone

## Success Metrics

**Phase 1: Beat PID (cost < 75)**
- Proves FF is being used effectively
- Validates architecture and training

**Phase 2: Competitive (cost < 60)**
- 20% improvement over PID
- Shows preview helps significantly

**Phase 3: Winner-level (cost < 45)**
- 40% improvement over PID
- Demonstrates optimal preview + adaptation

## If This Works

1. **Analyze learned FF:**
   - Which conv filters activate when?
   - What temporal patterns does it recognize?
   - How does FF vary with speed?

2. **Ablations:**
   - Disable FF → should degrade to PID
   - Disable specific future_plan channels
   - Try different preview lengths

3. **Refinement:**
   - Bicycle model initialization
   - Fine-tune hyperparameters
   - Train longer

## If This Doesn't Work

**Check if FF is being used:**
- Monitor FF action magnitude during rollout
- Should be non-zero if learning anything

**Possible issues:**
1. Reward shaping insufficient (cost computed at end only)
2. Exploration too destructive (need smaller σ init?)
3. Network too large/small for problem
4. Need BC warm start after all?

**Fallback strategies:**
1. BC on synthetic "PID + simple preview" demos
2. Offline RL on recorded data
3. Direct optimization (CMA-ES on FF gains)
4. Hand-engineer FF, then fine-tune with PPO

---

## The Core Insight

Rate limit makes this a **preview problem**. PID can't preview. We must learn to anticipate.

But learning pure end-to-end is hard (exploration destroys stability).

**Hybrid = surgical:** Keep stable parts (PID), learn only what's missing (preview).

This is the clay-molding approach. Shape the architecture around physics. Let PPO find the optimal weights.

