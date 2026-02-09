# Comma.ai Controls Challenge - Comprehensive Data Analysis

**Date:** Feb 2, 2026  
**Analyst:** Deep analysis of all 20,000 trajectories  
**Total Data Points:** 12,000,000

---

## Executive Summary

This report analyzes the full comma.ai steering control dataset to understand:
1. What we're trying to control
2. Why this problem is hard
3. What patterns exist in the data
4. How to approach solving it

**Key Finding:** The problem is fundamentally about **predictive control with nonlinear dynamics** - the controller must anticipate the target trajectory using the 5-second lookahead while compensating for vehicle state variations and road disturbances.

---

## Critical Finding: Speed-Dependent Steering Gain

From the logged expert data (warmup period), the steer-to-target relationship varies with speed:

| Speed (m/s) | Gain | Offset | Correlation | Samples |
|-------------|------|--------|-------------|---------|
| 0-5 | 0.60 | 0.08 | 0.62 | 19,504 |
| 5-10 | 1.54 | 0.17 | 0.89 | 17,731 |
| 10-15 | 1.28 | 0.19 | 0.82 | 30,103 |
| 15-20 | 1.43 | 0.22 | 0.82 | 41,681 |
| 20-25 | 1.51 | 0.18 | 0.78 | 42,258 |
| 25-30 | 1.76 | 0.13 | 0.77 | 78,592 |
| 30-35 | 1.63 | 0.07 | 0.74 | 86,943 |
| 35-40 | 1.84 | 0.05 | 0.79 | 7,096 |

**To achieve target≈1 m/s² at different speeds:**

| Speed (m/s) | Steering Needed |
|-------------|-----------------|
| 0-5 | 0.68 ± 0.23 |
| 5-10 | 0.58 ± 0.26 |
| 10-15 | 0.44 ± 0.22 |
| 15-20 | 0.40 ± 0.22 |
| 20-25 | 0.33 ± 0.18 |
| 25-30 | 0.28 ± 0.15 |
| 30-35 | 0.30 ± 0.12 |

**This is physics:** `lateral_accel ≈ v² × curvature`  
Controllers MUST adjust steering based on speed!

---

## 1. Dataset Overview

### Scale
- **20,000 trajectories** (one per CSV file)
- **600 timesteps per trajectory** (60 seconds @ 10 Hz)
- **12 million total data points**
- Control window: steps 100-500 (40 seconds of evaluation)

### Data Fields
| Field | Description | Range |
|-------|-------------|-------|
| `t` | Time (seconds) | 0-60s |
| `vEgo` | Vehicle speed (m/s) | 0-43 m/s (0-155 km/h) |
| `aEgo` | Vehicle acceleration (m/s²) | -13 to +9 m/s² |
| `roll` | Road bank angle (rad) | ±0.17 rad (±10°) |
| `targetLateralAcceleration` | Target lat accel (m/s²) | ±5 m/s² in control window |
| `steerCommand` | Steering command | ±2 (clipped) |

---

## 2. Global Statistics (All 20k Files)

### Vehicle Speed (vEgo)
```
Mean:   23.43 m/s (84 km/h, 52 mph)
Std:     9.19 m/s
Range:   0 - 43 m/s (0 - 155 km/h)
```
→ Highway speeds dominate, but significant low-speed data exists

### Target Lateral Acceleration (Control Window Only)
```
Mean:    0.011 m/s² (nearly centered)
Std:     0.462 m/s²
99%:    [-1.80, 1.90] m/s² (about ±0.2g)
Range:  [-4.97, 6.02] m/s² (about ±0.5g)
```
→ Most values are small; extremes are rare but important

### Road Roll
```
Range:  ±10 degrees
Effect: Up to ±1.7 m/s² lateral acceleration
```
→ Road banking is a significant disturbance!

---

## 3. Why This Problem Is Hard

### 3.1 Nonlinear Dynamics

The physics model is a **neural network (black box)**:
- Steer → Lataccel relationship is complex and nonlinear
- Speed significantly affects the steering gain
- 20-step history context matters

**Speed-Steer Correlation by Speed Bucket:**
| Speed (m/s) | Correlation | Data Points |
|-------------|-------------|-------------|
| 0-10 | 0.47 | 1.17M |
| 10-20 | 0.74 | 2.61M |
| 20-30 | 0.75 | 4.49M |
| 30-40 | 0.73 | 3.64M |

→ Low speed behavior is fundamentally different!

### 3.2 Delayed Response

The ONNX physics model has:
- **Context length:** 20 timesteps (2 seconds of history)
- **Tokenized lataccel:** Discretized to 1024 bins
- **Stochastic output:** temperature=0.8

This means:
- Actions don't have instant effect
- Controller must predict/anticipate
- Cannot achieve perfect tracking (physical impossibility)

### 3.3 Disturbances

**Road Roll:**
- Adds up to 1.7 m/s² of lateral acceleration
- Acts as unmeasured disturbance
- Controller must compensate

**Vehicle Dynamics:**
- Speed changes affect steering sensitivity
- Acceleration affects response

### 3.4 Jerk Penalty

The cost function:
```
total_cost = lataccel_cost × 50 + jerk_cost

where:
  lataccel_cost = mean((target - actual)²) × 100
  jerk_cost = mean((Δactual / 0.1s)²) × 100
```

→ Tracking error is **50× more important** than jerk
→ But jerk still matters - can't bang-bang control
→ Trade-off between tracking and smoothness

### 3.5 Trajectory Diversity

**Difficulty varies wildly:**
- Easiest: Nearly constant target (highway straight)
- Hardest: High-frequency oscillations (winding roads)
- Some trajectories are **10× harder** than others

**Per-Trajectory Target Range:**
```
Mean:  1.64 m/s²
Std:   3.25 m/s²
Min:   0.001 m/s² (dead straight)
Max:   262 m/s² (data artifact at t=0 only)
```

---

## 4. Frequency Analysis

Dominant frequencies in target lateral acceleration:
```
0.002 Hz (very slow changes)
0.010-0.022 Hz (slow curves)
```

→ Most target changes are **slow** (< 0.1 Hz)
→ Fast oscillations (> 1 Hz) are rare
→ A good controller doesn't need very fast response

---

## 5. Theoretical Limits

### Perfect Tracking (Impossible)
If we could perfectly track the target:
- `lataccel_cost = 0`
- `jerk_cost = mean(target_jerk²) × 100 = 11.82`
- `total_cost = 11.82`

**But this is physically impossible** because:
- The physics model has delay/dynamics
- Steer → lataccel is not instantaneous
- The model is stochastic

### Current Performance
| Controller | Total Cost | Status |
|------------|------------|--------|
| PID Baseline | ~107 | Reference |
| Best BC (exp023) | ~103 | 4% improvement |
| Target | <45 | 56% improvement needed |

---

## 6. Data Quality Notes

### Outliers
- Some trajectories have extreme values (-260 m/s²!) at t=0
- **All outliers are OUTSIDE the control window**
- Control window (steps 100-500) is clean

### Trajectory Lengths
- Most trajectories: 600 timesteps
- Some shorter (min: 442 timesteps)
- All contain the full control window

---

## 7. Key Insights for Controller Design

### What the Controller Sees
1. **Current state:** `roll_lataccel`, `v_ego`, `a_ego`
2. **Target:** `target_lataccel` (current)
3. **Future plan:** 50 steps (5 seconds) of targets ahead
4. **Current lataccel:** feedback from physics model

### What Makes a Good Controller

1. **Speed-aware:** Steering gain varies significantly with speed
2. **Predictive:** Use the 5-second lookahead effectively
3. **Smooth:** Minimize jerk while tracking
4. **Robust:** Handle roll disturbances and dynamics variations

### Promising Approaches

1. **MPC (Model Predictive Control):**
   - Uses the physics model for prediction
   - Can optimize over future horizons
   - Computationally expensive but effective
   - Current experiments achieving ~100-150 cost

2. **Learned Controllers:**
   - Behavior cloning from expert data
   - PPO for direct optimization
   - Can be fast at inference time

3. **Hybrid:**
   - Use MPC to generate expert demonstrations
   - Train NN to imitate MPC behavior
   - Get speed of NN with quality of MPC

---

## 8. Visualization Summary

Generated figures in `analysis/data_analysis_figures/`:

| Figure | Description |
|--------|-------------|
| `01_global_distributions.png` | Speed, accel, roll, target distributions |
| `02_correlations.png` | Speed vs target, speed vs steer, target vs steer |
| `03_sample_trajectories.png` | 16 random trajectory examples |
| `04_trajectory_statistics.png` | Per-trajectory variance distributions |
| `05_time_aligned_stats.png` | Mean/std over time across all trajectories |
| `06_frequency_analysis.png` | Power spectral density of targets |
| `07_extreme_cases.png` | Easiest, hardest, fastest, hilliest trajectories |

---

## 9. Recommendations

### For Immediate Progress
1. **Focus on speed-dependent gain:** The controller needs different behavior at different speeds
2. **Use the future plan:** 5 seconds of lookahead is very valuable
3. **Handle roll:** Compensate for road banking explicitly

### For Breakthrough Performance
1. **Need predictive model:** Either learned or physics-based
2. **Optimize for the true cost:** Not just imitation
3. **Handle the delay:** The physics model has lag that must be anticipated

---

## 9. State of the Art: Comma's Results

From [comma.ai's RL Controls blog post](https://blog.comma.ai/rlcontrols/):

### Key Results
- **CMA-ES achieves cost 48.0** with only 6 parameters
- PPO fails to converge on this problem
- Evolutionary methods outperform gradient-based RL

### Why PPO Fails

The GPT-based physics model creates **autocorrelated noise**:
```
At time t, noise depends on noise at t-1, t-2, ... t-20
PPO assumes Markovian transitions (noise is independent)
This assumption is VIOLATED
```

The neural network uses its own past predictions as context, creating temporal correlations that break standard RL assumptions.

### Why CMA-ES Works

Evolutionary methods:
- Operate on **parameter level** (not action level)
- Don't require per-step gradient signals
- Turn the problem from MDP to **contextual bandit**
- Only care about **total cost** over a rollout (noise averages out)

### Recommended Approach

Based on comma's success and our data analysis:

1. **Design a feature-rich controller with 6-10 parameters:**
   ```python
   steer = (p0 + p1/v_ego) * error       # Speed-adaptive P
         + (p2 + p3/v_ego) * target      # Speed-adaptive FF
         + p4 * roll_lataccel            # Roll compensation
         + p5 * future_delta             # Anticipation
   ```

2. **Use CMA-ES to optimize parameters:**
   - Evaluate on many trajectories per iteration
   - Cost = average over N segments

3. **Key features to include:**
   - Speed-dependent terms (critical - gain varies 2× across speed range)
   - Roll compensation
   - Anticipation using future_plan

---

## Appendix: Running the Analysis

```bash
cd /path/to/controls_challenge
python analysis/data_deep_dive.py
```

This will:
1. Load all 20,000 CSV files
2. Compute comprehensive statistics
3. Generate visualization figures
4. Print detailed analysis to console
