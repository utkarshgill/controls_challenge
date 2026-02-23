# exp050 Wall-to-Wall Technical Guide

---

## Table of Contents

1. [Problem Statement and Simulator Internals](#1-problem-statement-and-simulator-internals)
2. [The Batched Simulator](#2-the-batched-simulator)
3. [The Observation Space (256-dim)](#3-the-observation-space-256-dim)
4. [The Neural Network Architecture](#4-the-neural-network-architecture)
5. [Delta Actions](#5-delta-actions)
6. [Training Pipeline](#6-training-pipeline)
7. [The Batched Rollout (CPU + GPU Paths)](#7-the-batched-rollout-cpu--gpu-paths)
8. [Distributed Training (MacBook Cluster + Cloud)](#8-distributed-training-macbook-cluster--cloud)
9. [The MPC System (Inference-Time)](#9-the-mpc-system-inference-time)
10. [The 375x Speedup Journey](#10-the-375x-speedup-journey)
11. [The Journey Narrative](#11-the-journey-narrative)
12. [Potential Interview Questions and Answers](#12-potential-interview-questions-and-answers)

---

## 1. Problem Statement and Simulator Internals

### 1.1 The Challenge

comma.ai's Controls Challenge v2: track a target lateral acceleration profile using a black-box ONNX physics simulator. The simulator is a GPT-style autoregressive transformer — it takes steering commands and past predictions as input, and outputs the next lateral acceleration. 600 timesteps per route, ~20,000 routes in the dataset.

The controller sees four things each step:

| Input | Type | Description |
|---|---|---|
| `target_lataccel` | float | What lateral acceleration the car should have right now |
| `current_lataccel` | float | What lateral acceleration the car actually has |
| `state` | `State(roll_lataccel, v_ego, a_ego)` | Road roll component, ego velocity (m/s), ego longitudinal acceleration |
| `future_plan` | `FuturePlan(lataccel, roll_lataccel, v_ego, a_ego)` | Next 50 timesteps of each quantity |

The controller returns a single float: the steering command, in `[-2, 2]`.

### 1.2 The Cost Function

```
total_cost = lataccel_cost * 50 + jerk_cost
```

where:

```python
lataccel_cost = mean((target - predicted)^2) * 100       # over steps 100-500
jerk_cost     = mean((diff(predicted) / 0.1)^2) * 100    # over steps 100-500
```

The `50x` multiplier on lataccel means tracking accuracy matters far more than smoothness. To drop from 45 to 35, you need to roughly halve the RMS tracking error. Jerk cost has a noise floor around 18-20 because the ONNX model's stochastic sampling creates irreducible jitter.

### 1.3 Key Constants in `tinyphysics.py`

| Constant | Value | Meaning |
|---|---|---|
| `ACC_G` | 9.81 | Gravitational acceleration, used to convert road roll angle to lateral accel component |
| `FPS` | 10 | Simulator runs at 10 Hz |
| `CONTROL_START_IDX` | 100 | Controller output is ignored before step 100 (CSV steer used instead) |
| `COST_END_IDX` | 500 | Cost is computed over steps 100-500 only |
| `CONTEXT_LENGTH` | 20 | ONNX model looks at last 20 timesteps |
| `VOCAB_SIZE` | 1024 | Tokenizer bins for discretizing lateral acceleration |
| `LATACCEL_RANGE` | [-5, 5] | Clamp range for tokenization |
| `STEER_RANGE` | [-2, 2] | Valid steering command range |
| `MAX_ACC_DELTA` | 0.5 | Maximum change in lataccel per step (physics constraint) |
| `DEL_T` | 0.1 | Timestep duration (1/FPS = 0.1s) |
| `LAT_ACCEL_COST_MULTIPLIER` | 50.0 | Weight on tracking cost vs jerk cost |
| `FUTURE_PLAN_STEPS` | 50 | `FPS * 5` = 5 seconds of future plan |

### 1.4 `LataccelTokenizer`

The ONNX model is a language model over a discrete vocabulary. Lateral acceleration (continuous float) must be tokenized:

```python
class LataccelTokenizer:
    def __init__(self):
        self.bins = np.linspace(-5, 5, 1024)   # 1024 evenly-spaced bin centers

    def encode(self, value):
        value = np.clip(value, -5, 5)
        return np.digitize(value, self.bins, right=True)   # index into bins

    def decode(self, token):
        return self.bins[token]                             # bin center -> float
```

The `right=True` in `np.digitize` is critical. It means intervals are closed on the right: bin `i` covers `(bins[i-1], bins[i]]`. Getting this wrong shifts every token by one position, and the autoregressive simulation diverges silently over 500 steps.

The bins are float64. Decoding with float32 bins loses ~7 decimal digits of precision. Over 500 autoregressive steps this compounds — a float32 decode bug caused a 20% score regression (49.7 -> 59.2).

### 1.5 `TinyPhysicsModel` — The ONNX Physics Model

```python
class TinyPhysicsModel:
    def __init__(self, model_path, debug):
        # Single-threaded ONNX Runtime session
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.ort_session = ort.InferenceSession(model_bytes, options, ['CPUExecutionProvider'])
```

The model takes two inputs:

| Input | Shape | Dtype | Content |
|---|---|---|---|
| `states` | `(1, 20, 4)` | float32 | `[steer_action, roll_lataccel, v_ego, a_ego]` for last 20 steps |
| `tokens` | `(1, 20)` | int64 | Tokenized past lateral acceleration predictions |

Output: `(1, 20, 1024)` logits. Only the last timestep matters (autoregressive next-token prediction).

The `predict()` method:
1. Runs ONNX inference to get `(1, 20, 1024)` logits
2. Applies softmax with temperature 0.8 on the last timestep
3. Samples from the resulting probability distribution via `np.random.choice`
4. Returns the sampled token index

The `get_current_lataccel()` method:
1. Tokenizes past predictions via `encode()`
2. Builds the states array: `column_stack([actions, raw_states])` -> `(20, 4)`
3. Expands to batch dim: `(1, 20, 4)` float32 + `(1, 20)` int64
4. Calls `predict()` -> sampled token
5. Decodes token back to float via `decode()`

**The model is stochastic.** It samples from a probability distribution, not argmax. The RNG is seeded per-episode via `np.random.seed(md5(filepath) % 10**4)`.

### 1.6 `TinyPhysicsSimulator` — The Simulation Loop

The simulator runs episodes. Each episode:

1. **Load CSV data** via `get_data()`: reads `roll` (converted to `sin(roll)*g`), `vEgo`, `aEgo`, `targetLateralAcceleration`, and `-steerCommand` (sign flip: logged left-positive, sim uses right-positive)

2. **Reset**: initialize histories from first `CONTEXT_LENGTH=20` rows of CSV. Seed RNG with `md5(path)`.

3. **Main loop** (steps 20 through 599):
   ```
   for step_idx in range(CONTEXT_LENGTH, len(data)):
       state, target, futureplan = get_state_target_futureplan(step_idx)
       action = controller.update(target, current_lataccel, state, futureplan)
       if step_idx < CONTROL_START_IDX:
           action = csv_steer[step_idx]     # override controller during warmup
       action = clip(action, -2, 2)
       pred = onnx_model.get_current_lataccel(last_20_states, last_20_actions, last_20_preds)
       pred = clip(pred, current ± 0.5)     # MAX_ACC_DELTA physics constraint
       if step_idx >= CONTROL_START_IDX:
           current_lataccel = pred           # use ONNX prediction
       else:
           current_lataccel = target         # perfect tracking during warmup
   ```

4. **Cost computation** over steps 100-500 only.

Key subtlety: the `MAX_ACC_DELTA=0.5` clamp means the physics model can't change lateral acceleration by more than 0.5 per step (5.0/s). This is a hard physical constraint on how fast the car's lateral dynamics can respond.

The `set_model()` hook: if the controller has a `set_model` method, the simulator calls it during `__init__`, passing the `TinyPhysicsModel` instance. This is how the MPC controller gets access to the ONNX model for lookahead predictions.

### 1.7 `State` and `FuturePlan`

```python
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
```

`State` is a single timestep. `FuturePlan` contains lists of up to 50 future values for each quantity. The future plan comes directly from the CSV — it's the ground truth future trajectory that the car is supposed to follow.

---

## 2. The Batched Simulator

### 2.1 Why Batching

The reference simulator runs one episode at a time: 580 ONNX calls per episode, each processing `(1, 20, 4)`. For 5000 episodes, that's 2,900,000 ONNX calls.

But the ONNX model's transformer handles the batch dimension natively. Instead of 5000 x 580 calls of shape `(1, 20, 4)`, we can do 580 calls of shape `(5000, 20, 4)`. Same total compute, 5000x fewer kernel launches.

The constraint: autoregressive. Step N's prediction feeds into step N+1's input. Can't parallelize across timesteps. But all episodes can run in lockstep at the same timestep.

### 2.2 `BatchedPhysicsModel` (mirrors `TinyPhysicsModel`)

Same structure as the reference, but operates on `(N, ...)` arrays:

**CPU path (`_predict_cpu`):**
1. Run ONNX: input `(N, 20, 4)` float32 + `(N, 20)` int64 -> output `(N, 20, 1024)` logits
2. Softmax at temperature 0.8 on last timestep: `(N, 1024)` probabilities
3. Per-episode sampling via CDF + searchsorted (replacing `np.random.choice`)
4. Return `(N,)` sampled token indices

**GPU path (`_predict_gpu`):**
1. Accept pre-built GPU tensors (states, tokens) — zero CPU->GPU transfer
2. Run ONNX via `IOBinding`: bind GPU tensor pointers directly as inputs/outputs
3. Softmax on GPU via PyTorch
4. Sampling on GPU: `torch.cumsum` -> `torch.searchsorted`
5. Return GPU tensor of sampled indices

**`get_current_lataccel()` CPU path:**
```python
tokens = tokenizer.encode(past_preds)          # (N, 20) int via np.digitize
states = concat([actions[:,:,None], sim_states], axis=-1)  # (N, 20, 4)
sampled_tokens = _predict_cpu(states, tokens)
return tokenizer.decode(sampled_tokens)        # (N,) float64
```

**`get_current_lataccel()` GPU path:**
All on GPU:
```python
# Tokenize: torch.clamp + torch.bucketize (NOT np.digitize)
tokens = torch.bucketize(clamped_preds, bins_gpu, right=False)  # right=False!
# Build states in pre-allocated buffer
states_gpu[:,:,0] = actions.float()
states_gpu[:,:,1:] = sim_states.float()
# Predict via IOBinding
sample_tokens = predict(states_gpu, tokens)
# Decode: index into float64 bins on GPU
return bins_gpu[sample_tokens]                 # float64 GPU tensor
```

**Critical parity gotcha:** `torch.bucketize(right=False)` corresponds to `np.digitize(right=True)`. The semantics are inverted. Getting this wrong silently produces wrong tokens and the simulation diverges.

### 2.3 `BatchedSimulator` (mirrors `TinyPhysicsSimulator`)

Histories are `(N, T)` arrays instead of Python lists:

```python
self.action_history = np.zeros((N, T), np.float64)      # or torch GPU tensor
self.state_history = np.zeros((N, T, 3), np.float64)
self.current_lataccel_history = np.zeros((N, T), np.float64)
```

**RNG parity:** Each episode must use the same random sequence as the reference. The reference seeds `np.random.seed(md5(path) % 10**4)` inside `reset()`, then consumes one random value per step. The batched version pre-generates all random values:

```python
for i, f in enumerate(csv_files):
    seed = int(md5(str(f).encode()).hexdigest(), 16) % 10**4
    rng = np.random.RandomState(seed)
    self._rng_all[i, :] = rng.rand(T - CONTEXT_LENGTH)
```

The RNG generates values only for steps `CONTEXT_LENGTH` through `T`, indexed by `step_idx - CL`. Generating for all steps 0 through T would waste `CL` random draws, shifting the entire sequence.

**The rollout loop:**

GPU path:
```python
for step_idx in range(CL, T):
    actions = controller_fn(step_idx, self)    # returns GPU tensor (N,)
    # Write state from GPU data dict
    self.state_history[:, h, 0] = data_gpu['roll_lataccel'][:, step_idx]
    self.state_history[:, h, 1] = data_gpu['v_ego'][:, step_idx]
    self.state_history[:, h, 2] = data_gpu['a_ego'][:, step_idx]
    self.control_step(step_idx, actions)       # clip + store action
    self.sim_step(step_idx)                    # ONNX predict + clip + update current_lataccel
```

CPU path: same but with numpy arrays, and the controller receives `(step_idx, target, current_la, state_dict, future_plan)`.

**Cost computation (GPU):**
```python
target_gpu = data_gpu['target_lataccel'][:, 100:500]
pred_gpu = current_lataccel_history[:, 100:500]
lat_cost = (target_gpu - pred_gpu).pow(2).mean(dim=1) * 100
jerk = torch.diff(pred_gpu, dim=1) / DEL_T
jerk_cost = jerk.pow(2).mean(dim=1) * 100
total = lat_cost * 50 + jerk_cost
# Single GPU->CPU transfer: 3 vectors of size N
```

### 2.4 `CSVCache`

Loading 5000 CSVs via pandas takes ~6 seconds. Doing this every epoch is wasteful.

```python
class CSVCache:
    def __init__(self, all_csv_files):
        self._master = preload_csvs(all_csv_files)    # one-time: ~47s at startup
        self._rng_all = precompute_all_rng()           # deterministic per-file RNG

    def slice(self, subset_files):
        idxs = [self._file_to_idx[f] for f in subset_files]
        return {k: self._master[k][idxs] for k in keys}, self._rng_all[idxs]
```

After the one-time startup cost, each epoch's "CSV loading" is a numpy fancy-index operation: ~0.1 seconds.

`preload_csvs()` uses `ThreadPoolExecutor(max_workers=32)` for parallel CSV reads, then packs into `(N, T)` arrays. Short CSVs are edge-padded (repeat last row).

### 2.5 `compare_steppers.py` — Parity Verification

A 3-way comparison to prove the batched simulator produces identical trajectories:

1. **Reference** (`TinyPhysicsSimulator`): run N CSVs one at a time with greedy argmax physics
2. **Batched N=1**: run each CSV individually through `BatchedSimulator`
3. **Batched N=10**: run all N CSVs together in one batch

If N=1 matches reference exactly, the code logic is correct. If N=10 differs from N=1, that's ONNX batch-size floating-point non-determinism (acceptable).

Also tests stochastic parity: with per-episode seeded RNG, the sampling must produce identical trajectories.

### 2.6 GPU-Resident Data

```python
self.data_gpu = {}
for k in ('roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'steer_command'):
    self.data_gpu[k] = torch.from_numpy(data[k]).cuda()
```

Now `data_gpu['target_lataccel'][:, step_idx]` is a GPU tensor slice. No CPU involvement at any point during the rollout.

### 2.7 IOBinding

Default ONNX GPU path: numpy -> CPU staging -> GPU -> inference -> GPU -> CPU staging -> numpy. Every step. 580 times.

IOBinding binds pre-allocated GPU tensors directly:

```python
io = session.io_binding()
io.bind_input('states', 'cuda', 0, np.float32, shape, gpu_tensor.data_ptr())
io.bind_output('output', 'cuda', 0, np.float32, shape, out_gpu.data_ptr())
session.run_with_iobinding(io)
```

ONNX reads from and writes to persistent GPU tensors. Zero bus traffic per step.

---

## 3. The Observation Space (256-dim)

### 3.1 Dimension Breakdown

```
 Dims   Content                              Scaling
 ────   ───────                              ───────
 [0]    target_lataccel                      / 5.0   (S_LAT)
 [1]    current_lataccel                     / 5.0
 [2]    error (target - current)             / 5.0
 [3]    k_tgt (target curvature)             / 0.02  (S_CURV)
 [4]    k_cur (current curvature)            / 0.02
 [5]    k_tgt - k_cur (curvature error)      / 0.02
 [6]    v_ego                                / 40.0  (S_VEGO)
 [7]    a_ego                                / 4.0   (S_AEGO)
 [8]    roll_lataccel                        / 2.0   (S_ROLL)
 [9]    prev_action (h_act[-1])              / 2.0   (S_STEER)
 [10]   error_integral (mean of last 20 errors * DEL_T)  / 5.0
 [11]   target derivative (fplan_lat0 - target) / DEL_T  / 5.0
 [12]   lataccel derivative (current - prev_lat) / DEL_T / 5.0
 [13]   action derivative (prev_act - prev_prev_act) / DEL_T / 2.0
 [14]   friction circle usage: sqrt(current^2 + a_ego^2) / 7.0
 [15]   friction headroom: max(0, 1 - friction_usage)

 [16:36]   last 20 actions (h_act)            / 2.0
 [36:56]   last 20 lataccels (h_lat)          / 5.0

 [56:106]  future target_lataccel (50 steps)  / 5.0
 [106:156] future roll_lataccel (50 steps)    / 2.0
 [156:206] future v_ego (50 steps)            / 40.0
 [206:256] future a_ego (50 steps)            / 4.0

 Total: 16 + 20 + 20 + 50*4 = 256 dimensions
```

Everything is clipped to `[-5, 5]` after scaling.

### 3.2 Why Curvature Encoding

Curvature is the road geometry, independent of speed:

```python
def _curv(lat, roll, v):
    return (lat - roll) / max(v * v, 1.0)
```

Physics: lateral acceleration = v^2 * curvature + roll_component. So curvature = (lat - roll) / v^2.

At 30 km/h, a target_lataccel of 2.0 means a tight turn. At 120 km/h, the same 2.0 means a gentle curve. The raw number is meaningless without speed context. Curvature captures the geometry directly.

The moment curvature was added to the observation, BC cost dropped from 130 to 75. One line of physics. This is the "cheap meal" from the No Free Lunch theorem — guessing the structure of the problem correctly.

### 3.3 Scaling Constants

Each scaling constant is chosen so the typical range of that feature maps to roughly [-1, 1]:

| Constant | Value | Typical range of raw feature |
|---|---|---|
| `S_LAT = 5.0` | Lateral accel range is [-5, 5] | So /5 maps to [-1, 1] |
| `S_STEER = 2.0` | Steering range is [-2, 2] | So /2 maps to [-1, 1] |
| `S_VEGO = 40.0` | Typical highway speed ~30-40 m/s | So /40 maps to ~[0, 1] |
| `S_AEGO = 4.0` | Typical longitudinal accel [-4, 4] | So /4 maps to [-1, 1] |
| `S_ROLL = 2.0` | Roll lataccel typically [-2, 2] | So /2 maps to [-1, 1] |
| `S_CURV = 0.02` | Curvature typically [-0.02, 0.02] | So /0.02 maps to [-1, 1] |

Bad scaling was an early bug. In the first attempts, `v_ego` (~33) was 100x larger than `error` (~0.1) and 330,000x larger than curvatures (~0.0001). The network couldn't learn because gradient updates were dominated by the largest features.

### 3.4 The 16 Core Features — Line by Line

```python
core = np.array([
    target_lataccel / S_LAT,              # [0] what we want
    current_lataccel / S_LAT,             # [1] what we have
    error / S_LAT,                        # [2] the gap (redundant but helpful)
    k_tgt / S_CURV,                       # [3] road curvature we need to match
    k_cur / S_CURV,                       # [4] curvature we're actually on
    (k_tgt - k_cur) / S_CURV,            # [5] curvature error (speed-invariant)
    state.v_ego / S_VEGO,                # [6] how fast we're going
    state.a_ego / S_AEGO,                # [7] how fast speed is changing
    state.roll_lataccel / S_ROLL,         # [8] road bank angle contribution
    h_act[-1] / S_STEER,                 # [9] what we steered last step
    error_integral / S_LAT,              # [10] accumulated tracking error (like PID I-term)
    (fplan_lat0 - target) / DEL_T / S_LAT,  # [11] how fast the target is changing
    (current - h_lat[-1]) / DEL_T / S_LAT,  # [12] lataccel rate of change
    (h_act[-1] - h_act[-2]) / DEL_T / S_STEER,  # [13] steering rate of change
    fric,                                 # [14] friction circle usage (combined lat + lon demand)
    max(0.0, 1.0 - fric),               # [15] friction margin remaining
], dtype=np.float32)
```

**Error integral** (dim 10): `mean(last_20_errors) * DEL_T`. Acts like a PID integral term — it accumulates bias. If the controller consistently undershoots, this grows positive and biases the output upward.

**Friction circle** (dims 14-15): `sqrt(current_lataccel^2 + a_ego^2) / 7.0`. Measures how much of the tire's grip budget is being used. When close to 1.0, the car is at the limits of adhesion — steering aggressively here risks instability. The headroom term `1 - friction` tells the network how much spare grip is available.

**Target derivative** (dim 11): `(fplan_lat0 - target) / DEL_T`. How fast the target lateral acceleration is changing. Positive means the target is increasing — the controller should start steering into the coming curve now, not wait for the error to appear.

### 3.5 History Buffers

```python
self._h_act   = [0.0] * 20    # last 20 steering actions
self._h_lat   = [0.0] * 20    # last 20 lateral accelerations
self._h_v     = [0.0] * 20    # last 20 velocities
self._h_a     = [0.0] * 20    # last 20 longitudinal accelerations
self._h_roll  = [0.0] * 20    # last 20 roll lataccel values
self._h_error = [0.0] * 20    # last 20 tracking errors
```

Ring buffer pattern: `self._h_act = self._h_act[1:] + [new_value]`. Shift left, append right.

Only `h_act` and `h_lat` go into the observation (dims 16-55). The others (`h_v`, `h_a`, `h_roll`) are used internally for MPC context building. `h_error` is used to compute the error integral.

### 3.6 Future Plan

```python
_future_raw(future_plan, 'lataccel', target_lataccel) / S_LAT    # 50 dims
_future_raw(future_plan, 'roll_lataccel', state.roll_lataccel) / S_ROLL  # 50 dims
_future_raw(future_plan, 'v_ego', state.v_ego) / S_VEGO          # 50 dims
_future_raw(future_plan, 'a_ego', state.a_ego) / S_AEGO          # 50 dims
```

`_future_raw()` handles edge cases:
- If `len(vals) >= 50`: take first 50
- If `0 < len(vals) < 50`: edge-pad (repeat last value)
- If `vals` is None or empty: fill with fallback (current value)

The future plan is raw values (not curvatures). Early experiments used future curvatures (49 dims), but the final 256-dim design uses all four future channels separately, giving the network more information to work with.

---

## 4. The Neural Network Architecture

### 4.1 `ActorCritic` Module

```python
class ActorCritic(nn.Module):
    def __init__(self):
        # Actor: 256 -> 256 -> 256 -> 256 -> 2  (4 layers, ReLU, output is alpha_raw + beta_raw)
        a = [nn.Linear(256, 256), nn.ReLU()]
        for _ in range(3):
            a += [nn.Linear(256, 256), nn.ReLU()]
        a.append(nn.Linear(256, 2))
        self.actor = nn.Sequential(*a)

        # Critic: 256 -> 256 -> 256 -> 256 -> 1  (4 layers, ReLU, output is state value)
        c = [nn.Linear(256, 256), nn.ReLU()]
        for _ in range(3):
            c += [nn.Linear(256, 256), nn.ReLU()]
        c.append(nn.Linear(256, 1))
        self.critic = nn.Sequential(*c)
```

Total parameters: ~530K (actor: ~265K, critic: ~265K). Small enough to run at 10 Hz per episode during inference.

### 4.2 Orthogonal Initialization

```python
def _ortho_init(module, gain=np.sqrt(2)):
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.zeros_(module.bias)

# Hidden layers: gain = sqrt(2) (standard for ReLU)
# Actor output: gain = 0.01 (near-zero initial actions -> stable start)
# Critic output: gain = 1.0 (standard)
```

The actor output gain of 0.01 is crucial. It means the initial policy outputs near-zero deltas — the car doesn't steer. This is safe because during the warmup period (steps 0-99), the controller's output is overridden anyway. Once control starts, the policy makes small corrections rather than wild swings.

### 4.3 Beta Distribution

The actor outputs 2 values: `alpha_raw` and `beta_raw`. These become Beta distribution parameters:

```python
alpha = softplus(alpha_raw) + 1.0    # always > 1.0
beta  = softplus(beta_raw) + 1.0     # always > 1.0
```

The `+1.0` ensures `alpha, beta > 1`, which gives a unimodal distribution (single peak, no weird bimodal shapes).

The Beta distribution has support on `(0, 1)`. To map to steering delta range `[-1, 1]`:

```python
# Training (stochastic):
x = Beta(alpha, beta).sample()    # in (0, 1)
raw = 2*x - 1                     # in (-1, 1)

# Inference (deterministic):
raw = 2 * alpha/(alpha+beta) - 1  # Beta mean, mapped to (-1, 1)
```

Then `raw` is scaled and clipped to get the actual steering delta.

### 4.4 Why Beta Over Gaussian

**Gaussian with clipping:**
- Sample from N(mu, sigma), then clip to [-1, 1]
- Problem: if mu=0.8 and sigma=0.5, many samples land above 1.0 and get clipped to 1.0
- The gradient for "I picked 5.0" and "I picked 1.0" are both attributed to action 1.0 — confused learning
- Wasted exploration: samples outside [-1, 1] provide no useful information

**Beta distribution:**
- Samples are already in (0, 1) by definition
- No clipping, no wasted samples, no confused gradients
- The distribution can be sharp (high concentration) or broad (low concentration) and always stays in bounds
- Natural fit for bounded action spaces

**The BC-Beta bug:** When pretraining with behavioral cloning using MSE loss, only the mean `alpha/(alpha+beta)` is matched to the target. The concentration `alpha+beta` has zero gradient incentive to increase. Result: sigma stays at ~0.45 after BC. Fix: use NLL loss `(-Beta(alpha, beta).log_prob(target))` which trains both the mean and the concentration, shrinking sigma properly.

---

## 5. Delta Actions

### 5.1 The Parameterization

```python
delta  = clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)    # raw in [-1,1], delta in [-0.25, 0.25]
action = clip(prev_action + delta, STEER_RANGE[0], STEER_RANGE[1])  # in [-2, 2]
```

With `DELTA_SCALE = 0.25` and `MAX_DELTA = 0.5`:
- Network output `raw = +1.0` means "increase steering by 0.25"
- Network output `raw = -1.0` means "decrease steering by 0.25"
- Maximum possible change per step: 0.5 (but typical range is ±0.25)

### 5.2 Why Delta Actions Are Essential

The simulator is autoregressive: each prediction feeds back as input. Consider what happens with exploration noise:

**Absolute actions with noise:**
- Step 100: policy says steer=0.5, exploration noise adds +0.3, actual steer=0.8
- ONNX model sees 0.8, predicts a lataccel that's wrong for this trajectory
- Step 101: wrong lataccel feeds back, wrong next prediction, wrong step 102...
- The whole rollout goes bad from one noisy action

**Delta actions with noise:**
- Step 100: prev_steer=0.5, policy says delta=0.1, noise adds +0.05, actual delta=0.15, steer=0.65
- Next step: prev_steer=0.65, policy says delta=-0.05, steer=0.60
- The previous action anchors the output. Noise causes a small wiggle, then the policy corrects

Delta actions make exploration safe in autoregressive environments. The noise stays local instead of cascading.

### 5.3 The Constants

`DELTA_SCALE=0.25` and `MAX_DELTA=0.5` were found empirically. The commit messages show "revert to 47 config" after trying other values. Earlier experiments used `DELTA_SCALE=0.1, MAX_DELTA=0.3` (exp048) which was more conservative. The final values are more aggressive — they allow the policy to change steering faster, which helps track rapid target changes.

The `MAX_DELTA=0.5` also implicitly limits jerk. If steering can change by at most 0.5 per step (0.1s), that's 5.0/s rate limit. This prevents the catastrophic steering jumps that would produce huge jerk cost.

---

## 6. Training Pipeline

### 6.1 Behavioral Cloning Pretrain

BC gives the policy a warm start by imitating the CSV's steer commands.

**Data extraction (`_bc_worker`):**

For each CSV, iterate steps `CONTEXT_LENGTH` to `CONTROL_START_IDX` (steps 20-99):
```python
# Build the observation exactly as during RL
obs = build_obs(target, current=target, state, future_plan, h_act, h_lat, ...)
# Compute what delta the CSV steer command implies
delta = csv_steer[step] - prev_action
raw_target = clip(delta / DELTA_SCALE, -1, 1)    # map back to Beta support
```

Note: BC assumes perfect tracking (`current = target`), which is true during the warmup period. BC only trains on steps 20-99 — the warmup window before controller output matters.

**NLL loss:**
```python
alpha, beta = ac.beta_params(obs_batch)
x_target = (raw_target + 1) / 2              # map [-1,1] to (0,1) for Beta
x_target = clamp(x_target, 1e-6, 1-1e-6)    # avoid log(0)
loss = -Beta(alpha, beta).log_prob(x_target).mean()
```

This trains both the mean (what action to take) and the concentration (how confident to be). MSE loss only trains the mean — this was the BC-Beta bug that kept sigma at 0.45.

**Optimizer:** AdamW (weight decay 1e-4), cosine annealing LR from `BC_LR=0.01` down to 0, gradient clipping at 2.0. Default 20 epochs, batch size 8192.

### 6.2 PPO — The Core Algorithm

**Hyperparameters:**

| Parameter | Value | Why |
|---|---|---|
| `PI_LR` | 3e-4 | Standard PPO actor learning rate |
| `VF_LR` | 3e-4 | Same as actor (earlier experiments used 1e-3 for critic) |
| `GAMMA` | 0.95 | Short-horizon discounting — matches the ~0.3s lookahead insight from exp047 |
| `LAMDA` | 0.9 | GAE lambda for bias-variance tradeoff |
| `K_EPOCHS` | 4 | Number of PPO update passes per batch |
| `EPS_CLIP` | 0.2 | PPO clipping range [0.8, 1.2] |
| `VF_COEF` | 1.0 | Value function loss coefficient |
| `ENT_COEF` | 0.001 | Entropy bonus (small — prevent premature convergence) |
| `ACT_SMOOTH` | 5.0 | Penalty on |delta_action|^2 in reward |
| `MINI_BS` | 100,000 | Mini-batch size for SGD |
| `CRITIC_WARMUP` | 4 | First 4 epochs: train critic only, freeze actor |

### 6.3 Reward Function

```python
def compute_rewards(traj):
    lat  = (target - current)^2 * 100 * LAT_ACCEL_COST_MULTIPLIER    # 50x tracking
    jerk = diff(current, prepend=current[0]) / DEL_T                   # d(lataccel)/dt
    act_d = diff(action, prepend=action[0]) / DEL_T                    # d(steer)/dt
    return -(lat + jerk^2 * 100 + act_d^2 * ACT_SMOOTH) / 500.0
```

This matches the eval cost function exactly (with the 50x multiplier and correct jerk formula). The `/500` normalization brings rewards to a manageable scale. The `ACT_SMOOTH` term penalizes rapid steering changes, encouraging smooth control.

**The reward bug history:** Early training scripts had three bugs in the reward:
1. Missing `LAT_ACCEL_COST_MULTIPLIER` (50x) — network didn't know to prioritize tracking
2. Jerk computed as `(cur-prev)^2` instead of `((cur-prev)/0.1)^2` — off by 100x
3. BC trained on pre-tanh output but eval used tanh — action space mismatch

These bugs explain months of "PPO can't beat BC." The network was optimizing the wrong objective.

### 6.4 Generalized Advantage Estimation (GAE)

```python
def gae(all_r, all_v, all_d):
    # Normalize rewards by running return std
    all_r = [r / running_std for r in all_r]

    for each episode (r, v, d):
        T = len(r)
        g = 0.0
        for t in range(T-1, -1, -1):      # backward pass
            next_value = v[t+1] if t < T-1 else 0.0
            delta = r[t] + GAMMA * next_value * (1 - done[t]) - v[t]
            g = delta + GAMMA * LAMDA * (1 - done[t]) * g
            advantage[t] = g
        returns = advantage + v
```

**GPU vectorized GAE (`gae_gpu`):**
```python
# rew_2d, val_2d, done_2d are (N, S) CUDA tensors
g = torch.zeros(N)
for t in range(S-1, -1, -1):
    nv = val_2d[:, t+1] if t < S-1 else zeros
    mask = 1.0 - done_2d[:, t]
    delta = rew_2d[:, t] + GAMMA * nv * mask - val_2d[:, t]
    g = delta + GAMMA * LAMDA * mask * g
    adv[:, t] = g
```

This vectorizes across episodes (N dimension) while keeping the time dimension sequential (required for correct GAE).

### 6.5 Reward Normalization

`RunningMeanStd` uses Welford's online algorithm to track running mean and variance of returns:

```python
class RunningMeanStd:
    def update(self, x):
        # Welford's: stable incremental mean/variance update
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        # ... variance update ...
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # GPU optimization: compute moments on GPU, transfer 2 scalars
        # instead of transferring the entire (N, S) reward tensor
```

Rewards are divided by `running_std` before GAE computation. This is critical for stable PPO — without it, the advantage estimates have wildly different scales across training, and the policy gradient is unstable.

### 6.6 PPO Update

```python
def update(self, ...):
    # Old policy log-probs and values (frozen)
    with torch.no_grad():
        a_old, b_old = ac.beta_params(obs)
        old_lp = Beta(a_old, b_old).log_prob(x)
        old_val = ac.critic(obs).squeeze(-1)

    for _ in range(K_EPOCHS):         # 4 passes
        for idx in randperm(N).split(MINI_BS):    # mini-batches of 100K
            # Per-minibatch advantage normalization
            mb_adv = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)

            # Value loss: clipped Huber
            val = critic(obs[idx])
            v_clipped = old_val[idx] + (val - old_val[idx]).clamp(-10, 10)
            vf_loss = max(huber(val, ret, delta=10), huber(v_clipped, ret, delta=10)).mean()

            # Policy loss: clipped PPO objective
            alpha_cur, beta_cur = beta_params(obs[idx])
            lp = Beta(alpha_cur, beta_cur).log_prob(x[idx])
            ratio = exp(lp - old_lp[idx])
            pi_loss = -min(ratio * mb_adv, clip(ratio, 0.8, 1.2) * mb_adv).mean()

            # Entropy bonus
            ent = Beta(alpha_cur, beta_cur).entropy().mean()

            # Combined loss
            loss = pi_loss + 1.0 * vf_loss - 0.001 * ent
```

**Key design decisions:**

- **Per-minibatch advantage normalization:** Not global. Each mini-batch normalizes its own advantages. This reduces sensitivity to the overall advantage distribution.

- **Huber value loss (delta=10):** More robust than MSE for large value errors. The clipped value function prevents the critic from changing too fast.

- **Value function clipping (±10):** Prevents catastrophic value updates. The critic can't jump by more than 10 from its old prediction in a single update.

- **Separate optimizers:** Actor and critic have separate Adam optimizers. This allows different learning rates and prevents the value loss from dominating the actor gradients.

- **Gradient clipping at 0.5:** Both actor and critic gradients are clipped to max norm 0.5. Prevents instability from large gradients.

- **`zero_grad(set_to_none=True)`:** Saves a memset operation per zero_grad call. Small but adds up over 4 epochs x many mini-batches.

- **GPU randperm:** `torch.randperm(N, device=obs_t.device)` generates the permutation on GPU, avoiding hundreds of implicit CPU->GPU index transfers per update.

### 6.7 Critic Warmup

```python
critic_only = epoch < CRITIC_WARMUP    # first 4 epochs
if critic_only:
    # Only update critic, don't touch actor
    vf_opt.zero_grad()
    vf_loss.backward()
    clip_grad_norm_(critic.parameters(), 0.5)
    vf_opt.step()
```

The critic needs a reasonable baseline before actor updates begin. Without warmup, the advantage estimates are garbage (critic is random), and the first actor updates are essentially random too. Warming up the critic for 4 epochs gives it time to learn the value landscape.

When resuming from a checkpoint, warmup is skipped (`warmup_off = CRITIC_WARMUP`) because the critic already has a good baseline.

### 6.8 The Training Loop

```python
def train():
    ctx = Ctx()    # creates ActorCritic, PPO, loads CSVs, creates pool/cache

    if RESUME:
        load checkpoint (weights + optimizer + RunningMeanStd state)
    else:
        pretrain_bc(ctx.ac, all_csvs)

    baseline = evaluate(ctx, val_files)
    save_best()

    for epoch in range(MAX_EP):
        batch = random.sample(train_files, CSVS_EPOCH)
        # Rollout: run batch through simulator with current policy
        # Update: PPO update on collected data
        # Eval: every EVAL_EVERY epochs, evaluate on held-out val set
```

**`Ctx` class:** Holds all training state:
- `ac`: the ActorCritic model
- `ppo`: the PPO trainer (optimizers, running stats)
- `mdl_path`: path to ONNX model
- `tr_f`: training CSV files (shuffled)
- `va_f`: validation CSV files (first 100)
- `best`: best validation cost so far
- `ort_session`: shared ONNX session (GPU mode)
- `csv_cache`: CSVCache instance (GPU mode)
- `pool`: multiprocessing pool (CPU mode)

**Checkpoint saving:**
```python
torch.save({
    'ac': ac.state_dict(),
    'pi_opt': pi_opt.state_dict(),
    'vf_opt': vf_opt.state_dict(),
    'ret_rms': {'mean': rms.mean, 'var': rms.var, 'count': rms.count},
}, BEST_PT)
```

Saving optimizer state + RunningMeanStd state enables seamless resume with identical training dynamics.

---

## 7. The Batched Rollout (CPU + GPU Paths)

### 7.1 CPU Path (`_batched_rollout_cpu`)

```python
def _batched_rollout_cpu(sim, ac, N, T, deterministic):
    # Pre-allocate numpy ring buffers
    h_act   = np.zeros((N, 20), np.float64)
    h_lat   = np.zeros((N, 20), np.float32)
    h_error = np.zeros((N, 20), np.float32)
    obs_buf = np.empty((N, 256), np.float32)

    # Pre-allocate training data collection arrays
    all_obs = np.empty((max_steps, N, 256), np.float32)
    all_raw = np.empty((max_steps, N), np.float32)
    all_val = np.empty((max_steps, N), np.float32)

    def controller_fn(step_idx, target, current_la, state_dict, future_plan):
        # 1. Update error history
        # 2. Build obs in obs_buf (numpy ops on (N,) arrays)
        # 3. Forward pass: obs -> torch -> GPU -> actor -> beta params -> sample -> CPU -> numpy
        # 4. Compute delta action, apply to prev action
        # 5. Update ring buffers
        # 6. Store training data
        return action    # (N,) numpy array

    cost_dict = sim.rollout(controller_fn)

    # Post-rollout: transpose (steps, N, obs_dim) -> (N, steps, obs_dim)
    # Compute rewards, dones, pack into per-episode tuples
    return [(obs_i, raw_i, rew_i, val_i, done_i, cost_i) for i in range(N)]
```

The CPU path has 3 CPU<->GPU round-trips per step:
1. `current_la = sim.current_lataccel` (already CPU in this path)
2. `obs_t = torch.from_numpy(obs_buf).to(device)` — CPU -> GPU
3. `raw = (2*x - 1).cpu().numpy()` — GPU -> CPU

### 7.2 GPU Path (`_batched_rollout_gpu`)

```python
def _batched_rollout_gpu(sim, ac, N, T, deterministic):
    # All buffers are CUDA tensors
    h_act   = torch.zeros((N, 20), dtype=torch.float64, device='cuda')
    h_act32 = torch.zeros((N, 20), dtype=torch.float32, device='cuda')
    h_lat   = torch.zeros((N, 20), dtype=torch.float32, device='cuda')
    h_error = torch.zeros((N, 20), dtype=torch.float32, device='cuda')
    obs_buf = torch.empty((N, 256), dtype=torch.float32, device='cuda')

    # Training data: also on GPU
    all_obs = torch.empty((max_steps, N, 256), dtype=torch.float32, device='cuda')
    all_raw = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
    all_val = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')

    def controller_fn(step_idx, sim_ref):
        # Everything on GPU — zero CPU touches
        target = dg['target_lataccel'][:, step_idx]      # GPU slice
        current_la = sim_ref.current_lataccel              # GPU tensor
        v_ego = dg['v_ego'][:, step_idx]                  # GPU slice

        # Build obs entirely on GPU
        obs_buf[:, 0] = target / S_LAT
        obs_buf[:, 1] = current_la / S_LAT
        # ... all 256 dims built via torch ops ...
        obs_buf.clamp_(-5.0, 5.0)

        # Forward pass: already on GPU, no transfer
        with torch.inference_mode():
            alpha, beta = ac.beta_params(obs_buf)
            val = ac.critic(obs_buf).squeeze(-1)

        # Sample or take mean
        raw = 2.0 * Beta(alpha, beta).sample() - 1.0      # GPU tensor
        delta = (raw.double() * DELTA_SCALE).clamp(-MAX_DELTA, MAX_DELTA)
        action = (h_act[:, -1] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        # Store training data (GPU -> GPU, no transfer)
        all_obs[step_ctr] = obs_buf
        all_raw[step_ctr] = raw
        all_val[step_ctr] = val

        return action    # GPU tensor (N,)

    cost_dict = sim.rollout(controller_fn)

    # Rewards computed on GPU
    rew = (-(lat_r + jerk_r**2 * 100 + act_dr**2 * ACT_SMOOTH) / 500.0).float()

    # Return pre-flattened GPU tensors — no per-episode split needed
    return dict(obs=obs_flat, raw=raw_flat, val_2d=val_2d, rew=rew, done=dones, costs=costs)
```

The GPU path eliminates all CPU<->GPU transfers during the rollout. The controller closure reads `data_gpu` directly, builds observations with torch ops, and returns a GPU tensor. The sim receives it as a GPU tensor, clips it, stores it in GPU history, runs GPU ONNX prediction.

The only CPU touch is at the very end: `costs.cpu().numpy()` for logging.

**Dual dtype precision:** `h_act` is float64 (matches the simulator's action_history precision) while `h_act32` is a float32 shadow copy used for observation building (the actor network is float32). This avoids 5000-way float64->float32 casts every step.

### 7.3 Post-Rollout Data Reshaping

GPU path returns a flat dict:
```python
obs_flat = all_obs[:S].permute(1, 0, 2).reshape(-1, OBS_DIM)  # (N*S, 256)
raw_flat = all_raw[:S].T.reshape(-1)                            # (N*S,)
val_2d   = all_val[:S].T                                        # (N, S)
```

The `permute(1, 0, 2)` swaps from (steps, episodes, obs_dim) to (episodes, steps, obs_dim), then reshape flattens episodes and steps together.

PPO receives this dict and calls `gae_gpu` directly on the 2D tensors — no per-episode splitting or concatenation needed.

---

## 8. Distributed Training (MacBook Cluster + Cloud)

### 8.1 The MacBook Cluster Architecture

Three MacBooks connected via Thunderbolt USB-C cables:
- Main Mac (M4 Pro): runs the training loop
- Remote 1 (M4 Air): runs `remote_server.py` on port 5555
- Remote 2 (friend's Mac): runs `remote_server.py` on another port

```
Main Mac ──USB-C──> Mac Air (169.254.159.243:5555)
    └──WiFi/USB-C──> Friend's Mac (192.168.1.42:5555)
```

### 8.2 TCP Protocol

**Persistent connection:** One TCP socket per remote, opened once and reused across epochs. No per-epoch SSH handshake.

**Request format (pickle blob):**
```python
payload = pickle.dumps({
    'mode': 'train',          # or 'eval'
    'ckpt': checkpoint_bytes,  # raw bytes of best_model.pt
    'csvs': ['data/00000.csv', 'data/00001.csv', ...]
})
```

**Response format (NPZ):**
```python
# For training:
np.savez(buf, obs=obs_all, raw=raw_all, rew=rew_all,
         val=val_all, done=done_all, costs=costs, ep_lens=ep_lens)

# For eval:
np.savez(buf, costs=costs)
```

**Wire protocol:**
```python
def _tcp_send(sock, data):
    sock.sendall(struct.pack('>I', len(data)))    # 4-byte big-endian length header
    sock.sendall(data)

def _tcp_recv(sock):
    header = recvall(sock, 4)
    length = struct.unpack('>I', header)[0]
    return recvall(sock, length)
```

### 8.3 Load Balancing

Naive 50/50 split was bad — the M4 Air is ~40% slower due to passive cooling and efficiency cores. Main Mac finishes in 40s and sits idle waiting for Air to finish in 60s.

```python
# FRAC = "local_weight:remote1_weight:remote2_weight"
# FRAC=1.4:1 means local gets 58%, remote gets 42%
# FRAC=3.5:3.5:3 means local=35%, remote1=35%, remote2=30%

FRAC_LOCAL = frac_parts[0] / sum(frac_parts)
FRAC_REMOTES = [frac_parts[i+1] / sum(frac_parts) for i in range(n_remotes)]
```

`_split_files()` allocates CSVs proportionally:
```python
def _split_files(all_files):
    N = len(all_files)
    remote_slices = []
    offset = 0
    for frac in FRAC_REMOTES:
        n_r = int(N * frac)
        remote_slices.append(all_files[offset:offset+n_r])
        offset += n_r
    local_files = all_files[offset:]
    return local_files, remote_slices
```

Remote rollouts run in parallel threads:
```python
for ri, r_files in enumerate(remote_slices):
    def _do(idx=ri, csvs=r_csvs):
        remote_results[idx] = _remote_request(idx, ckpt_path, csvs, mode='train')
    threading.Thread(target=_do).start()
# Meanwhile, local rollout runs on main thread
# Join all threads, merge results
```

### 8.4 Thermal Wall

The fanless M4 Air thermal throttled after 30+ minutes:
```
E 0   45s    <- cold
E 25  55s    <- warm
E 50  65s    <- throttled
```

Rubbing ice cubes wrapped in cloth on the Air helped temporarily. The cluster got the policy from random to val ~55 before cloud compute was needed.

### 8.5 Cloud Migration

**Google Cloud 48-core CPU:** The `gcloud-cpu-run.txt` log shows:
- 48 workers, 5000 CSVs per epoch
- 35s rollout + 20s PPO update = 55s per epoch
- Training from val 50.1 down to val 49.7 over 47 epochs
- Total cost on GCloud: ~$2-3 for the compute instance

**Vast.ai RTX 5060 Ti:** The GPU batched simulator made multi-machine approaches irrelevant. All 5000 episodes in a single GPU batch. One ONNX call per timestep. The "cluster" became a single GPU tensor.

---

## 9. The MPC System (Inference-Time)

### 9.1 Overview

MPC (Model Predictive Control) uses the ONNX physics model as a forward simulator at inference time. Instead of just taking the policy's action, it:
1. Proposes multiple action candidates
2. Simulates each candidate forward through the ONNX model
3. Scores each trajectory with the real cost function
4. Applies the best first action

This reliably improves cost by 2-4 points. The controller supports three MPC modes via environment variables.

### 9.2 Newton Correction (`MPC=1`)

The simplest MPC mode. Single-step lookahead with a Newton-style correction.

```python
def _mpc_correct(self, action, current_lataccel, state, future_plan):
    # Build ONNX context from histories
    h_preds = array(self._h_lat + [current_lataccel])[-20:]
    h_states = array(zip(h_roll, h_v, h_a) + [current_state])[-20:]
    h_actions = array(self._h_act)[-20:]

    # 1-step ONNX prediction with the proposed action
    a_seq = concatenate([h_actions[1:], [action]])
    pred_1 = _onnx_expected(a_seq, h_states, h_preds)

    # Target: weighted average favoring current state (smooth correction)
    target_0 = future_plan.lataccel[0]
    x_star = (target_0 + 2.0 * current_lataccel) / 3.0

    # Newton correction: nudge action to reduce prediction error
    da = clip(MPC_K * (pred_1 - x_star), -MPC_MAX, MPC_MAX)
    corrected = clip(action - da, -2, 2)
```

`x_star = (target + 2*current) / 3` is a compromise target — 1/3 of the way from current to target. This prevents over-correction: if the policy is tracking well, the correction is small.

With `MPC_H > 1`, both the original and corrected actions are validated by unrolling H steps. The one with lower total cost wins:

```python
cost_orig = _unroll_cost(action, ...)
cost_corr = _unroll_cost(corrected, ...)
return corrected if cost_corr < cost_orig else action
```

### 9.3 Shooting MPC (`MPC_N > 0`)

The full shooting MPC. Environment variables: `MPC_N=16` (candidates), `MPC_H=5` (horizon).

**Step 1: Generate candidates from the policy.**
```python
alpha, beta = _beta_params(obs)             # policy's Beta parameters
samples = mpc_rng.beta(alpha, beta, size=N) # N samples from policy
samples[0] = alpha / (alpha + beta)         # slot 0 = deterministic mean

# Convert to steering actions
deltas = clip((2*samples - 1) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
actions_0 = clip(prev_steer + deltas, -2, 2)
```

**Slot 0 is always the policy mean.** This guarantees the MPC never picks something worse than the pure policy. If all other candidates are bad, slot 0 (the policy mean) wins.

The `_mpc_rng = np.random.RandomState(42)` is a decoupled RNG — independent of the simulator's global `np.random` state. This prevents MPC from consuming random values that would shift the simulator's sampling sequence.

**Step 2: Unroll each candidate H steps via ONNX.**

```python
for step in range(H):
    # Batched ONNX: predict next lataccel for all N candidates at once
    pred_la = _onnx_expected(action_seqs, states, predictions)  # (N,)

    # Score with real cost function
    costs += (target - pred_la)^2 * 100 * 50     # lataccel cost
    costs += ((pred_la - prev_la) / 0.1)^2 * 100  # jerk cost

    # Shift ONNX context (autoregressive)
    predictions = concat([predictions[:, 1:], pred_la[:, None]], axis=1)
    actions = concat([actions[:, 1:], cur[:, None]], axis=1)

    # Policy-rolled continuations for steps 1+
    if MPC_ROLL and step < H-1:
        # Run the policy forward to get realistic next actions
        obs_batch = _build_obs_batch(...)
        alpha, beta = ac.beta_params(obs_batch)
        next_samples = mpc_rng.beta(alpha, beta)
        next_deltas = clip(...)
        cur = clip(prev + next_deltas, -2, 2)
```

**Step 3: Pick best candidate.**
```python
return actions_0[argmin(costs)]    # first action of the best trajectory
```

### 9.4 `_onnx_expected()` — Expected Value Computation

Instead of sampling (which introduces noise), MPC computes the **expected** next lataccel:

```python
def _onnx_expected(p_actions, p_states, p_preds):
    # Tokenize predictions
    tokenized = tokenizer.encode(p_preds)               # (P, 20)
    # Build ONNX input
    states_in = concat([p_actions[:,:,None], p_states], axis=-1)  # (P, 20, 4)
    # Run ONNX
    logits = ort_session.run(None, {
        'states': states_in.float32,
        'tokens': tokenized.int64,
    })[0][:, -1, :]                                     # (P, 1024) last timestep

    # Softmax at temperature 0.8
    logits = logits / 0.8
    probs = softmax(logits)                              # (P, 1024)

    # Expected value: sum(prob * bin_center) for each candidate
    return sum(probs * tokenizer.bins, axis=-1)          # (P,)
```

This gives a deterministic, smooth prediction. Sampling would add noise that makes it hard to compare candidates fairly.

### 9.5 GPU Shooting (`_mpc_shoot_gpu`)

All the same logic but with CUDA tensors:
- IOBinding for ONNX: `io.bind_input('states', 'cuda', 0, np.float32, shape, tensor.data_ptr())`
- Tokenization via `torch.bucketize`
- Expected value: `(probs * bins_f32.unsqueeze(0)).sum(dim=-1)`
- Beta sampling still done on CPU (tiny N=16, not worth GPU overhead)
- Final selection: `actions_0[costs.argmin()].item()`

### 9.6 MPC Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MPC` | 0 | Enable Newton correction (1) |
| `MPC_K` | 0.2 | Newton correction gain |
| `MPC_MAX` | 0.1 | Max Newton correction magnitude |
| `MPC_H` | 1 | Lookahead horizon (steps) |
| `MPC_ROLL` | 0 | 1=policy-rolled unrolls, 0=zero-order hold |
| `MPC_N` | 0 | Shooting candidates (0=off, 16=recommended) |
| `LPF_ALPHA` | 0 | Low-pass filter: blend with prev action (0=off, 0.15=subtle) |
| `RATE_LIMIT` | 0 | Max steer change per step (0=off) |
| `VNORM` | 0 | Speed-normalize steer output |

### 9.7 The MPC Journey

MPC went through 4 phases:

1. **Full CEM MPC (exp046):** 100 random samples, H=30, 4 CEM iterations. Cost 157. Slow, proposals were random and uninformed.

2. **Local 1-step (exp046_v2):** 30 samples within radius 0.3 of PID output. Small improvement over PID.

3. **MPC + PPO (first attempt):** Failed. Both lataccel and jerk got worse. Root cause: off-by-one bug in ONNX context — predicting from stale context. Also: holding actions constant (zero-order hold) was wrong when candidates were close.

4. **Hybrid Policy-MPC Shooting (final):** Sample from the policy's own distribution. The policy has already learned good actions — the MPC just picks the best sample. Reliable 2-4 point improvement.

---

## 10. The 375x Speedup Journey

### 10.1 The Progression

| Step | What Changed | Epoch Time | Cumulative Speedup |
|---|---|---|---|
| Baseline | Sequential single-episode | ~50 min | 1x |
| Batched CPU | N episodes in lockstep, 10 workers | ~60s | 75x |
| MacBook Cluster | 3 Macs, TCP, FRAC splits | ~55s (3000 CSVs) | 3x data/time |
| GPU ONNX | CUDAExecutionProvider | ~45s | 120x |
| IOBinding + GPU tokenize | Zero-copy ONNX I/O | ~40s | 150x |
| GPU-resident histories | CUDA tensor histories | ~33s | 150x |
| CSVCache | Parse CSVs once at startup | ~20s | 150x (epoch: 150x) |
| GPU controller + data_gpu | All obs/ctrl/data on GPU | ~12s rollout | 250x |
| TensorRT FP16 | Fused kernels, engine caching | ~8s rollout | 375x |

### 10.2 Why Each Layer Matters

**Batching (75x):** The single biggest win. 5000 x 580 ONNX calls → 580 ONNX calls. Same compute, 5000x fewer kernel launches. The ONNX transformer handles batch natively.

**GPU ONNX (1.6x):** Swap `CPUExecutionProvider` for `CUDAExecutionProvider`. Single process, one session, no multiprocessing.

**IOBinding (1.25x):** Eliminate CPU<->GPU staging buffers per ONNX call. Bind pre-allocated GPU tensors directly as inputs/outputs.

**GPU tokenization:** `np.clip + np.digitize` on CPU → `torch.clamp + torch.bucketize` on GPU. Gotcha: inverted `right` flag semantics.

**GPU histories:** `action_history`, `state_history`, `current_lataccel_history` as CUDA tensors. Slicing produces GPU tensor views. ONNX reads directly. Prediction stays on GPU.

**CSVCache (halved epoch time):** CSV re-parsing was 6s/epoch. Parse once at startup, fancy-index per epoch → 0.1s/epoch.

**GPU controller (7x on ctrl time):** Eliminated 3 CPU<->GPU round-trips per step. All 256 obs dimensions built with torch ops on GPU. `ctrl: 9.5s → 1.3s`.

**TensorRT FP16 (2x on sim time):** Fused transformer layers, auto-tuned kernels, FP16 where safe. 30s first-time compilation, cached engine thereafter. `sim: 10s → 5s`.

### 10.3 What I Got Wrong

**Float32 decode bug:** Decoding tokens with float32 bins instead of float64. Over 500 autoregressive steps, the precision loss compounded. Score jumped from 49.7 to 59.2 — a 20% regression from one line of code.

**torch.bucketize vs np.digitize:** Inverted `right` semantics. `np.digitize(right=True)` = `torch.bucketize(right=False)`. Wrong flag → every token shifts by 1 → simulation diverges silently.

**RNG parity bug:** Pre-generated random values for all T steps instead of T-CL steps. Wasted CL random draws, shifted the entire sequence.

**The /venv incident:** Deleted `/venv` on a rented GPU to free disk space. That was the system Python. The virtualenv's symlink broke instantly. Had to rebuild the environment.

### 10.4 The Compounding Effect

Each optimization layer enabled the next:
- GPU-resident histories → enabled IOBinding (GPU tensors to bind)
- IOBinding → enabled zero-copy predict (no staging)
- `data_gpu` → enabled GPU controller (no CPU data reads)
- CSVCache → enabled fast reset (no per-epoch I/O)
- GPU controller → enabled GPU training data (no CPU arrays to cat)

Can't cherry-pick. They compound.

---

## 11. The Journey Narrative

### 11.1 Timeline

```
Dec 25, 2025    Fork the repo on Christmas Day. 6 commits.
                Pure PPO from scratch: cost 10,000+ ("basically random")
                8 CSVs per epoch, 10-dim observation
                Reward bugs: missing 50x multiplier, wrong jerk formula

Dec 26          Cleanup. BC from PID: ~88. PPO residual on PID: 72.
                Then 37 days of silence in git.

Jan 2026        (No commits, but thinking)
                MPC experiments: CEM H=50 -> cost 157 (can't beat PID)
                LunarLander solved: isolated PPO skills as invariant
                Neural atom (exp047): single neuron with decay weights -> 81.46

Feb 2           "checkpoint" — first commit in 37 days

Feb 9           "ppo shows promise" — PPO from scratch beating PID: 88 -> 80 -> 79

Feb 11          Curvature encoding: 130 -> 75 in one line
                256-dim obs, Beta distribution: 96 -> 72 -> 66
                "65.4 validation is a real breakthrough after months of work"
                Batched simulator born

Feb 12          NLL BC + reward normalization: 112 -> 63.8 in 11 epochs

Feb 13          Cosine LR decay: 57 -> 50 -> 47
                "I had to start it 8-9 times to reach 47 cost"

Feb 14          16 commits in one day. THE DAY.
                "fix ratio collapse" -> "fix log ratio clamp" -> "new best 43"
                MPC shooting: tried, reverted, tried different horizon

Feb 15          20+ commits. GPU optimization blitz.
                42.5 on 5000 routes. #3 on leaderboard.
```

### 11.2 The LunarLander Prologue

Before the controls challenge, there were too many unknowns: bad results could be broken PPO, wrong architecture, or insufficient data. Three suspects, entangled.

Decision: stop. Solve LunarLander first. Not because LunarLander matters, but because it isolates "PPO skills" as a variable. If LunarLander works in 100 epochs, PPO is not the suspect.

What transferred directly:
- Separate actor/critic optimizers
- Learnable state-independent log_std (before switching to Beta)
- GAE with advantage normalization
- Gradient clipping at 0.5
- The confidence that PPO works when implemented correctly

Key lesson from LunarLander: **state-dependent sigma is a trap.** It accelerates stochastic training by letting the network avoid hard states, but wrecks deterministic eval because `E[tanh(mu + sigma*epsilon)] != tanh(mu)`.

### 11.3 The Curvature Breakthrough

Cost was capped at 130-150 with raw future plan inputs. The moment curvature encoding was added:

```python
kappa = (lataccel - roll_lataccel) / max(v_ego^2, 1.0)
```

Cost dropped to 75 with just BC pretrain. One line of physics. The NFL theorem insight: "the world hands out cheap meals if you guess its structure correctly."

Why it works: lateral acceleration at 30 km/h and 120 km/h means completely different things for steering. Curvature is the road geometry — it's speed-invariant. The MLP can now learn a single mapping from curvature to steering, instead of learning separate mappings for every speed.

### 11.4 The Debugging Grind

**EPS_CLIP = 0:** The PPO clipping range was `[1-0, 1+0] = [1.0, 1.0]`. Every ratio was clamped to exactly 1.0. The actor never updated. PPO was literally a no-op while appearing to train normally (critic loss went down). This explains months of "PPO can't beat BC."

**Reward mismatch:** Three separate bugs in early reward functions. Missing 50x multiplier meant the network didn't know tracking matters more than smoothness. Wrong jerk formula (`(cur-prev)^2` vs `((cur-prev)/0.1)^2`) was off by 100x. BC pre-tanh vs eval post-tanh meant the training action space didn't match eval.

**BC-Beta MSE bug:** MSE loss only trains `alpha/(alpha+beta)` to match the target. Zero gradient on concentration. Sigma stays at 0.45 instead of shrinking. Fix: NLL loss that trains both mean and concentration.

**State-dependent sigma:** The network learned to inflate sigma on hard states, avoiding them during training. E[tanh(mu + sigma*epsilon)] != tanh(mu). At eval time (deterministic), actions are too aggressive because the network compensated for noise during training.

### 11.5 Key Philosophies

**Fix one invariant at a time:** The throughline of the entire project.
- LunarLander → fixes PPO skills
- PID baseline → fixes the cost floor
- One neuron (exp016) → fixes "can learning work at all" (yes: 3 weights recover PID gains)
- exp046's 12 ablations → fixes which feedforward structure matters
- Delta actions → fixes action parameterization
- Beta distribution → fixes distribution choice
- NLL for BC → fixes sigma initialization

Each fix eliminates one suspect. The remaining improvement is always attributable.

**The Bitter Lesson (Sutton, 2019):** "General methods that leverage computation are ultimately the most effective." This explains:
- Why PPO over hand-tuned PID+feedforward (exp046 hit 81, ceiling)
- Why scale data (8 → 5000 CSVs/epoch) instead of engineer features
- Why batch and GPU instead of clever algorithms
- Why the MacBook cluster, then cloud, then GPU
- The winning MPC is just "sample 16 from policy, simulate all, pick best" — pure search

**No Free Lunch:** "There is no free lunch — but the world hands out cheap meals if you guess its structure correctly." Curvature was the cheap meal.

### 11.6 Score Progression

```
Dec 25:   PPO from scratch         10,000+   (8 CSVs/epoch, 10D obs)
Dec 25:   PPO residual on PID          72    (best early result)
Jan:      MPC (CEM, H=50)             157    (exp045)
Jan:      Neural atom (tau=0.9)         81    (exp047, single neuron!)
Jan:      PID baseline                  85    (exp046)
Feb 9:    PPO from scratch (clean)  88->79    (107D obs, 5-layer, exp049)
Feb 11:   Physics-first obs        96->66     (256D, Beta, exp050)
Feb 12:   NLL BC + reward norm    112->63.8   (11 epochs)
Feb 13:   Cosine LR decay          57->47     ("had to restart 8-9 times")
Feb 14:   Fine-tuning              47->43     (16 commits, ratio fixes)
Feb 15:   5000-route eval           42.5      (#3 on leaderboard)
```

### 11.7 Observation Space Evolution

```
 3D    exp016, exp040       [error, integral, diff]
10D    early PPO            [error, lataccel, v_ego, a_ego, curvatures]
55D    bc.py, exp031        [error, roll, v_ego, a_ego, 50 future lataccels]
56D    parallel pipeline    [error, integral, v_ego, curvature, 50 future curvs]
54D    exp048 (THE TURN)    [error, v_ego, prev_act, k_now, 50 future k]
107D   exp049               [error, integral, v_ego, a_ego, roll, 50 k, 49 dk]
256D   exp050 (FINAL)       [16 core + 40 hist + 200 future]
```

### 11.8 Action Space Evolution

```
Direct steering:    action = network(obs)                    (bc.py)
Tanh squashing:     action = tanh(raw) * 2                   (exp031)
PID + residual:     action = PID(error) + 0.1 * network     (exp032)
PID + feedforward:  action = PID + linear(future_features)   (exp046)
Delta actions:      action = prev + clip(raw*0.25, ±0.5)    (exp048+, WINNER)
```

---

## 12. Potential Interview Questions and Answers

### Architecture & Design

**Q: Why did you use a Beta distribution instead of Gaussian?**

A: The steering delta lives in [-1, 1]. A Gaussian with clipping wastes exploration outside the bounds and confuses gradients — samples at 5.0 and 1.0 both become 1.0 after clipping, but the log-probability gradients are different. The Beta distribution has support on (0, 1) natively. No clipping, no wasted samples, no confused gradients. I map to [-1, 1] via `2*x - 1`. The BC pretrain also benefits: NLL loss on Beta trains both the mean and the concentration (how confident the policy is), whereas MSE only trains the mean and leaves sigma at ~0.45.

**Q: Why delta actions instead of absolute steering?**

A: The simulator is autoregressive — each prediction feeds back as input. With absolute actions, one noisy sample at step 100 corrupts the state, which corrupts step 101, which corrupts 102... the whole rollout goes bad. With delta actions, noise causes a small wiggle — the previous action anchors the output. Noise stays local. This is the single most important architectural decision for stable RL in autoregressive environments.

**Q: Walk me through the 256-dim observation vector.**

A: 16 core features (target, current, error, two curvatures and their difference, velocity, longitudinal accel, road roll, previous action, error integral, three derivatives, friction circle usage and headroom). Then 20 past actions and 20 past lataccels as history. Then 4 x 50 = 200 future plan values (target_lataccel, roll, v_ego, a_ego for the next 5 seconds). Everything is scaled so typical values are in [-1, 1] and clipped to [-5, 5].

**Q: Why curvature encoding? What is it?**

A: Curvature is road geometry independent of speed: `kappa = (lataccel - roll) / v^2`. At 30 km/h, a target_lataccel of 2.0 means a tight turn. At 120 km/h, the same 2.0 is a gentle curve. Raw lateral acceleration is meaningless without speed context. Curvature captures what the road is actually doing. The moment I added it, BC cost dropped from 130 to 75. One line of physics.

**Q: Why 4 layers for both actor and critic?**

A: The observation space is already well-engineered — curvatures, normalized features, history. The network doesn't need to extract complex features, just learn the mapping. 4 layers with 256 hidden gives enough capacity without overfitting. Earlier experiments with 5 layers (exp049) and deeper networks showed no improvement. The orthogonal initialization with small actor output gain (0.01) is important — the policy starts by outputting near-zero deltas, which is safe.

**Q: Why separate optimizers for actor and critic?**

A: Decouples learning dynamics. The critic can learn faster or slower than the actor without interfering. In early experiments with a shared optimizer, the value loss dominated and the actor barely moved. Separate optimizers with separate learning rates and gradient clipping give independent control.

### Training

**Q: How does the reward function work?**

A: `reward = -(lataccel_error^2 * 100 * 50 + jerk^2 * 100 + action_delta^2 * 5) / 500`. This matches the eval cost function exactly: the 50x multiplier on tracking, correct jerk formula with the 0.1s timestep in the denominator. The `/500` normalization brings rewards to a manageable scale. The `ACT_SMOOTH=5` term penalizes rapid steering changes. Getting the reward to match eval exactly was critical — early bugs here caused months of failure.

**Q: Why behavioral cloning pretrain?**

A: BC gives a warm start by imitating CSV steer commands during the warmup window (steps 20-99). The policy learns the basics of steering — roughly how much to steer for a given state. PPO then fine-tunes from this starting point. Training from scratch works but takes much longer. The key was using NLL loss (not MSE) so the Beta distribution learns both the correct mean and appropriate confidence.

**Q: Why gamma=0.95?**

A: Short horizon. The exp047 "neural atom" experiment showed that only the first 3 future timesteps matter (~0.3s lookahead). Beyond that, the autoregressive uncertainty grows too fast for planning to help. Gamma=0.95 means the effective horizon is ~20 steps (2 seconds), which matches this insight. Higher gamma (0.99) caused training instability — too much weight on uncertain future returns.

**Q: What is critic warmup?**

A: The first 4 epochs train only the critic (value function), freezing the actor. The critic needs a reasonable baseline before actor updates begin. Without warmup, advantage estimates are garbage (random critic), and the first actor updates are essentially random — often destroying the BC pretrain. With warmup, the critic learns the value landscape first, then actor updates use meaningful advantages.

**Q: What is reward normalization and why is it important?**

A: RunningMeanStd tracks the running mean and variance of returns using Welford's online algorithm. Rewards are divided by the running standard deviation before GAE. Without this, the advantage scale varies wildly across training — early episodes might have advantages in the thousands, later episodes in the single digits. The policy gradient is proportional to advantage magnitude, so without normalization, learning rates are effectively random.

### Optimization & Infrastructure

**Q: Explain how the batched simulator works.**

A: The ONNX physics model accepts dynamic batch dimensions. Instead of running 5000 episodes sequentially (5000 x 580 ONNX calls), I run them in lockstep: at each timestep, all 5000 episodes feed into a single ONNX call of shape (5000, 20, 4). That's 580 batched calls instead of 2,900,000 individual calls. The simulator histories are (N, T) arrays instead of Python lists. Every operation (clip, append, slice) works on arrays.

**Q: The existing `tinyphysics.py` already runs 5000 routes in ~2 minutes with `process_map`. Why write a batched version? Why not just use that for training?**

A: The 2-minute figure is misleading for training. Three reasons:

First, that 2 minutes is **eval with a PID controller** — a handful of multiplies per step. My controller runs a 530K-parameter neural network forward pass at every step. With the RL policy, a single episode takes ~0.6s instead of ~0.33s. 5000 episodes with the policy controller and trajectory collection is more like 5-8 minutes with a pool of 10-16 workers, not 2.

Second, even 5 minutes per epoch kills you for RL iteration. PPO needs hundreds of epochs. 5 min × 200 epochs = 17 hours for one training run. And I needed many runs — I restarted training 8-9 times just to reach cost 47, tuning hyperparameters between runs. At 5 min/epoch that's days per attempt. The batched GPU version does an epoch in 8-11 seconds. 200 epochs = 30 minutes. I can try an idea and know if it works before lunch.

Third — and this is the real reason — **multiprocessing is a dead end for the GPU path.** The whole point of moving to GPU is to run one fat ONNX call of shape `(5000, 20, 4)` instead of 5000 skinny calls of `(1, 20, 4)`. That requires all episodes in the same process, same GPU memory space. You can't batch across process boundaries. And once the simulator is GPU-resident, you want the controller GPU-resident too — building observations with torch ops, keeping training data on GPU, feeding it directly to PPO without any CPU round-trip. Multiprocessing makes all of that impossible.

I actually did use multiprocessing initially — the CPU code path in `train.py` still has it: a pool of workers each running batched chunks of ~500 CSVs. That got epochs down to ~60 seconds. But the GPU single-process path got it to 8 seconds. The batched rewrite wasn't just about speed — it was the architectural foundation that made the all-GPU pipeline possible.

But there's actually a more fundamental reason the eval machinery can't be repurposed for training. Look at how the reference simulator handles randomness: `TinyPhysicsSimulator.reset()` calls `np.random.seed(md5(path) % 10**4)` — it sets the **global** numpy random state. And `TinyPhysicsModel.predict()` samples the next lataccel via `np.random.choice(1024, p=probs)` — it reads from that same global state. This works for the `process_map` eval because each forked process gets its own copy of the global RNG. Process isolation gives you per-episode determinism for free.

But training can't use `process_map`. You need all episodes in a single process (or at least sharing the same model weights), and you need them batched together at each timestep for the fat ONNX call. The moment you run N episodes in lockstep in the same process, one global `np.random.seed` can't serve N independent random streams simultaneously. Episode 0 needs one sequence of random draws, episode 1 needs a different sequence, and they're all at the same timestep. That's fundamentally incompatible with `np.random.seed` — it gives you exactly one stream.

This is why `tinyphysics_batched.py` had to replace the global seed with per-episode `RandomState` objects:
```python
for i, f in enumerate(csv_files):
    seed = int(md5(str(f).encode()).hexdigest(), 16) % 10**4
    rng = np.random.RandomState(seed)
    self._rng_all[i, :] = rng.rand(T - CL)    # pre-generate all random values
```
And on GPU, these become a `(n_steps, N)` tensor of pre-generated uniform values indexed by `step_idx - CL`, with sampling done via `torch.searchsorted` on the CDF instead of `np.random.choice`. The entire RNG mechanism had to be rewritten from scratch.

To put it concretely: the multiprocessing overhead per epoch includes writing a checkpoint to disk, each worker loading it, each worker maintaining its own ONNX session, and then serializing ~2 million training tuples (256-dim obs, actions, values, rewards) back across process boundaries via IPC. The batched GPU version does zero disk I/O, zero IPC, zero serialization. The training data never leaves GPU memory between rollout and PPO update.

Beyond all the performance arguments, there are two **hard architectural incompatibilities** that make the eval machinery fundamentally unusable for training — not just slow, but structurally impossible:

First, **`Controller()` takes zero arguments.** The eval machinery instantiates controllers via `importlib.import_module(f'controllers.{controller_type}').Controller()`. The interface is closed — there's no way to inject the current in-memory training weights. Each worker creates an independent controller that loads its own checkpoint from whatever path is hardcoded in `__init__`. For eval, this is fine: every worker loads the same frozen file. For training, the weights change every epoch and live in the training process's memory. You can't thread a live `ActorCritic` through `importlib.import_module().Controller()`. You'd have to break the `Controller()` contract itself.

Second, **`process_map` workers are ephemeral.** Any state mutation inside a worker — learned weights, updated buffers, accumulated trajectory data — is destroyed when the worker returns its result. The return type is `(cost_dict, target_history, current_history)`: three small objects. Training data (per-step observations, actions, log-probs, values, rewards) is never returned because the interface wasn't designed for it. Even if you put all the training logic *inside* the controller (a self-contained learner), the learned weights would vanish when the worker process exits. `process_map` is a one-shot parallel map. Training is an iterative feedback loop: rollout with W₀ → update to W₁ → rollout with W₁ → update to W₂. You can't express an iterative loop as a parallel map. The workers have no mechanism to receive updated weights between batches of episodes, and no mechanism to persist learned state across calls.

These aren't performance issues you can optimize around. They're API-level incompatibilities. The batched rewrite wasn't an optimization of the eval machinery — it was a replacement, because the eval machinery's abstraction ("episode in, cost out") is fundamentally the wrong shape for training ("episode in, full trajectory out, with shared mutable model").

**Q: What was the hardest part of the batched simulator?**

A: Numerical parity with the reference. Three bugs each took days to find:
1. `np.digitize(right=True)` has inverted semantics from `torch.bucketize(right=False)` — one wrong flag shifts every token
2. Float32 bins for decoding instead of float64 — precision loss compounds over 500 autoregressive steps, causing a 20% score regression
3. RNG: the reference seeds at CONTEXT_LENGTH and generates values for steps CL-T. Pre-generating for steps 0-T wastes CL random draws, shifting the entire sequence.

I built `compare_steppers.py` to verify: run N CSVs through both the reference and batched simulator with greedy argmax, compare trajectories. The batched version must produce identical results.

**Q: How does the GPU path eliminate CPU-GPU transfers?**

A: Everything lives on GPU: the CSV data dict (`data_gpu`), the simulator histories (`action_history`, `state_history`, `current_lataccel_history`), the controller's ring buffers, the observation buffer, the training data arrays. The ONNX model reads from and writes to GPU tensors via IOBinding. The controller builds observations with torch ops and returns a GPU tensor. The only CPU touch is the final `costs.cpu().numpy()` for logging. Ctrl time went from 9.5s to 1.3s — a 7x speedup just from eliminating three round-trips per step.

**Q: What is IOBinding?**

A: Default ONNX GPU inference: numpy -> CPU staging buffer -> GPU copy -> inference -> GPU copy -> CPU staging -> numpy. Every call. IOBinding lets you bind pre-allocated GPU tensors directly as inputs and outputs: `io.bind_input('states', 'cuda', 0, np.float32, shape, tensor.data_ptr())`. ONNX reads from and writes to your GPU memory. Zero copy per call.

**Q: How did TensorRT help?**

A: TensorRT fuses multiple transformer operations into single optimized kernels, auto-tunes for the specific GPU, and runs FP16 where numerically safe. ONNX Runtime supports it as an execution provider. First epoch pays ~30s compilation cost, then the cached engine loads in <1s. Sim time: 10s -> 5s. TensorRT automatically detected LayerNorm overflow in FP16 and forced those layers back to FP32.

**Q: Explain the MacBook cluster.**

A: Three MacBooks connected via Thunderbolt USB-C (40 Gbps point-to-point). Each remote runs a persistent TCP server. The training loop sends a pickle blob with checkpoint bytes and CSV file list. Remote workers run rollouts and return NPZ results. Load balancing via `FRAC` weights: `FRAC=3.5:3.5:3` allocates 35%/35%/30% of CSVs to each machine. The cluster doubled training data throughput at the same wall time. Thermal throttling on the fanless Air was the main limitation.

### MPC

**Q: How does the shooting MPC work?**

A: Sample 16 action candidates from the policy's own Beta distribution. Slot 0 is always the deterministic mean (no-regression guarantee). For each candidate, unroll 5 steps through the ONNX model, computing expected lataccel (not sampled — smoother). Score each 5-step trajectory with the real cost function (lataccel error * 50 + jerk). Apply the first action of the best-scoring trajectory. The policy proposes candidates in its own learned action space; the ONNX model validates them.

**Q: Why use expected value instead of sampling in MPC?**

A: Sampling adds noise. Two candidates might have identical true costs, but sampling luck makes one look better. Expected value (`E[lataccel] = sum(probs * bins)`) gives a deterministic, smooth prediction. Fair comparison between candidates.

**Q: Why does slot 0 matter?**

A: Slot 0 is the policy mean. It guarantees the MPC never picks something worse than the pure policy. If all 15 random samples are bad, slot 0 (the policy mean) wins. Without this, the MPC could occasionally regress. "We will never diverge from the policy action basin because the policy is the one who proposed all the candidates."

**Q: What's the difference between Newton correction and shooting MPC?**

A: Newton correction is cheap (1 extra ONNX call) and makes a small analytical correction to the policy's action. Shooting MPC is expensive (N * H ONNX calls) but explores the action space more thoroughly. Newton correction is a first-order local fix. Shooting is global search within the policy's distribution. In practice, shooting gives 2-4 point improvement; Newton gives 1-2 points.

### Philosophy & Journey

**Q: What was the key insight that unlocked progress?**

A: Curvature encoding. Representing the road as curvature `kappa = (lat - roll) / v^2` instead of raw lateral acceleration made the observation speed-invariant. BC cost dropped from 130 to 75 from one line of physics. The No Free Lunch theorem says there's no universal algorithm — but if you guess the problem's structure correctly, you get a "cheap meal." Curvature was the cheap meal.

**Q: Why did you solve LunarLander first?**

A: Too many unknowns in the controls challenge. Bad results could be broken PPO, wrong architecture, or insufficient data — three suspects, entangled. LunarLander isolates PPO as a variable. If it works there, PPO is no longer the suspect. This "fix one invariant at a time" approach was the methodological throughline of the project.

**Q: What failed?**

A: Many things. Pure PPO with sigma=1.0 was catastrophic (cost 10,000+). BC → PPO with MSE loss never beat BC — months of this before finding the NLL fix. State-dependent sigma wrecked deterministic eval. Tanh squashing created a train/eval gap. PID residual had hidden integral state mismatch. MPC smoothness penalty hurt (internal cost must match eval cost exactly). The ONNX MPC had an off-by-one bug predicting from stale context. The innovation/Kalman feature doubled compute for marginal benefit and was removed. An early reward function was missing the 50x multiplier entirely.

**Q: What's the gap from your score (42.5) to the #1 score (35.9)?**

A: The #1 solution (haraschax, 35.97) uses heavier MPC compute — more candidates, longer horizons, more ONNX calls per step. The policy gets to ~45 on its own. The ONNX model as a forward simulator for MPC gets to ~36. Both are needed. I had the policy. I didn't push MPC hard enough. The leaderboard is telling you: the policy gets you to 45, the ONNX model gets you to 36.

**Q: How long did this take?**

A: Fork on Christmas Day 2025. Intense 2-week sprint Feb 2-15, 2026. Most productive days: Feb 14 (16 commits, cost 47 -> 43) and Feb 15 (20+ commits, GPU blitz, 42.5 on 5000 routes). The 37-day gap (Dec 26 - Feb 2) was spent on MPC experiments, LunarLander, and thinking about the problem.

**Q: What would you do differently?**

A: Push MPC harder and earlier. The gap from 42.5 to 35.9 is almost entirely from test-time compute (more shooting candidates, longer horizons). Also: start with correct reward function from day one. The reward bugs caused months of wasted effort. And start with NLL loss for BC instead of MSE — the sigma issue wasted weeks.

### Technical Deep Dives

**Q: How does the ONNX physics model work internally?**

A: It's a GPT-style transformer. Input: (batch, 20, 4) float32 states (steer, roll, v, a) and (batch, 20) int64 tokens (discretized past lataccel). Output: (batch, 20, 1024) logits. Only the last timestep matters — it predicts the distribution over next lataccel. The 1024 bins span [-5, 5] evenly. Sampling: softmax at temperature 0.8, then categorical sample. The model is autoregressive — step N's sampled output becomes step N+1's input token.

**Q: What is the friction circle feature?**

A: `friction = sqrt(current_lataccel^2 + a_ego^2) / 7.0`. This combines lateral and longitudinal acceleration demands. When close to 1.0, the tire is near its grip limit. The `headroom = max(0, 1 - friction)` tells the policy how much spare grip is available. Steering aggressively with low headroom risks instability. The `/7.0` normalizer corresponds to roughly 0.7g total acceleration budget.

**Q: Why is the error integral computed as mean of last 20 errors times DEL_T?**

A: It approximates the integral of tracking error over the last 2 seconds. Like a PID integral term — it accumulates persistent bias. If the controller consistently undershoots, this grows positive and biases output upward. The windowed mean (20 steps) prevents integral windup. The `* DEL_T` converts from per-step sum to time-integrated value.

**Q: How do you handle variable-length future plans?**

A: The `_future_raw()` helper handles three cases: (1) if the plan has >= 50 values, take first 50; (2) if it has fewer, edge-pad by repeating the last value; (3) if it's empty, fill with the current value as fallback. This ensures the observation vector is always exactly 256 dimensions regardless of how much future plan data is available.

**Q: What is the `set_model` hook?**

A: When the `TinyPhysicsSimulator` is created, it checks if the controller has a `set_model` method and calls it with the `TinyPhysicsModel` instance. This gives the MPC controller access to the ONNX session for lookahead predictions. The controller can call `model.ort_session.run()` to simulate future trajectories. During training, the model access comes through the batched simulator instead.

**Q: How do you ensure the MPC's RNG doesn't interfere with the simulator's RNG?**

A: `self._mpc_rng = np.random.RandomState(42)` — a separate random state. The simulator uses `np.random.seed(md5(path))` which sets the global numpy RNG. If MPC consumed values from the global RNG, it would shift the simulator's sampling sequence and break parity. The decoupled RNG keeps MPC and simulator independent.

**Q: What is the `_unroll_cost` function doing?**

A: It simulates H future steps using the ONNX model and computes the total cost. For each step: (1) build action sequence with the candidate action, (2) run `_onnx_expected` to get predicted next lataccel, (3) compute lataccel cost and jerk cost, (4) shift the ONNX context (autoregressive), (5) optionally run the policy forward to get the next action (`MPC_ROLL=1`) instead of holding the action constant. Returns total cost across all H steps.

**Q: Explain the `compare_steppers.py` test.**

A: Three-way parity test. Run N CSVs through: (1) the original `TinyPhysicsSimulator` with greedy argmax, (2) `BatchedSimulator` with N=1 (one CSV at a time), (3) `BatchedSimulator` with N=10 (all at once). If N=1 matches reference exactly, the code logic is correct. If N=10 differs from N=1, that's just ONNX batch floating-point non-determinism. Also tests stochastic parity with seeded RNG. This test was essential for trusting the batched simulator's correctness.

**Q: Why cosine LR decay?**

A: Cosine annealing smoothly reduces the learning rate from its initial value to near-zero over the total training epochs: `lr = base * 0.5 * (1 + cos(pi * epoch / total))`. It gives the policy freedom to explore early (high LR) and gradually settles into fine-tuning (low LR). Step 47 -> 43 happened after enabling cosine decay. The smooth schedule avoids the discontinuities of step decay.

**Q: What does the `Ctx` class do?**

A: Central training context. Holds the model (`ActorCritic`), the PPO trainer (with optimizers and running stats), the ONNX model path, training and validation file lists, the best validation score, and infrastructure objects (multiprocessing pool for CPU mode, ONNX session and CSV cache for GPU mode). It manages checkpoint saving and provides a clean interface for `train_one_epoch()` and `evaluate()`.

---

## 13. Hard / Pointed Questions (Elite Interviewer)

These questions are curated from a deep code audit. Each one catches a real bug, dead code, a subtle inconsistency, a precision issue, or a non-obvious architectural consequence. They test whether you truly understand the *why* behind every line, not just the *what*.

### Gotcha / Bug Questions

**Q: `MAX_DELTA=0.5` — when does this clip ever fire?**

A: It doesn't. With `DELTA_SCALE=0.25`, the raw policy output is in `[-1, 1]`, so `raw * DELTA_SCALE` maps to `[-0.25, 0.25]`. The `np.clip(..., -MAX_DELTA, MAX_DELTA)` with `MAX_DELTA=0.5` never activates because `0.25 < 0.5`. This is dead code — either `MAX_DELTA` was once smaller, or `DELTA_SCALE` was once larger. In the stochastic MPC path, `(2*s - 1) * DELTA_SCALE` where `s ~ Beta(a,b)` in `[0,1]` also maps to `[-0.25, 0.25]`, still inside the clip. The *real* saturation happens at the `STEER_RANGE` clip on the next line. `MAX_DELTA` is vestigial and provides zero safety benefit in the current config. If I were cleaning this up, I'd either remove the `MAX_DELTA` clip entirely or set it equal to `DELTA_SCALE` so the invariant is obvious.

**Q: In `compute_rewards` (train.py line 233), `act = np.array([t.get('act', 0.0) for t in traj])` — when is the `'act'` key ever set in the trajectory dict?**

A: Never. `DeltaController.update()` appends `dict(obs=obs, raw=raw, val=val, tgt=target_lataccel, cur=current_lataccel)` — five keys, none named `'act'`. So `t.get('act', 0.0)` always returns `0.0`, which means `act_d = np.diff(act) / DEL_T` is all zeros, and the `ACT_SMOOTH` penalty is **always zero** in the sequential CPU path. The GPU and batched-CPU rollout paths compute action differences correctly from the actual action history buffers. This is a real bug — the action smoothness reward is silently disabled when `BATCHED=0`. In practice it didn't matter because I used `BATCHED=1` (batched CPU) or `CUDA=1` (GPU) for all real training runs. But if someone ran the sequential path expecting the smoothness penalty to work, they'd get different training dynamics with no error message.

**Q: `pretrain_bc` is called with `all_csvs = sorted((ROOT / 'data').glob('*.csv'))` (train.py line 1206), but RL training uses `ctx.tr_f` which excludes the first 100 files (the validation set). Is there data leakage?**

A: Yes, technically. BC sees all files including the 100 validation files; RL only trains on the remainder. The BC-initialized actor weights have been optimized (via NLL loss) on the validation set. The validation evaluation is therefore biased — the BC-initialized policy has seen those trajectories. In practice, the leakage is minor for three reasons: (1) BC only trains on the pre-control window (steps 20–100) with a perfect-tracking assumption, so it's learning generic steering patterns, not memorizing specific trajectories; (2) RL quickly overwrites BC weights during fine-tuning; (3) the validation set is 100 out of ~10,000 files (1%). But methodologically it's sloppy. The fix is one line: `pretrain_bc(ctx.ac, ctx.tr_f)` instead of `pretrain_bc(ctx.ac, all_csvs)`.

**Q: `torch.save` on train.py line 1024 writes to `.ckpt.pt`, and pool workers read it via `_load_ckpt`. What if a worker reads while the main process is writing?**

A: This is a real race condition. `torch.save` writes directly to the file — it's not atomic. A pool worker calling `_load_ckpt` might see a partially written file, causing a corrupt `torch.load`. The mtime-based cache check makes it worse: the worker could read a new mtime (mid-write) but get incomplete data. The fix is to write to a temp file and then `os.rename()`, which is atomic on POSIX. In practice, the race window is small (the checkpoint is ~2MB, written in milliseconds), and I never hit it during training. But it's the kind of bug that causes one-in-a-thousand training crashes that are impossible to reproduce.

**Q: `torch.load(..., weights_only=False)` appears in both the controller (line 99) and train.py (lines 953, 1186). Why is this a concern?**

A: `weights_only=False` allows arbitrary Python object deserialization via pickle. A malicious `.pt` file could execute arbitrary code on load. Since the model path is controlled via environment variable (`MODEL` env var in the controller), an attacker who can place a file on disk gets code execution. The safe option is `weights_only=True`, but that requires the checkpoint to contain only safe types (tensors, dicts, lists). In this case, the checkpoints are just `{'ac': state_dict()}`, which is all tensors — so `weights_only=True` should work. I used `weights_only=False` because that was the default behavior before PyTorch 2.6 tightened the security model, and I never went back to update it.

### Precision / Numerical Questions

**Q: The CPU path computes softmax with `np.exp`. The GPU path uses `torch.softmax`. Are these guaranteed to produce the same probabilities?**

A: No. NumPy auto-promotes float32 inputs to float64 for transcendental operations like `np.exp`. So the CPU softmax is effectively computed in float64. PyTorch's `torch.softmax` does *not* promote — it stays in float32 on a float32 input. The resulting probability distributions differ by ~1e-7 relative error, which shifts the CDF and can change the sampled token on some fraction of steps. This is the single largest source of CPU/GPU result divergence. If bit-exact matching is needed, the GPU softmax should operate on `.double()` tensors and convert back. In practice, I accepted this divergence — the `compare_steppers.py` test verifies parity with greedy argmax (which isn't affected by softmax precision), and stochastic parity is tested statistically (cost distributions overlap) rather than bit-exactly.

**Q: The ONNX model's `temperature=0.8` is hardcoded in `_onnx_expected` and `_onnx_expected_gpu`. Where does this come from? What happens at a different temperature?**

A: The 0.8 comes from the reference simulator's `get_current_lataccel()`, which calls `predict(input_data, temperature=0.8)`. It's the physics model's configured sampling temperature — slightly sharpened from a uniform temperature=1.0. In the MPC expected-value path, we compute `E[lataccel] = sum(softmax(logits / T) * bins)`. Lower temperature concentrates probability on fewer bins, making the expected value closer to the mode. At temperature=1.0, the distribution is wider and the expected value is more regressed toward the mean. The MPC should use the same temperature the simulator uses, because it's predicting what the simulator will *actually produce* — and the simulator samples at 0.8. If you used 1.0 in MPC but the simulator runs at 0.8, the MPC's predictions would be systematically more diffuse than reality, leading to suboptimal action selection. The fact that it's hardcoded rather than parameterized is a code quality issue — it should be a constant imported from the same place the simulator reads it.

**Q: TensorRT is enabled with `trt_fp16_enable: True`. What does this mean for simulator fidelity?**

A: FP16 inference changes the ONNX model's internal computation precision. The logits fed into softmax will differ from FP32, producing different probability distributions and different sampled tokens. This means TRT-FP16 mode produces a **different simulator** — not just faster, but with different dynamics. A controller tuned against TRT-FP16 training dynamics may not transfer perfectly to the FP32 evaluation that the leaderboard runs. In practice, TensorRT is smart about this: it automatically detected that LayerNorm layers would overflow in FP16 and forced those layers back to FP32. The actual logit divergence was small enough that training still transferred. But it's a real concern — I validated this by running the final eval with the standard (non-TRT) ONNX runtime to confirm the score held.

### Architecture / Design Questions

**Q: The actor outputs two values mapped through `softplus(x) + 1.0` to get Beta(α, β). Why add 1.0 after softplus? What does this prevent?**

A: `softplus(x) ≥ 0`, so `softplus(x) + 1.0 ≥ 1.0`. This forces α, β ≥ 1, which means the Beta distribution is **always unimodal** — it can never be U-shaped (α < 1, β < 1) or J-shaped (one parameter < 1). This is a deliberate architectural constraint: unimodal policies are stable for PPO training because the log-probability gradient is well-behaved everywhere. The cost is expressiveness — the policy can never represent "I want to be at one of the extremes but not the middle" (bimodal uncertainty). For a tracking controller this is fine: at any given state, there's typically one correct steering direction, not two equally valid ones. The `+ 1.0` also prevents α or β from being exactly 0 (which would make the distribution degenerate) or very close to 0 (which would create infinite log-prob gradients at the boundaries).

**Q: `action *= v / V_REF` (line 676) happens *after* the LPF and rate-limit. Doesn't this mean the effective rate limit scales with speed?**

A: Yes, and that's likely unintended. The rate-limit on line 668 caps `|Δaction|` at `RATE_LIMIT` per step. But then VNORM multiplies the final action by `v/V_REF`. At 40 m/s with `V_REF=20`, the action is doubled *after* rate-limiting, so the effective rate limit becomes `2 * RATE_LIMIT`. At 5 m/s, it's quartered to `0.25 * RATE_LIMIT`. If the intent was to rate-limit the *physical* steer command, the VNORM should be applied *before* rate-limiting, or the rate limit should be specified in post-VNORM units. In practice, `VNORM=0` in the final submission (it's off by default), so this bug is dormant. But if someone enables it, the speed-dependent rate limit could cause jerky behavior at high speed and sluggish response at low speed.

**Q: The MPC's `_unroll_cost` predicts `pred_la = _onnx_expected(...)` and uses it directly. But the real simulator clips predictions to `current ± MAX_ACC_DELTA` (0.5). Does MPC model this clip?**

A: No. The real simulator (`tinyphysics.py` line 139) applies `np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)` before updating `current_lataccel`. The MPC's `_unroll_cost` skips this clip entirely. This means the MPC's world model can predict lataccel jumps larger than 0.5 per step, which the real simulator would never produce. The consequence: MPC's jerk estimates are **optimistic** — it thinks a large correction can happen in one step, so it underestimates the jerk cost. In practice, the ONNX model rarely predicts jumps > 0.5 (it's autoregressive and learned smooth dynamics), so the clip rarely matters. But on edge cases — aggressive maneuvers, high-speed transitions — the MPC could select an action whose real-world jerk is higher than predicted. The fix is one line: `pred_la = np.clip(pred_la, prev_la - MAX_ACC_DELTA, prev_la + MAX_ACC_DELTA)` inside the unroll loop.

**Q: During warmup (`n <= WARMUP_N`), the controller pushes `0.0` into `_h_act`. But the simulator applies the CSV's actual steer commands for those steps. Isn't this a false initial condition?**

A: Yes. The simulator overrides the controller's action with `self.data['steer_command'].values[step_idx]` for steps before `CONTROL_START_IDX` (tinyphysics.py line 150). But the controller pushes `0.0` into its own `_h_act` history during warmup. At the moment control starts (step 101), the controller's internal action history says "I've been steering 0.0 for the last 20 steps" while the car was actually executing non-zero steer commands from the CSV. This is a transient mismatch — within 20 steps (2 seconds), the real actions flush out the zeros. The impact depends on how sensitive the policy is to the action history feature. The alternative would be to push the CSV steer commands during warmup, but the controller doesn't have access to them (only the simulator does). The `set_model` hook gives the controller the physics model, but not the raw CSV data. So the controller does the best it can: assume zero initial steer.

**Q: There are three independent implementations of observation building: `_build_obs` (scalar), `_build_obs_batch` (CPU batch), and `_build_obs_batch_gpu` (GPU batch). How do you guarantee they produce identical observations?**

A: I don't have an automated guarantee. They were written independently — the scalar version first, then the batch versions as vectorized translations. Any change to the feature set requires updating all three in lockstep. There are no unit tests comparing their outputs. This is a classic "shotgun surgery" code smell. In practice, I verified them manually during development: insert a batch of 1 into the batch version and compare against the scalar version. But if someone changes one and forgets the others, the CPU and GPU MPC paths would silently make different decisions. The proper fix is to refactor so the batch version calls the scalar version (or vice versa), or at minimum add a test: `assert np.allclose(_build_obs_batch(N=1), _build_obs())`.

**Q: BC trains with `current_la = tgt[step_idx]` — perfect tracking. What happens to the error features when RL starts?**

A: During BC, `error = target - current = 0` at every step, `error_integral = 0`, and `(target - current) / S_LAT = 0` in the observation. The actor learns that these features are always zero and effectively ignores them (the corresponding weights drift toward small values or cancel out). When RL begins and the policy starts actually controlling — producing imperfect tracking with nonzero error — these features suddenly activate. The policy sees out-of-distribution inputs on the error channels. This could cause a sharp initial performance drop at the BC-to-RL transition. The `CRITIC_WARMUP=4` epochs help (critic adapts before the actor changes), but the actor itself is still calibrated for zero-error observations. The alternative would be to run BC with the actual simulator (not perfect tracking), but that's much more expensive because you need ONNX rollouts, not just static feature extraction from CSV data. The BC is deliberately cheap — it's just there to put the policy in the right ballpark so RL doesn't start from random noise.

**Q: `MINI_BS=100000` — with ~500 CSVs × ~200 steps, that's ~100K samples. What happens when you call `randperm(N).split(MINI_BS)` with N ≈ MINI_BS?**

A: You get exactly one minibatch containing all samples. So `K_EPOCHS=4` means 4 gradient steps on the **entire** buffer — this is full-batch gradient descent, not SGD. There's no minibatch stochasticity, no per-minibatch advantage normalization variance. The upside: deterministic, stable updates. The downside: you lose the implicit regularization that SGD minibatching provides. In standard PPO, per-minibatch advantage normalization acts as a form of adaptive learning rate (each minibatch's gradients are scaled by its own std). With full-batch, the normalization is fixed. For this problem, full-batch worked fine — the environment is low-noise and the policy is small (530K params), so SGD regularization isn't critical. If the model were larger or the environment noisier, proper minibatching would matter more.

**Q: `GAMMA=0.95, LAMBDA=0.9`. The effective GAE horizon is `1/(1-γλ)`. What is that, and is it appropriate for a 400-step trajectory?**

A: `1/(1-0.95*0.9) = 1/(1-0.855) ≈ 6.9` steps. The GAE advantage estimate is dominated by rewards within ~7 steps of the current state. For a 400-step tracking trajectory, this is extremely myopic — the agent barely looks ahead. But for a tracking controller, this is arguably correct: tracking error is a *local* property. The optimal steer at step 100 depends on the target in the next ~0.7 seconds, not what happens at step 300. A longer horizon would mix in distant rewards that the current action has little influence over, adding variance to the advantage estimate. The low gamma also helps with credit assignment: if the agent makes a mistake at step 100, the penalty is attributed to steps 93–107, not spread across the whole trajectory. I tried `GAMMA=0.99` early on and it was strictly worse — the advantage estimates were noisier and the policy updated more slowly.

**Q: At low speed (v < 1 m/s), `_curv` computes `(lat - roll) / max(v², 1.0)`. The curvature features are divided by `S_CURV=0.02` and clipped at ±5. What information is lost?**

A: At v ≈ 1 m/s, the `max(v², 1.0)` guard clamps the denominator to 1.0, so `_curv` returns `(lat - roll)` directly rather than true curvature. If `lat - roll = 3 m/s²`, curvature = 3.0. After `/ S_CURV (0.02)` = 150, clipped to 5.0. The network sees the same value for a 0.1 curvature and a 3.0 curvature — it's completely saturated. Multiple curvature-derived features (target curvature, current curvature, curvature error) all clip simultaneously. The network is effectively **blind to curvature magnitude at low speed**. In practice, this matters less than it sounds: low-speed segments are rare in the dataset (highway driving), and at low speed the required steer authority is small anyway. The scaling constants (`S_CURV`, `S_LAT`, etc.) were tuned empirically so the clip rarely engages in the operating regime that dominates the cost function.

**Q: The multiprocessing pool in `tinyphysics_batched.py` uses forked processes. If CUDA was initialized in the parent, what happens?**

A: CUDA contexts are not fork-safe. If the parent process initializes CUDA (e.g., by creating a `BatchedPhysicsModel` with `CUDA=1`) and then forks to create pool workers, the child processes inherit a copy of the CUDA context, leading to undefined behavior: hangs, segfaults, or silently wrong results. On macOS the default multiprocessing start method is `spawn` (safe — fresh processes), but on Linux it's `fork`. The code doesn't explicitly set the start method. The `pool_init` function correctly creates fresh ONNX sessions in each worker, but only if the parent hasn't touched CUDA first. In `train.py`, the CPU path creates the pool *before* any GPU initialization, so it's safe. But if someone reorganized the initialization order, this would silently break. The defensive fix: `multiprocessing.set_start_method('spawn')` at the top of the script.

**Q: RNG seeds are `md5(path) % 10**4`. With 10,000 possible seeds and 5,000 files, what's the collision probability? Does it matter?**

A: By the birthday paradox, the collision probability with N items in a space of size S is approximately `1 - exp(-N²/(2S))`. With N=5000 and S=10000: `1 - exp(-25000000/20000) = 1 - exp(-1250) ≈ 1.0`. Collisions are virtually guaranteed. In fact, with 5000 files in a 10000-slot space, you'd expect ~1250 collisions on average. Two files with the same seed share the same physics sampling sequence — their stochastic dynamics are correlated. This doesn't affect correctness (each file still gets a valid rollout), but it reduces the effective diversity of the evaluation. This is inherited from the reference implementation (same seeding logic), so changing it would break parity with the official eval. The fix would be `% 2**31` for a much larger seed space, but only if the reference also changes.
