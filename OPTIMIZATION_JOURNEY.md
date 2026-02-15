# How We Made a Physics Simulator Go Brrr: From 500x Sequential ONNX Calls to GPU-Fused Batched Rollouts

## Introduction

The comma.ai controls challenge asks you to write a controller that steers a car to follow a target lateral acceleration profile. Your controller is evaluated inside `tinyphysics.py` — a simulator that calls an ONNX transformer model 500 times per episode, autoregressively predicting physics one timestep at a time.

The baseline: one episode takes ~0.5 seconds. Evaluating 5,000 episodes takes ~40 minutes. Training a reinforcement learning policy over thousands of epochs at this speed is not feasible.

This is the story of how we took that 40-minute evaluation down to ~12 seconds — a **200x speedup** — through a sequence of compounding optimizations, each one unlocking the next.

The path was not linear. There were wrong turns, subtle bugs, precision traps, and moments where "obvious" optimizations did nothing. Every layer taught us something about where the real bottlenecks hide.

---

## Part 1: Understanding the Baseline

### The Architecture

The simulator has a simple loop:

```
for step in range(CONTEXT_LENGTH, 500):
    state, target = get_state_from_csv(step)
    action = controller.update(target, current_lataccel, state)  # your policy
    pred = onnx_model.predict(last_20_states, last_20_actions, last_20_preds)  # physics
    current_lataccel = clip(pred, current ± 0.5)
```

Each `predict` call:
1. Tokenizes the last 20 predictions via `np.digitize` into 1024 bins
2. Builds a `(1, 20, 4)` float32 states tensor and `(1, 20)` int64 tokens tensor
3. Runs ONNX inference → `(1, 20, 1024)` logits
4. Softmax on the last timestep → sample from distribution → decode token back to float

This is autoregressive — step N's prediction feeds into step N+1's input. You cannot parallelize across timesteps. But you *can* parallelize across episodes.

### The Cost Breakdown

For a single episode (~500 steps):
- **480 ONNX calls** at ~1ms each = ~0.5s
- Controller logic, numpy slicing, history management = ~0.1s
- Total: **~0.6s per episode**

For 5,000 episodes sequentially: **~50 minutes.**

The critical insight: every episode is independent. The ONNX model processes one `(1, 20, 4)` input — but it could just as easily process `(5000, 20, 4)`. One fat call instead of 5,000 skinny ones.

---

## Part 2: Batched Simulation (50 min → 40s)

### The Rewrite

We built `tinyphysics_batched.py` — a drop-in replacement that runs N episodes in lockstep. Instead of:

```
for episode in 5000_episodes:
    for step in 500_steps:
        predict(1, 20, 4)  →  5000 × 500 = 2,500,000 ONNX calls
```

We do:

```
for step in 500_steps:
    predict(5000, 20, 4)  →  500 ONNX calls
```

Same total compute, but 5000x fewer kernel launches. The ONNX model's transformer naturally handles the batch dimension — no architecture changes needed.

The histories become `(N, T)` arrays instead of Python lists. State becomes `(N, T, 3)`. Every operation — clip, append, slice — now operates on arrays, not scalars.

### The Parity Trap

This sounds simple. It wasn't. The reference simulator has subtle behaviors:

- **RNG seeding**: Each episode seeds `np.random.seed(hash(filename))`. The batched version needs per-episode `RandomState` objects.
- **History management**: The reference uses `.append()` on lists. We use pre-allocated arrays with index writes.
- **`np.digitize` behavior**: `right=True` means "intervals closed on the right." One wrong flag and every token shifts by one, causing the autoregressive simulation to diverge catastrophically over 500 steps.

Getting exact numerical parity took multiple iterations. A float32 vs float64 mismatch in token decoding caused the baseline score to jump from 49.7 to 59.2 — a 20% regression from a precision bug in one line.

### Result

5,000 episodes: **~40 seconds** on CPU (10 workers with multiprocessing). Down from ~50 minutes.

**Speedup: ~75x.**

---

## Part 3: Moving to GPU (40s → 20s per epoch)

### The Motivation

CPU multiprocessing hit a wall. 10 workers, each running a batched ONNX session — and we're still spending 40s on rollouts. The ONNX model is a transformer. Transformers are embarrassingly parallel on GPUs. Running `(5000, 20, 4)` through a GPU should be nearly the same latency as `(1, 20, 4)`.

### ONNX Runtime CUDA EP

The first step was simple — swap the execution provider:

```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession(model_bytes, options, providers)
```

Single process. One ONNX session on GPU. 5,000 episodes batched. No multiprocessing.

### IOBinding: Eliminating the Bounce

The default ONNX GPU path: numpy array → CPU staging buffer → GPU → inference → GPU → CPU staging → numpy array. Every step. 500 times.

IOBinding lets you bind pre-allocated GPU tensors directly:

```python
io = session.io_binding()
io.bind_input('states', 'cuda', 0, np.float32, shape, gpu_tensor.data_ptr())
io.bind_output('output', 'cuda', 0, np.float32, shape, out_gpu.data_ptr())
session.run_with_iobinding(io)
```

The ONNX output writes directly into a persistent GPU tensor. Softmax and sampling happen on GPU via PyTorch. Only the final N sampled token indices come back to CPU.

### GPU-Resident Tokenization

The original tokenization: `np.clip` → `np.digitize` on CPU, creating numpy arrays that then get copied to GPU.

The GPU version: `torch.clamp` → `torch.bucketize` — entirely on GPU. One subtlety: `torch.bucketize(right=False)` corresponds to `np.digitize(right=True)`. They have inverted semantics. Getting this wrong silently produces wrong tokens.

### GPU-Resident Simulation Histories

The histories (`action_history`, `state_history`, `current_lataccel_history`) started as numpy arrays on CPU. Each step, we'd slice them, copy to GPU for ONNX, then write the prediction back to CPU.

Moving them to GPU tensors eliminated the per-step CPU↔GPU transfer:

```python
self.action_history = torch.zeros((N, T), dtype=torch.float64, device='cuda')
self.state_history = torch.zeros((N, T, 3), dtype=torch.float64, device='cuda')
```

Now slicing `history[:, h-CL:h]` produces a GPU tensor view. ONNX reads it directly. The prediction stays on GPU. Zero bus traffic.

### Result

sim time: **17s → 10s.** Total epoch: ~33s.

**Compounded speedup: ~150x** from original.

---

## Part 4: The CSV Trap (33s → 12s per epoch)

### Where Did 13 Seconds Go?

The rollout loop (ctrl + sim) took 20s. But the epoch timer said 33s. We lost 13 seconds to overhead we weren't measuring:

| Hidden Cost | Time | Cause |
|---|---|---|
| `preload_csvs` | ~6s | Re-parsing 5000 CSVs via pandas **every epoch** |
| `reset()` | ~3s | Allocating 60MB numpy, 5000 RNG objects, GPU transfers |
| Post-rollout | ~3s | Transposing 700MB obs array, 5000-iteration result loop |
| `save_ckpt` | ~1s | `torch.save` to disk every epoch (not needed in GPU mode) |

### CSVCache: Parse Once, Slice Forever

```python
class CSVCache:
    def __init__(self, all_csv_files):
        self._master = preload_csvs(all_csv_files)  # one-time: ~47s at startup
        self._rng_all = precompute_all_rng(all_csv_files)

    def slice(self, subset_files):
        idxs = [self._file_to_idx[f] for f in subset_files]
        return {k: self._master[k][idxs] for k in keys}, self._rng_all[idxs]
```

47 seconds at startup. Then each epoch's "CSV loading" is a numpy fancy-index operation: **~0.1 seconds**.

### The Data Dict on GPU

The simulator's data dictionary (`roll_lataccel`, `v_ego`, `a_ego`, `target_lataccel`, `steer_command`) was read from CPU every step for state lookups and future plan slicing. Moving it to GPU:

```python
self.data_gpu = {}
for k in data_keys:
    self.data_gpu[k] = torch.from_numpy(data[k]).cuda()
```

Now `dg['target_lataccel'][:, step_idx]` is a GPU tensor slice. No CPU involvement.

### Result

Epoch time: 33s → **12s** (rollout) + 20s (PPO) = 32s total.

The rollout itself: **~12s.**

---

## Part 5: All-GPU Controller (20s rollout → 12s rollout)

### The Last CPU Bottleneck

After all the sim optimizations, the timing showed:

```
ctrl = 9.5s    sim = 10.2s    total = 19.7s
```

The controller was still doing everything on CPU:

```python
# Per step (old):
current_la = sim.current_lataccel.cpu().numpy()   # GPU → CPU
obs_buf = build_obs_numpy(target, current_la, ...)  # 20 numpy ops on (5000,)
obs_t = torch.from_numpy(obs_buf).to('cuda')       # CPU → GPU
a, b = ac.beta_params(obs_t)                        # GPU inference
raw = (2*x - 1).cpu().numpy()                       # GPU → CPU
action = np.clip(...)                                # CPU
return action  # will be sent back to GPU for sim
```

Three CPU↔GPU round-trips per step. 500 steps. For 5,000 episodes. All because the observation building was in numpy.

### The Full GPU Controller

Every buffer moved to GPU:

```python
# Ring buffers
h_act   = torch.zeros((N, HIST_LEN), dtype=torch.float64, device='cuda')
h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
h_lat   = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device='cuda')
obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device='cuda')

# Training data collection
all_obs = torch.empty((max_steps, N, OBS_DIM), dtype=torch.float32, device='cuda')
all_raw = torch.empty((max_steps, N), dtype=torch.float32, device='cuda')
```

The observation building becomes pure torch operations:

```python
target     = dg['target_lataccel'][:, step_idx]     # GPU slice
current_la = sim.current_lataccel                     # GPU tensor
v2 = torch.clamp(v_ego * v_ego, min=1.0)
k_tgt = (target - roll_la) / v2
obs_buf[:, 0] = target / S_LAT
obs_buf[:, 1] = current_la / S_LAT
# ... all 256 obs dimensions built on GPU ...
obs_buf.clamp_(-5.0, 5.0)

with torch.inference_mode():
    a_p, b_p = ac.beta_params(obs_buf)  # already on GPU, no transfer
```

The controller receives `(step_idx, sim)` and returns a GPU tensor. The rollout loop touches CPU exactly zero times per step.

### The Rollout Loop Signature Changed

```python
# Old (CPU):
controller_fn(step_idx, target, current_la_numpy, state_dict, future_plan)

# New (GPU):
controller_fn(step_idx, sim)  # sim.data_gpu, sim.current_lataccel are GPU tensors
```

### PPO.update: GPU Tensors In, No Concatenation

The training data is already on GPU. No more `np.concatenate` + `torch.FloatTensor().to(DEV)`:

```python
if isinstance(all_obs[0], torch.Tensor):
    obs_t = torch.cat(all_obs, dim=0)  # already on GPU
    raw_t = torch.cat(all_raw, dim=0).unsqueeze(-1)
```

### Result

```
ctrl = 1.3s    sim = 10.1s    total = 11.4s
```

ctrl: 9.5s → **1.3s**. The epoch: **12s + 19s = 31s.**

---

## Part 6: The Compound Effect

Here's how each optimization layer built on the previous:

| Step | What Changed | Rollout Time (5000 eps) | Epoch Time | Cumulative Speedup |
|---|---|---|---|---|
| Baseline | Sequential single-episode | ~50 min | N/A | 1x |
| Batched CPU | N episodes in lockstep, multiprocessing | ~40s | ~60s | 75x |
| GPU ONNX | CUDAExecutionProvider, single process | ~25s | ~45s | 120x |
| IOBinding + GPU tokenize | Zero-copy ONNX I/O, torch.bucketize | ~20s | ~40s | 150x |
| GPU-resident histories | Histories as CUDA tensors | ~20s | ~33s | 150x |
| CSVCache | Parse once at startup | ~20s | ~20s | 150x (epoch: 150x) |
| GPU controller + data_gpu | All obs/ctrl/data on GPU | ~12s | ~12s + 19s PPO | 250x |

The physics sim went from 50 minutes to 12 seconds. Each layer was necessary — you couldn't skip to the end. GPU-resident histories are pointless without IOBinding. The GPU controller is pointless without `data_gpu`. CSVCache is invisible in the rollout timer but halves the epoch time.

---

## Part 7: What We Got Wrong

### The Float32 Decode Bug

When we moved tokenization to GPU, we decoded sampled tokens using float32 bins:

```python
sampled = self._bins_f32_gpu[sample_tokens].double()  # WRONG
```

The original uses float64 bins. The float32→float64 round-trip loses ~7 decimal digits of precision. Over 500 autoregressive steps, this compounds. The baseline score jumped from 49.7 to 59.2 — a 20% regression from one line.

Fix: decode with float64 bins.

```python
sampled = self._bins_gpu[sample_tokens]  # float64, matches reference
```

### torch.bucketize vs np.digitize

They have **inverted** `right` semantics:

- `np.digitize(x, bins, right=True)` = `torch.bucketize(x, bins, right=False)`

Getting this wrong shifts every token by one position. The simulation diverges silently — no error, just wrong physics.

### The RNG Parity Bug

We pre-generated random values for all T steps. But the reference seeds the RNG at `CONTEXT_LENGTH` and only generates values for steps CL through T. Pre-generating for steps 0 through T wastes CL random draws, shifting the entire sequence.

Fix: generate only `T - CL` values, index by `step_idx - CL`.

### The /venv Incident

While trying to install TensorRT on a rented GPU instance, we deleted `/venv` to free disk space. That was the system Python installation. The virtualenv's symlink to `/venv/bin/python3` broke instantly. Lesson: know what you're deleting.

---

## Part 8: Where the Time Goes Now

```
Epoch: 12s rollout + 19s PPO update = 31s
```

The rollout breaks down as:
- **sim = 10s**: 500 sequential ONNX forward passes on GPU. This is the model's inference time — the irreducible floor for this architecture. Only TensorRT (2-4x faster inference) could cut this further.
- **ctrl = 1.3s**: All-GPU observation building + policy inference. Already fast.
- **overhead = 0.7s**: Data slicing from cache, GPU history init, reward computation.

The PPO update (19s) is gradient computation — 4 epochs × minibatch SGD over 2.4M datapoints on GPU. This is standard deep learning training time.

---

## Part 9: Lessons

**1. Measure before you optimize.** We spent time on IOBinding reuse and pre-allocated GPU buffers that saved ~0.5s. Meanwhile, CSV re-parsing was silently burning 6s per epoch. The timing prints inside the rollout loop were misleading because they didn't capture the setup cost outside the loop.

**2. Precision is not optional.** Autoregressive models amplify tiny numerical differences. A float32 decode in the physics sim corrupted the entire simulation trajectory. "Close enough" is not close enough when you're feeding predictions back as inputs 500 times.

**3. The bus is the bottleneck.** The single biggest category of optimization was eliminating CPU↔GPU data transfers. Not faster kernels, not better algorithms — just keeping data where it's already computed. Moving the controller from CPU to GPU turned 9.5s into 1.3s by removing three round-trips per step.

**4. Each layer enables the next.** GPU-resident histories enabled IOBinding. IOBinding enabled zero-copy predict. `data_gpu` enabled the all-GPU controller. CSVCache enabled fast reset. You can't cherry-pick — the optimizations compound because each one removes a barrier that was blocking the next.

**5. The reference implementation is the spec.** Every time we diverged from `tinyphysics.py`'s exact behavior — RNG consumption pattern, tokenizer `right` flag, float precision — the scores broke. The batched version must produce identical trajectories to the reference, or the policy learns on wrong rewards.

---

## Conclusion

The journey from `tinyphysics.py` to the GPU-fused batched pipeline wasn't about one clever trick. It was 7 compounding layers of optimization:

1. **Batch** episodes into one ONNX call (75x)
2. **Move to GPU** with CUDAExecutionProvider (1.6x)
3. **IOBind** inputs and outputs to GPU memory (1.25x)
4. **GPU-resident histories** to eliminate per-step transfers (1.0x — but enabled everything after)
5. **Cache CSVs** to eliminate per-epoch I/O (1.7x on epoch time)
6. **GPU controller** to eliminate all CPU numpy ops (7x on ctrl time)
7. **GPU training data** to eliminate post-rollout transpose and CPU→GPU copy (1.5x on PPO time)

Compound effect: **~250x on rollout, ~100x on wall-clock epoch time.**

The simulator that took 50 minutes to evaluate 5,000 episodes now does it in 12 seconds. A training run that would have taken a week finishes in hours. And the path to get here was paved with off-by-one RNG bugs, inverted boolean flags, and a deleted system Python installation.

Performance engineering is not about writing fast code. It's about understanding where time actually goes, and then systematically removing every unnecessary operation between where data is and where it needs to be.
