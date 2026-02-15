# How We Made a Physics Simulator Go Brrr: From 500x Sequential ONNX Calls to GPU-Fused Batched Rollouts

## Introduction

The comma.ai controls challenge asks you to write a controller that steers a car to follow a target lateral acceleration profile. Your controller is evaluated inside `tinyphysics.py` — a simulator that calls an ONNX transformer model 580 times per episode, autoregressively predicting physics one timestep at a time.

The baseline: one episode takes ~0.6 seconds. Evaluating 5,000 episodes takes ~50 minutes. Training a reinforcement learning policy over thousands of epochs at this speed is not feasible.

This is the story of how we took that 50-minute evaluation down to 8 seconds — a **375x speedup** — through a sequence of compounding optimizations, each one unlocking the next.

The path was not linear. There were wrong turns, subtle bugs, precision traps, and moments where "obvious" optimizations did nothing. Every layer taught us something about where the real bottlenecks hide.

---

## Part 1: Understanding the Baseline

### The Architecture

The simulator has a simple loop:

```
for step in range(CONTEXT_LENGTH, len(data)):   # 580 steps (20 to 600)
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

### The Cost Breakdown (measured on M4 Pro MacBook)

For a single episode (580 steps):
- **580 ONNX calls** at ~0.5ms each = ~0.3s (81% of episode time)
- Tokenize + input building = ~10ms
- Softmax + sampling = ~36ms
- State lookups (pandas `.iloc`, slicing) = ~11ms
- Controller logic (PID baseline) = ~2ms
- Total with PID: **~0.33s per episode**

But we're not running PID — we're training an RL policy. A neural-network controller adds ~0.3s per episode (forward pass through a 256-dim, 4-layer actor per step). Total with RL controller: **~0.6s per episode**.

For 5,000 episodes sequentially: **~50 minutes.**

The critical insight: every episode is independent. The ONNX model processes one `(1, 20, 4)` input — but it could just as easily process `(5000, 20, 4)`. One fat call instead of 5,000 skinny ones.

---

## Part 2: Batched Simulation (50 min → 40s)

### The Rewrite

We built `tinyphysics_batched.py` — a drop-in replacement that runs N episodes in lockstep. Instead of:

```
for episode in 5000_episodes:
    for step in 580_steps:
        predict(1, 20, 4)  →  5000 × 580 = 2,900,000 ONNX calls
```

We do:

```
for step in 580_steps:
    predict(5000, 20, 4)  →  580 ONNX calls
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

## Part 2.5: The MacBook Cluster — Scaling Out Before Scaling Up

### The Desperation

Training a PPO policy requires thousands of epochs. Each epoch rolls out 1000+ episodes through the ONNX simulator, collects trajectories, and runs a gradient update. On a single MacBook Pro (M4 Pro, 12 cores), one epoch with 1000 CSVs took **~90 seconds** — 82s for rollouts, 8s for the PPO update. At that rate, 1000 epochs would take over 24 hours. And we needed more data per epoch, not less.



Then I looked at the spare MacBook Air sitting on my desk.

### The USB-C Cluster

Two MacBooks connected via a Thunderbolt USB-C cable. No WiFi, no router — a raw 10 Gbps point-to-point link over Thunderbolt Bridge networking. Each Mac auto-assigned a `169.254.x.x` link-local IP on the bridge interface.

The first approach was SSH-based: each epoch, `scp` the checkpoint to the spare Mac, `ssh` to run a `remote_worker.py` script, `scp` the results back. It worked, but the SSH handshake overhead added up — ~2-3 seconds per epoch in connection setup alone.

We replaced it with a **persistent TCP server**. The spare Mac ran `remote_server.py` — a long-lived process listening on port 5555. The training script kept a persistent socket open and, each epoch, sent a pickle blob containing the checkpoint bytes and CSV file list. The server ran rollouts locally and streamed back a compressed NPZ with all trajectory data. Zero handshake overhead after the first connection.

### The Load Balancing Problem

The naive 50/50 split (500 CSVs each) revealed a painful truth: **the M4 Air is slow**. No fan. Passive cooling. And 6 of its 10 cores are efficiency cores — ~40% the speed of the performance cores. The main Mac would finish its 500 CSVs in 40s and then sit idle waiting for the Air to finish in 60s.

```
[split] local=500csvs 42s | r0=500csvs 58s   ← main Mac waiting 16s
```

The fix: weighted splitting via a `FRAC` environment variable.

```bash
REMOTE_HOSTS=169.254.159.243 FRAC=1.4:1 CSVS=1000 WORKERS=10 python train.py
```

`FRAC=1.4:1` means the local Mac gets 58% of CSVs, the remote gets 42%. Both finish around the same time. No idle waiting.

### The Real Insight: More Data, Not Faster Epochs

The distributed setup didn't make 1000 CSVs twice as fast — the Air was too slow for that. But it unlocked something better: **double the data in the same wall time**. With `CSVS=2000`, the main Mac processes ~1200 CSVs while the Air handles ~800. Total wall time: roughly what the main Mac alone needed for 1000. Twice the training data per PPO update. Better gradient estimates. Faster convergence.

### Three Macs, One Training Loop

Then came the question: *can I daisy-chain three MacBooks?*

The multi-remote code already supported it. Extended `REMOTE_HOSTS` to take comma-separated IPs, each running its own `remote_server.py` TCP server:

```bash
REMOTE_HOSTS=169.254.159.243,192.168.1.42 FRAC=3.5:3.5:3 CSVS=3000 python train.py
```

Three MacBooks. One main (M4 Pro), two remotes (M4 Air + a friend's machine). 3000 CSVs split proportionally across all three. The training loop sends checkpoint bytes to both remotes in parallel, kicks off local rollouts simultaneously, and merges all results when the slowest machine finishes.

```
[split] local=1050csvs 43s | r0=1050csvs 41s | r1=900csvs 44s
```

**3000 CSVs in ~45 seconds.** The same amount of data that would have taken 270 seconds on a single Mac.

### The Thermal Wall

The MacBook cluster had a shelf life. After 30+ minutes of sustained 100% CPU load, the fanless Air started thermal throttling — epoch times crept from 45s to 55s to 65s. Even the Pro throttled slightly. Pointing a desk fan at the Air helped (seriously). But this was a band-aid.

```
E 0   ⏱45s     ← cold
E 25  ⏱55s     ← warm
E 50  ⏱65s     ← throttled
```

The cluster got us through the early training phase — from random policy to a val cost of ~55. But to push further, we needed real compute. That meant the cloud.

### The Cloud Migration

First stop: **Google Cloud** — a 48-core CPU instance. 5000 CSVs in 6 seconds. The MacBook cluster was immediately obsolete.

But the CPU bottleneck was fundamental. ONNX inference is sequential per episode (autoregressive), and each worker occupies an entire core for ~0.5s. Cores scale linearly; GPUs scale with parallelism.

Second stop: **Vast.ai** — a rented RTX 5060 Ti. This is where the real story begins. The batched simulator rewrite (Part 3 onward) made the entire multi-machine approach irrelevant. Instead of distributing 5000 episodes across dozens of CPU cores, we run them all in a single GPU batch. One ONNX call. One kernel launch. The "cluster" became a single GPU tensor.

But the MacBook cluster phase taught us something important: **the first 80% of scaling is just about showing up with more hardware**. The last 20% — the GPU rewrite, IOBinding, TensorRT — that's where the engineering lives. The three-MacBook cluster was beautifully scrappy, held together by USB-C cables and pickle sockets, and it got us from a random policy to a competitive one. Not every optimization needs to be elegant.

---

## Part 3: Moving to GPU (40s → 20s rollout)

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

The default ONNX GPU path: numpy array → CPU staging buffer → GPU → inference → GPU → CPU staging → numpy array. Every step. 580 times.

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

## Part 4: The CSV Trap (33s → 20s per epoch)

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

The 13s of hidden overhead collapsed to <1s. Epoch time: **33s → ~20s** (the rollout itself was still ~20s — ctrl + sim hadn't changed yet). But the epoch no longer wasted half its time on data plumbing.

---

## Part 5: All-GPU Controller (20s → 12s rollout)

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

Three CPU↔GPU round-trips per step. 580 steps. For 5,000 episodes. All because the observation building was in numpy.

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
| MacBook Cluster | 3 Macs, persistent TCP, weighted splits | ~45s (3000 CSVs) | ~55s | 3x data/time |
| GPU ONNX | CUDAExecutionProvider, single process | ~25s | ~45s | 120x |
| IOBinding + GPU tokenize | Zero-copy ONNX I/O, torch.bucketize | ~20s | ~40s | 150x |
| GPU-resident histories | Histories as CUDA tensors | ~20s | ~33s | 150x |
| CSVCache | Parse once at startup | ~20s | ~20s | 150x (epoch: 150x) |
| GPU controller + data_gpu | All obs/ctrl/data on GPU | ~12s | ~12s + 19s PPO | 250x |
| TensorRT FP16 | Fused kernels, engine caching | ~8s | ~8s + 3s PPO | 375x |
| Final shave-offs | GPU randperm, GPU cost, GPU stats, 100K mini-batch | ~8s | ~8s + 3s PPO | ~375x |

The physics sim went from 50 minutes to 8 seconds. Each layer was necessary — you couldn't skip to the end. GPU-resident histories are pointless without IOBinding. The GPU controller is pointless without `data_gpu`. CSVCache is invisible in the rollout timer but halves the epoch time.

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

## Part 8: TensorRT — Halving the Sim Floor (10s → 5s)

### The Opportunity

The sim phase — 580 sequential ONNX forward passes — was 10s and appeared to be the irreducible floor. But the ONNX model was running through the default CUDA Execution Provider, which executes each op as a separate CUDA kernel. TensorRT fuses layers, auto-tunes kernels for the specific GPU, and can run in FP16 where safe.

### Wiring It In

ONNX Runtime supports TensorRT as an execution provider. The integration was straightforward:

```python
providers = []
if os.environ.get('TRT') == '1':
    providers.append(('TensorrtExecutionProvider', {
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': '/tmp/trt_cache',
        'trt_max_workspace_size': str(2 << 30),  # 2GB
    }))
providers.append(('CUDAExecutionProvider', {...}))
```

First epoch pays a ~30s TRT engine compilation cost. After that, the cached engine loads in <1s.

### FP16 LayerNorm Safety

TensorRT warned about FP16 overflow in layernorm nodes after self-attention. It automatically forced those specific Reduce/Pow layers back to FP32 — handling mixed precision at the op level. No accuracy loss, no manual intervention needed.

### Result

```
sim = 10.1s  →  5.1s  (TRT, after engine cache warm)
```

The sim floor nearly halved. For N=5000, total rollout dropped from ~12s to ~8s.

---

## Part 9: Final Shave-offs — Eliminating the Last CPU↔GPU Transfers

### The Audit

With TRT handling the sim floor, we audited every remaining CPU↔GPU transfer in the pipeline:

| Transfer | Location | Size | Direction |
|---|---|---|---|
| `torch.randperm(N)` | PPO update mini-batch loop | 1.6M int64 | CPU→GPU (hundreds per update) |
| `pred.cpu().numpy()` | `compute_cost()` | (N, 400) float | GPU→CPU |
| `rew_2d.cpu().numpy()` | `gae_gpu` ret_rms update | (N, S) float | GPU→CPU |

### Fix 1: GPU randperm

```python
# Before — creates permutation on CPU, implicit transfer every mini-batch:
for idx in torch.randperm(N).split(MINI_BS):

# After — permutation lives on GPU, indexing stays on-device:
for idx in torch.randperm(N, device=obs_t.device).split(MINI_BS):
```

With ~1.6M transitions and K=4 PPO epochs, that's hundreds of implicit CPU→GPU index transfers per update, all eliminated.

### Fix 2: GPU compute_cost

```python
# Before — pull full prediction history to CPU, compute in numpy:
pred = pred.cpu().numpy()
lat_cost = np.mean((target - pred)**2, axis=1) * 100

# After — compute entirely on GPU, transfer only final (N,) results:
lat_cost = (target_gpu - pred_gpu).pow(2).mean(dim=1) * 100
jerk = torch.diff(pred_gpu, dim=1) / DEL_T
jerk_cost = jerk.pow(2).mean(dim=1) * 100
total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
return {'total_cost': total.cpu().numpy()}  # (N,) not (N, 400)
```

Transfers 3 vectors of size N instead of an (N, 400) matrix.

### Fix 3: GPU running stats

The return normalization (RunningMeanStd) was pulling the entire (N, S) reward tensor to CPU just to compute mean and variance:

```python
# Before — 6.4MB GPU→CPU transfer:
self._ret_rms.update(rew_2d.detach().cpu().numpy().ravel())

# After — compute moments on GPU, transfer 2 scalars:
flat = rew_2d.reshape(-1)
self._ret_rms.update_from_moments(flat.mean().item(), flat.var().item(), flat.numel())
```

### Fix 4: Larger Mini-batches + Faster Zeroing

Increased `MINI_BS` from 20K to 100K to reduce kernel launch overhead per mini-batch, and used `optimizer.zero_grad(set_to_none=True)` to skip memset.

### Result

```
Before:  ⏱6+3s  (N=4000)   per-csv: 2.25ms
After:   ⏱8+3s  (N=5000)   per-csv: 2.20ms
```

Modest per-sample improvement. The gains are small because the pipeline was already well-optimized — the 5s TRT sim floor dominates.

---

## Part 10: Where the Time Goes Now

```
Epoch: ~8s rollout + ~3s PPO update = ~11s  (N=5000)
```

The rollout breaks down as:
- **sim = 5-6s**: 580 sequential TensorRT forward passes. This is the hard floor — autoregressive inference cannot be parallelized across timesteps.
- **ctrl = 1.2s**: All-GPU observation building + policy inference. 580 steps × 2.1ms each, dominated by kernel launch overhead for many small ops.
- **overhead < 0.5s**: GPU data slicing from cache, history init, cost computation.

The PPO update (3s) processes ~2.5M datapoints through 4 epochs of mini-batch SGD with 100K batch size. GPU randperm, GPU GAE, and `set_to_none=True` zeroing keep everything on-device.

---

## Part 11: Lessons

**1. Measure before you optimize.** We spent time on IOBinding reuse and pre-allocated GPU buffers that saved ~0.5s. Meanwhile, CSV re-parsing was silently burning 6s per epoch. The timing prints inside the rollout loop were misleading because they didn't capture the setup cost outside the loop.

**2. Precision is not optional.** Autoregressive models amplify tiny numerical differences. A float32 decode in the physics sim corrupted the entire simulation trajectory. "Close enough" is not close enough when you're feeding predictions back as inputs 500 times.

**3. The bus is the bottleneck.** The single biggest category of optimization was eliminating CPU↔GPU data transfers. Not faster kernels, not better algorithms — just keeping data where it's already computed. Moving the controller from CPU to GPU turned 9.5s into 1.3s by removing three round-trips per step.

**4. Each layer enables the next.** GPU-resident histories enabled IOBinding. IOBinding enabled zero-copy predict. `data_gpu` enabled the all-GPU controller. CSVCache enabled fast reset. You can't cherry-pick — the optimizations compound because each one removes a barrier that was blocking the next.

**5. The reference implementation is the spec.** Every time we diverged from `tinyphysics.py`'s exact behavior — RNG consumption pattern, tokenizer `right` flag, float precision — the scores broke. The batched version must produce identical trajectories to the reference, or the policy learns on wrong rewards.

---

## Conclusion

The journey from `tinyphysics.py` to the GPU-fused batched pipeline wasn't about one clever trick. It was a scrappy, compounding sequence of optimizations — some elegant, some held together with USB-C cables and desk fans:

1. **Batch** episodes into one ONNX call (75x)
2. **MacBook cluster** — 3 Macs over Thunderbolt, persistent TCP, weighted splits (3x data throughput)
3. **Move to GPU** with CUDAExecutionProvider (1.6x)
4. **IOBind** inputs and outputs to GPU memory (1.25x)
5. **GPU-resident histories** to eliminate per-step transfers (1.0x — but enabled everything after)
6. **Cache CSVs** to eliminate per-epoch I/O (1.7x on epoch time)
7. **GPU controller** to eliminate all CPU numpy ops (7x on ctrl time)
8. **GPU training data** to eliminate post-rollout transpose and CPU→GPU copy (1.5x on PPO time)
9. **TensorRT** FP16 inference with engine caching (2x on sim time)
10. **Final shave-offs** — GPU randperm, GPU cost computation, GPU running stats, larger mini-batches (1.1x)

Compound effect: **~375x on rollout** (50 min → 8s for 5000 episodes).

The simulator that took 50 minutes to evaluate 5,000 episodes now does it in 8 seconds. A training run that would have taken a week finishes in hours. And the path to get here was paved with off-by-one RNG bugs, inverted boolean flags, a deleted system Python installation, thermal-throttled MacBook Airs, and pickle blobs flying over Thunderbolt cables.

Performance engineering is not about writing fast code. It's about understanding where time actually goes, and then systematically removing every unnecessary operation between where data is and where it needs to be. Sometimes that means fusing GPU kernels. Sometimes it means plugging a USB-C cable into your friend's laptop.
