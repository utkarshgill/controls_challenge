"""
Batched TinyPhysics simulator — vectorized reimplementation of tinyphysics.py.

Runs N episodes in lockstep with a single ONNX call per timestep instead of N
individual calls.  Every method mirrors the corresponding method in
tinyphysics.py so the two files can be diffed line-by-line.
"""

import multiprocessing
import numpy as np
import onnxruntime as ort
import pandas as pd
import os

from hashlib import md5
from pathlib import Path
from typing import List, Callable, Dict

from tinyphysics import (
    ACC_G, CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    VOCAB_SIZE, LATACCEL_RANGE, STEER_RANGE, MAX_ACC_DELTA,
    DEL_T, LAT_ACCEL_COST_MULTIPLIER, FUTURE_PLAN_STEPS,
    LataccelTokenizer,
)


# ── Per-worker ONNX cache ─────────────────────────────────────

_pool_cache = {}


ORT_THREADS = int(os.getenv('ORT_THREADS', '1'))


def make_ort_session(model_path):
    """Create an ONNX Runtime session (same options as TinyPhysicsModel)."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_THREADS
    options.inter_op_num_threads = ORT_THREADS
    options.log_severity_level = 3
    use_cuda = os.getenv('CUDA', '0') == '1'
    use_trt = os.getenv('TRT', '0') == '1'
    if use_trt:
        trt_opts = {
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(Path(model_path).parent),
            'trt_max_workspace_size': str(2 << 30),  # 2 GB
        }
        providers = [
            ('TensorrtExecutionProvider', trt_opts),
            ('CUDAExecutionProvider', {}),
            ('CPUExecutionProvider', {}),
        ]
    elif use_cuda:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    with open(str(model_path), 'rb') as f:
        sess = ort.InferenceSession(f.read(), options, providers)
        actual = sess.get_providers()
        print(f"[ORT] requested={[p if isinstance(p,str) else p[0] for p in providers]}  actual={actual}")
        return sess


def pool_init(model_path):
    """Per-worker initializer: cache ONNX session + model path."""
    _pool_cache['model_path'] = str(model_path)
    _pool_cache['ort_session'] = make_ort_session(model_path)


def get_pool_cache():
    """Access per-worker cached ONNX session from pool workers."""
    return _pool_cache


# ── Parallel helpers ──────────────────────────────────────────

def chunk_list(lst, n_chunks):
    """Split list into n_chunks roughly equal pieces."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n_chunks)]


def run_parallel_chunked(pool, csv_files, worker_fn, n_workers, extra_args=()):
    """Chunk csv_files across workers, map worker_fn, flatten results."""
    chunks = chunk_list(csv_files, n_workers)
    args = [(chunk, *extra_args) for chunk in chunks if chunk]
    chunk_results = pool.map(worker_fn, args)
    flat = []
    for cr in chunk_results:
        if isinstance(cr, list):
            flat.extend(cr)
        else:
            flat.append(cr)
    return flat


# ── CSV loading ──────────────────────────────────────────────

def _parse_one_csv(path):
    df = pd.read_csv(str(path))
    return (np.sin(df['roll'].values) * ACC_G,
            df['vEgo'].values, df['aEgo'].values,
            df['targetLateralAcceleration'].values,
            -df['steerCommand'].values)


def preload_csvs(csv_files):
    """Load N CSVs into (N, T) numpy arrays via multiprocessing.

    Caches result to .npy_cache/ so subsequent runs load in <1s.
    """
    cache_dir = Path(csv_files[0]).parent / '.npy_cache'
    cache_file = cache_dir / f'n{len(csv_files)}.npz'
    if cache_file.exists():
        d = np.load(cache_file)
        return {k: d[k] for k in ('roll_lataccel', 'v_ego', 'a_ego',
                'target_lataccel', 'steer_command', 'N', 'T')}

    import multiprocessing as mp
    with mp.Pool(min(mp.cpu_count(), len(csv_files))) as pool:
        rows = pool.map(_parse_one_csv, csv_files)
    N = len(rows)
    T = max(len(r[0]) for r in rows)
    arrs = [np.empty((N, T), np.float64) for _ in range(5)]
    for i, cols in enumerate(rows):
        L = len(cols[0])
        for j, col in enumerate(cols):
            arrs[j][i, :L] = col
            if L < T: arrs[j][i, L:] = col[-1]
    names = ['roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'steer_command']
    result = {n: a for n, a in zip(names, arrs)}
    result['N'], result['T'] = np.int64(N), np.int64(T)

    cache_dir.mkdir(exist_ok=True)
    np.savez(cache_file, **result)
    return result


class CSVCache:
    """Load all CSVs once, store per-file (1, T) rows.  Slice by index per epoch."""

    def __init__(self, csv_files):
        import time as _t
        t0 = _t.time()
        self._files = list(csv_files)
        self._file_to_idx = {str(f): i for i, f in enumerate(self._files)}
        self._master = preload_csvs(self._files)  # (N_all, T)
        self.T = self._master['T']
        # Pre-compute per-file RNG seed + random values (deterministic)
        CL = CONTEXT_LENGTH
        n_steps = self.T - CL
        N_all = len(self._files)
        self._rng_all = np.empty((N_all, n_steps), dtype=np.float64)
        for i, f in enumerate(self._files):
            seed = int(md5(str(f).encode()).hexdigest(), 16) % 10**4
            rng = np.random.RandomState(seed)
            self._rng_all[i, :] = rng.rand(n_steps)
        print(f"  [CSVCache] {N_all} files, T={self.T}, "
              f"loaded in {_t.time()-t0:.1f}s", flush=True)

    def slice(self, csv_files):
        """Return (data_dict, rng_rows) for a subset of files."""
        idxs = np.array([self._file_to_idx[str(f)] for f in csv_files])
        N = len(idxs)
        data = {}
        for k in ('roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'steer_command'):
            data[k] = self._master[k][idxs]
        data['N'] = N
        data['T'] = self.T
        return data, self._rng_all[idxs]  # rng shape (N, n_steps)


# ── Batched physics model (mirrors TinyPhysicsModel) ──────────

class BatchedPhysicsModel:
    """Vectorized drop-in for TinyPhysicsModel.

    Mirrors softmax / predict / get_current_lataccel but operates on (N, ...)
    arrays so the two classes can be diffed line-by-line.
    """

    def __init__(self, model_path: str, ort_session=None) -> None:
        self.tokenizer = LataccelTokenizer()
        if ort_session is not None:
            self.ort_session = ort_session
        else:
            self.ort_session = make_ort_session(model_path)
        self._use_gpu = os.getenv('CUDA', '0') == '1'
        self._cached_N = 0   # for lazy GPU buffer allocation
        if self._use_gpu:
            import torch
            self._torch = torch
            self._out_name = self.ort_session.get_outputs()[0].name
            self._last_probs_gpu = None
            self._io = self.ort_session.io_binding()  # reused every step
            # GPU-resident tokenizer bins for torch.bucketize
            self._bins_gpu = torch.from_numpy(self.tokenizer.bins.astype(np.float64)).cuda()
            self._bins_f32_gpu = self._bins_gpu.float()
            self._lat_lo = float(LATACCEL_RANGE[0])
            self._lat_hi = float(LATACCEL_RANGE[1])

    def softmax(self, x, axis=-1):
        """Mirrors TinyPhysicsModel.softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=0.8,
                rng_u=None, rngs=None) -> np.ndarray:
        if self._use_gpu:
            return self._predict_gpu(input_data, temperature, rng_u)
        return self._predict_cpu(input_data, temperature, rng_u, rngs)

    def _ensure_gpu_bufs(self, N, CL):
        """Lazily allocate / resize GPU buffers when N changes."""
        if N != self._cached_N:
            torch = self._torch
            self._out_gpu = torch.empty((N, CL, VOCAB_SIZE),
                                        dtype=torch.float32, device='cuda')
            self._states_gpu = torch.empty((N, CL, 4),
                                           dtype=torch.float32, device='cuda')
            self._tokens_gpu = torch.empty((N, CL),
                                           dtype=torch.int64, device='cuda')
            self._clamped_buf = torch.empty((N, CL),
                                            dtype=torch.float64, device='cuda')
            # Cache shape lists for IOBinding (avoid per-step list() calls)
            self._states_shape = [N, CL, 4]
            self._tokens_shape = [N, CL]
            self._out_shape = [N, CL, VOCAB_SIZE]
            self._cached_N = N

    def _predict_gpu(self, input_data, temperature, rng_u):
        """All-GPU IOBinding path.  Accepts either numpy or torch GPU tensors
        for states/tokens.  If torch GPU tensors, zero CPU→GPU transfer."""
        torch = self._torch
        states = input_data['states']
        tokens = input_data['tokens']

        if isinstance(states, torch.Tensor):
            # Already on GPU — use directly
            states_gpu = states
            tokens_gpu = tokens
            N, CL = states.shape[:2]
            self._ensure_gpu_bufs(N, CL)
        else:
            # Numpy path — copy to GPU
            N, CL = states.shape[:2]
            self._ensure_gpu_bufs(N, CL)
            self._states_gpu.copy_(torch.from_numpy(states))
            self._tokens_gpu.copy_(torch.from_numpy(tokens))
            states_gpu = self._states_gpu
            tokens_gpu = self._tokens_gpu

        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        io.bind_input('states', 'cuda', 0, np.float32,
                      self._states_shape, states_gpu.data_ptr())
        io.bind_input('tokens', 'cuda', 0, np.int64,
                      self._tokens_shape, tokens_gpu.data_ptr())
        io.bind_output(self._out_name, 'cuda', 0, np.float32,
                       self._out_shape, self._out_gpu.data_ptr())

        self.ort_session.run_with_iobinding(io)

        # Softmax on GPU (only last timestep)
        probs = torch.softmax(self._out_gpu[:, -1, :] / temperature, dim=-1)
        self._last_probs_gpu = probs                   # keep on GPU
        self._last_probs = None                         # lazy CPU copy

        # Sampling on GPU — match np.random.choice: normalize CDF, searchsorted right
        cdf = torch.cumsum(probs, dim=1)
        cdf = cdf / cdf[:, -1:]  # normalize like np.random.choice
        if rng_u is not None:
            u = rng_u.unsqueeze(1) if rng_u.dim() == 1 else rng_u
        else:
            u = torch.rand(N, 1, device='cuda', dtype=torch.float64)
        samples = torch.searchsorted(cdf.double(), u.double()).squeeze(1).clamp(0, VOCAB_SIZE - 1)
        return samples

    def _predict_cpu(self, input_data, temperature, rng_u, rngs):
        """Original CPU path."""
        res = self.ort_session.run(None, input_data)[0]  # (N, CL, VOCAB_SIZE)
        probs = self.softmax(res / temperature, axis=-1)
        probs = probs[:, -1, :]
        self._last_probs = probs
        N = probs.shape[0]
        cdf = np.cumsum(probs, axis=1)
        cdf /= cdf[:, -1:]  # normalize like np.random.choice
        if rng_u is not None:
            u = rng_u.astype(np.float64)
        elif rngs is not None:
            u = np.array([rng.rand() for rng in rngs], dtype=np.float64)
        else:
            u = np.random.rand(N)
        samples = np.array([np.searchsorted(cdf[i], u[i], side='right') for i in range(N)], dtype=np.intp)
        return np.clip(samples, 0, VOCAB_SIZE - 1)

    def get_current_lataccel(self, sim_states, actions, past_preds,
                             rng_u=None, rngs=None,
                             return_expected: bool = False):
        """Batched get_current_lataccel.  Accepts numpy or torch GPU tensors.
        When GPU tensors: tokenize + build states on GPU, zero CPU transfer."""
        torch = getattr(self, '_torch', None)

        if self._use_gpu and torch is not None and isinstance(actions, torch.Tensor):
            return self._get_current_lataccel_gpu(
                sim_states, actions, past_preds, rng_u, return_expected)

        # CPU path (always use _predict_cpu to get numpy indices)
        N, CL = actions.shape
        if not hasattr(self, '_states_buf') or self._states_buf.shape[0] != N:
            self._states_buf = np.empty((N, CL, 4), np.float32)
            self._tokens_buf = np.empty((N, CL), np.int64)
        self._tokens_buf[:] = self.tokenizer.encode(past_preds)
        self._states_buf[:, :, 0] = actions
        self._states_buf[:, :, 1:] = sim_states
        input_data = {'states': self._states_buf, 'tokens': self._tokens_buf}
        sampled = self.tokenizer.decode(self._predict_cpu(input_data, temperature=0.8,
                                                          rng_u=rng_u, rngs=rngs))
        if not return_expected:
            return sampled
        if self._last_probs is None:
            self._last_probs = self._last_probs_gpu.cpu().numpy()
        expected = np.sum(self._last_probs * self.tokenizer.bins[None, :], axis=-1)
        return sampled, expected

    def _get_current_lataccel_gpu(self, sim_states, actions, past_preds,
                                   rng_u, return_expected):
        """All-GPU path: tokenize via torch.bucketize, build states,
        predict, decode — all on GPU.  Returns GPU tensor.
        Reuses pre-allocated buffers (zero per-step CUDA mallocs)."""
        torch = self._torch
        N, CL = actions.shape
        self._ensure_gpu_bufs(N, CL)

        # Tokenize on GPU: clamp in-place, bucketize
        torch.clamp(past_preds, self._lat_lo, self._lat_hi, out=self._clamped_buf)
        tokens = torch.bucketize(self._clamped_buf, self._bins_gpu, right=False)

        # Build states in pre-allocated buffer
        self._states_gpu[:, :, 0] = actions.float()
        self._states_gpu[:, :, 1:] = sim_states.float()

        input_data = {'states': self._states_gpu, 'tokens': tokens}
        sample_tokens = self.predict(input_data, temperature=0.8, rng_u=rng_u)
        sampled = self._bins_gpu[sample_tokens]

        if not return_expected:
            return sampled
        probs = self._last_probs_gpu
        expected = (probs * self._bins_f32_gpu.unsqueeze(0)).sum(dim=-1).double()
        return sampled, expected


# ── Batched simulator (mirrors TinyPhysicsSimulator) ──────────

class BatchedSimulator:
    """Vectorized drop-in for TinyPhysicsSimulator.

    Mirrors the original's reset / step / sim_step / control_step / rollout /
    compute_cost structure but operates on (N, ...) arrays.
    """

    def __init__(self, model_path: str, csv_files: list = None,
                 ort_session=None, cached_data=None, cached_rng=None) -> None:
        self.sim_model = BatchedPhysicsModel(model_path, ort_session=ort_session)
        self.csv_files = csv_files or []
        if cached_data is not None:
            self.data = cached_data
            self._cached_rng = cached_rng        # (N, n_steps) from CSVCache
        else:
            self.data = preload_csvs(csv_files)
            self._cached_rng = None
        self.N = self.data['N']
        self.T = self.data['T']
        self.compute_expected = False
        self.expected_lataccel = None
        self._gpu = os.getenv('CUDA', '0') == '1'
        if self._gpu:
            import torch as _torch
            self._torch = _torch
            # Move data dict to GPU once (read-only, reused every step)
            self.data_gpu = {}
            for k in ('roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'steer_command'):
                arr = np.ascontiguousarray(self.data[k], dtype=np.float64)
                self.data_gpu[k] = _torch.from_numpy(arr).cuda()
        else:
            self._torch = None
            self.data_gpu = None
        self.reset()

    # ── reset  (mirrors tinyphysics.py lines 110-120) ────────

    def reset(self) -> None:
        N, T = self.N, self.T
        CL = CONTEXT_LENGTH

        self._hist_len = CL

        # RNG: use cached values if available, else generate
        n_steps = T - CL
        if self._cached_rng is not None:
            # cached_rng is (N, n_steps), we need (n_steps, N)
            self._rng_all = self._cached_rng.T.copy()
        else:
            self.rngs = []
            for f in getattr(self, 'csv_files', []):
                seed = int(md5(str(f).encode()).hexdigest(), 16) % 10**4
                self.rngs.append(np.random.RandomState(seed))
            self._rng_all = np.empty((n_steps, N), dtype=np.float64)
            for i, rng in enumerate(self.rngs):
                self._rng_all[:, i] = rng.rand(n_steps)

        if self._gpu:
            _torch = self._torch
            self.action_history = _torch.zeros((N, T), dtype=_torch.float64, device='cuda')
            self.action_history[:, :CL] = self.data_gpu['steer_command'][:, :CL]
            self.state_history = _torch.zeros((N, T, 3), dtype=_torch.float64, device='cuda')
            self.state_history[:, :CL, 0] = self.data_gpu['roll_lataccel'][:, :CL]
            self.state_history[:, :CL, 1] = self.data_gpu['v_ego'][:, :CL]
            self.state_history[:, :CL, 2] = self.data_gpu['a_ego'][:, :CL]
            self.current_lataccel_history = _torch.zeros((N, T), dtype=_torch.float64, device='cuda')
            self.current_lataccel_history[:, :CL] = self.data_gpu['target_lataccel'][:, :CL]
            self.current_lataccel = self.current_lataccel_history[:, CL - 1].clone()
            self._rng_all_gpu = _torch.from_numpy(self._rng_all).cuda()
        else:
            self.action_history = np.zeros((N, T), np.float64)
            self.action_history[:, :CL] = self.data['steer_command'][:, :CL]
            self.state_history = np.zeros((N, T, 3), np.float64)
            self.state_history[:, :CL, 0] = self.data['roll_lataccel'][:, :CL]
            self.state_history[:, :CL, 1] = self.data['v_ego'][:, :CL]
            self.state_history[:, :CL, 2] = self.data['a_ego'][:, :CL]
            self.current_lataccel_history = np.zeros((N, T), np.float64)
            self.current_lataccel_history[:, :CL] = self.data['target_lataccel'][:, :CL]
            self.current_lataccel = self.current_lataccel_history[:, CL - 1].copy()
            self._rng_all_gpu = None

    # ── get_state_target_futureplan  (mirrors lines 154-165) ─

    def get_state_target_futureplan(self, step_idx: int):
        """Returns (roll_la, v_ego, a_ego, target) as (N,) arrays,
        plus future_plan dict with (N, K) arrays."""
        d = self.data
        T = self.T
        roll_la = d['roll_lataccel'][:, step_idx]
        v_ego   = d['v_ego'][:, step_idx]
        a_ego   = d['a_ego'][:, step_idx]
        target  = d['target_lataccel'][:, step_idx]

        end = min(step_idx + FUTURE_PLAN_STEPS, T)
        future_plan = {
            'lataccel':      d['target_lataccel'][:, step_idx+1:end],
            'roll_lataccel': d['roll_lataccel'][:, step_idx+1:end],
            'v_ego':         d['v_ego'][:, step_idx+1:end],
            'a_ego':         d['a_ego'][:, step_idx+1:end],
        }
        return roll_la, v_ego, a_ego, target, future_plan

    # ── sim_step  (mirrors lines 133-145) ────────────────────

    def sim_step(self, step_idx: int) -> None:
        """Batched ONNX physics prediction.

        Mirrors:
          pred = self.sim_model.get_current_lataccel(...)
          pred = clip(pred, current ± MAX_ACC_DELTA)
          if step >= CONTROL_START: current = pred  else: current = target
          current_lataccel_history.append(current)

        If self.compute_expected is True, also stores the expected (mean)
        lataccel from the probability distribution in self.expected_lataccel.
        """
        CL = CONTEXT_LENGTH
        h = self._hist_len
        rng_idx = step_idx - CL

        if self._gpu:
            torch = self._torch
            rng_u = self._rng_all_gpu[rng_idx]
            # All slices are GPU tensors — entire call stays on GPU
            result = self.sim_model.get_current_lataccel(
                sim_states=self.state_history[:, h-CL+1:h+1, :],
                actions=self.action_history[:, h-CL+1:h+1],
                past_preds=self.current_lataccel_history[:, h-CL:h],
                rng_u=rng_u,
                return_expected=self.compute_expected,
            )
            if self.compute_expected:
                pred, self.expected_lataccel = result
            else:
                pred = result

            pred = torch.clamp(pred,
                               self.current_lataccel - MAX_ACC_DELTA,
                               self.current_lataccel + MAX_ACC_DELTA)
            self.raw_pred = pred
            if step_idx >= CONTROL_START_IDX:
                self.current_lataccel = pred
            else:
                self.current_lataccel = self.data_gpu['target_lataccel'][:, step_idx].clone()

            self.current_lataccel_history[:, h] = self.current_lataccel
            self._hist_len += 1
        else:
            rng_u = self._rng_all[rng_idx]
            result = self.sim_model.get_current_lataccel(
                sim_states=self.state_history[:, h-CL+1:h+1, :],
                actions=self.action_history[:, h-CL+1:h+1],
                past_preds=self.current_lataccel_history[:, h-CL:h],
                rng_u=rng_u,
                return_expected=self.compute_expected,
            )
            if self.compute_expected:
                pred, self.expected_lataccel = result
            else:
                pred = result

            pred = np.clip(pred,
                           self.current_lataccel - MAX_ACC_DELTA,
                           self.current_lataccel + MAX_ACC_DELTA)
            self.raw_pred = pred
            if step_idx >= CONTROL_START_IDX:
                self.current_lataccel = pred
            else:
                self.current_lataccel = self.data['target_lataccel'][:, step_idx].copy()

            self.current_lataccel_history[:, h] = self.current_lataccel
            self._hist_len += 1

    # ── control_step  (mirrors lines 147-152) ────────────────

    def control_step(self, step_idx: int, actions) -> None:
        """Accept externally-provided actions (N,), clip, append to history.

        Mirrors:
          if step < CONTROL_START: action = CSV steer
          action = clip(action, STEER_RANGE)
          action_history.append(action)
        """
        if self._gpu:
            torch = self._torch
            if step_idx < CONTROL_START_IDX:
                actions = self.data_gpu['steer_command'][:, step_idx]
            elif not isinstance(actions, torch.Tensor):
                actions = self.action_history.new_tensor(actions)
            actions = torch.clamp(actions, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history[:, self._hist_len] = actions
        else:
            if step_idx < CONTROL_START_IDX:
                actions = self.data['steer_command'][:, step_idx].copy()
            actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history[:, self._hist_len] = actions

    # ── step  (mirrors lines 167-174) ────────────────────────

    def step(self, step_idx: int, actions) -> dict:
        """One full sim step.  Returns state info for the controller.

        Mirrors:
          state, target, futureplan = get_state_target_futureplan(step_idx)
          state_history.append(state)
          target_lataccel_history.append(target)
          control_step(step_idx)  → uses externally provided actions
          sim_step(step_idx)
        """
        roll_la, v_ego, a_ego, target, future_plan = \
            self.get_state_target_futureplan(step_idx)

        h = self._hist_len
        if self._gpu:
            state_np = np.stack([roll_la, v_ego, a_ego], axis=-1)  # (N,3)
            self.state_history[:, h, :] = self._torch.from_numpy(
                np.ascontiguousarray(state_np)).cuda()
        else:
            self.state_history[:, h, 0] = roll_la
            self.state_history[:, h, 1] = v_ego
            self.state_history[:, h, 2] = a_ego

        self.control_step(step_idx, actions)
        self.sim_step(step_idx)

        if self._gpu:
            cur_la = self.current_lataccel.cpu().numpy()
        else:
            cur_la = self.current_lataccel.copy()

        return dict(
            roll_lataccel=roll_la, v_ego=v_ego, a_ego=a_ego,
            target=target, future_plan=future_plan,
            current_lataccel=cur_la,
        )

    # ── rollout  (mirrors lines 195-213) ─────────────────────

    def rollout(self, controller_fn: Callable) -> Dict[str, np.ndarray]:
        """Run full rollout.

        GPU path: controller_fn(step_idx, sim) → GPU tensor actions (N,)
          where sim has .data_gpu, .current_lataccel (GPU tensors)
        CPU path: controller_fn(step_idx, target, current_lataccel, state_dict, future_plan)
            → numpy actions (N,)

        Returns dict with 'total_cost', 'lataccel_cost', 'jerk_cost' as (N,) arrays.
        """
        import time as _time
        t_ctrl, t_sim = 0.0, 0.0
        CL = CONTEXT_LENGTH

        if self._gpu:
            _torch = self._torch
            dg = self.data_gpu
            for step_idx in range(CL, self.T):
                _t0 = _time.perf_counter()
                # Controller receives step_idx + sim reference (all GPU)
                actions = controller_fn(step_idx, self)
                t_ctrl += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                h = self._hist_len
                # Write state from GPU data dict (zero CPU transfer)
                self.state_history[:, h, 0] = dg['roll_lataccel'][:, step_idx]
                self.state_history[:, h, 1] = dg['v_ego'][:, step_idx]
                self.state_history[:, h, 2] = dg['a_ego'][:, step_idx]

                self.control_step(step_idx, actions)
                self.sim_step(step_idx)
                t_sim += _time.perf_counter() - _t0
        else:
            for step_idx in range(CL, self.T):
                roll_la, v_ego, a_ego, target, future_plan = \
                    self.get_state_target_futureplan(step_idx)
                state_dict = dict(roll_lataccel=roll_la, v_ego=v_ego, a_ego=a_ego)

                cur_la_np = self.current_lataccel.copy()
                _t0 = _time.perf_counter()
                actions = controller_fn(
                    step_idx, target, cur_la_np,
                    state_dict, future_plan)
                t_ctrl += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                h = self._hist_len
                self.state_history[:, h, 0] = roll_la
                self.state_history[:, h, 1] = v_ego
                self.state_history[:, h, 2] = a_ego
                self.control_step(step_idx, actions)
                self.sim_step(step_idx)
                t_sim += _time.perf_counter() - _t0

        if int(os.environ.get('DEBUG', '0')) >= 2:
            print(f"  [rollout N={self.N}] ctrl={t_ctrl:.1f}s  sim={t_sim:.1f}s  total={t_ctrl+t_sim:.1f}s", flush=True)
        return self.compute_cost()

    # ── compute_cost  (mirrors lines 186-193) ────────────────

    def compute_cost(self) -> Dict[str, np.ndarray]:
        """Vectorized cost computation.  Returns dict with (N,) arrays."""
        if self._gpu:
            _torch = self._torch
            target_gpu = self.data_gpu['target_lataccel'][:, CONTROL_START_IDX:COST_END_IDX]
            pred_gpu = self.current_lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
            lat_accel_cost = (target_gpu - pred_gpu).pow(2).mean(dim=1) * 100
            jerk = _torch.diff(pred_gpu, dim=1) / DEL_T
            jerk_cost = jerk.pow(2).mean(dim=1) * 100
            total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
            # Single GPU->CPU transfer of final (N,) results
            return {
                'lataccel_cost': lat_accel_cost.cpu().numpy(),
                'jerk_cost': jerk_cost.cpu().numpy(),
                'total_cost': total_cost.cpu().numpy(),
            }
        target = self.data['target_lataccel'][:, CONTROL_START_IDX:COST_END_IDX]
        pred = self.current_lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
        lat_accel_cost = np.mean((target - pred)**2, axis=1) * 100
        jerk_cost = np.mean((np.diff(pred, axis=1) / DEL_T)**2, axis=1) * 100
        total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return {
            'lataccel_cost': lat_accel_cost,
            'jerk_cost': jerk_cost,
            'total_cost': total_cost,
        }
