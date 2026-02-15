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
    with open(str(model_path), 'rb') as f:
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if os.getenv('CUDA', '0') == '1' else ['CPUExecutionProvider'])
        return ort.InferenceSession(f.read(), options, providers)


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

def preload_csvs(csv_files):
    """Load N CSVs into (N, T) numpy arrays.

    Returns dict of float64 arrays (matching pandas native precision used by
    the original TinyPhysicsSimulator).
    """
    dfs = [pd.read_csv(str(f)) for f in csv_files]
    N = len(dfs)
    T = max(len(df) for df in dfs)
    roll_la = np.empty((N, T), np.float64)
    v_ego   = np.empty((N, T), np.float64)
    a_ego   = np.empty((N, T), np.float64)
    tgt_la  = np.empty((N, T), np.float64)
    steer   = np.empty((N, T), np.float64)
    for i, df in enumerate(dfs):
        L = len(df)
        roll_la[i, :L] = np.sin(df['roll'].values) * ACC_G
        v_ego[i, :L]   = df['vEgo'].values
        a_ego[i, :L]   = df['aEgo'].values
        tgt_la[i, :L]  = df['targetLateralAcceleration'].values
        steer[i, :L]   = -df['steerCommand'].values
        # Edge-pad short CSVs (repeat last row)
        if L < T:
            roll_la[i, L:] = roll_la[i, L - 1]
            v_ego[i, L:]   = v_ego[i, L - 1]
            a_ego[i, L:]   = a_ego[i, L - 1]
            tgt_la[i, L:]  = tgt_la[i, L - 1]
            steer[i, L:]   = steer[i, L - 1]
    return dict(roll_lataccel=roll_la, v_ego=v_ego, a_ego=a_ego,
                target_lataccel=tgt_la, steer_command=steer, N=N, T=T)


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

    def softmax(self, x, axis=-1):
        """Mirrors TinyPhysicsModel.softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=0.8,
                rngs: list = None) -> np.ndarray:
        """Batched predict.  Mirrors TinyPhysicsModel.predict.

        rngs: optional list of N np.random.RandomState objects (per-episode).
              When provided, uses np.random.choice per row to match the
              reference TinyPhysicsModel.predict exactly.
        Returns: (N,) int64 token indices.
        """
        res = self.ort_session.run(None, input_data)[0]  # (N, CL, VOCAB_SIZE)
        probs = self.softmax(res / temperature, axis=-1)
        probs = probs[:, -1, :]                           # (N, VOCAB_SIZE)
        self._last_probs = probs                           # cache for expected-value
        N = probs.shape[0]

        if rngs is not None:
            # Per-episode seeded sampling — matches reference exactly
            samples = np.empty(N, dtype=np.intp)
            for i in range(N):
                samples[i] = rngs[i].choice(probs.shape[1], p=probs[i])
        else:
            # Vectorized sampling (fast, non-deterministic vs reference)
            cumprobs = np.cumsum(probs, axis=1)
            u = np.random.rand(N, 1)
            samples = (cumprobs < u).sum(axis=1)
            samples = np.clip(samples, 0, VOCAB_SIZE - 1)
        return samples

    def get_current_lataccel(self, sim_states: np.ndarray,
                             actions: np.ndarray,
                             past_preds: np.ndarray,
                             rngs: list = None,
                             return_expected: bool = False):
        """Batched get_current_lataccel.  Mirrors TinyPhysicsModel method.

        sim_states: (N, CL, 3)  float64 — [roll_lataccel, v_ego, a_ego]
        actions:    (N, CL)     float64
        past_preds: (N, CL)     float64
        rngs:       optional list of N RandomState for per-episode seeding
        return_expected: if True, also return E[lataccel] = sum(probs * bins)
        Returns:    (N,) float64, or tuple ((N,), (N,)) if return_expected.
        """
        tokenized_actions = self.tokenizer.encode(past_preds)  # (N, CL)
        states = np.concatenate(
            [actions[:, :, None], sim_states], axis=-1)
        input_data = {
            'states': states.astype(np.float32),
            'tokens': tokenized_actions.astype(np.int64),
        }
        sampled = self.tokenizer.decode(self.predict(input_data, temperature=0.8,
                                                     rngs=rngs))
        if not return_expected:
            return sampled
        # Expected value from the SAME forward pass (reuse cached probs)
        expected = np.sum(self._last_probs * self.tokenizer.bins[None, :], axis=-1)
        return sampled, expected


# ── Batched simulator (mirrors TinyPhysicsSimulator) ──────────

class BatchedSimulator:
    """Vectorized drop-in for TinyPhysicsSimulator.

    Mirrors the original's reset / step / sim_step / control_step / rollout /
    compute_cost structure but operates on (N, ...) arrays.
    """

    def __init__(self, model_path: str, csv_files: list,
                 ort_session=None) -> None:
        self.sim_model = BatchedPhysicsModel(model_path, ort_session=ort_session)
        self.csv_files = csv_files
        self.data = preload_csvs(csv_files)
        self.N = self.data['N']
        self.T = self.data['T']
        self.compute_expected = False   # set True to piggyback E[lataccel]
        self.expected_lataccel = None   # (N,) after sim_step if compute_expected
        self.reset()

    # ── reset  (mirrors tinyphysics.py lines 110-120) ────────

    def reset(self) -> None:
        N, T = self.N, self.T
        CL = CONTEXT_LENGTH

        # Per-episode RNGs seeded identically to TinyPhysicsModel.__init__
        # which does: np.random.seed(int(md5(data_path.encode()).hexdigest(), 16) % 10**10)
        self.rngs = []
        for f in self.csv_files:
            seed = int(md5(('data/' + Path(f).name).encode()).hexdigest(), 16) % 10**4
            self.rngs.append(np.random.RandomState(seed))

        # action_history — float64 (matches original's Python-float list)
        # tinyphysics: self.action_history = self.data['steer_command'][:step_idx]
        self.action_history = self.data['steer_command'][:, :CL].copy()  # (N, CL)

        # state_history — (N, CL, 3) [roll_lataccel, v_ego, a_ego]
        # tinyphysics: self.state_history = [State(...) for i in range(step_idx)]
        self.state_history = np.stack([
            self.data['roll_lataccel'][:, :CL],
            self.data['v_ego'][:, :CL],
            self.data['a_ego'][:, :CL],
        ], axis=-1)  # (N, CL, 3)

        # current_lataccel_history — float64
        # tinyphysics: self.current_lataccel_history = [target_la for ...]
        self.current_lataccel_history = self.data['target_lataccel'][:, :CL].copy()

        # tinyphysics: self.current_lataccel = self.current_lataccel_history[-1]
        self.current_lataccel = self.current_lataccel_history[:, -1].copy()  # (N,)

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
        result = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[:, -CL:, :],
            actions=self.action_history[:, -CL:],
            past_preds=self.current_lataccel_history[:, -CL:],
            rngs=self.rngs,
            return_expected=self.compute_expected,
        )
        if self.compute_expected:
            pred, self.expected_lataccel = result
        else:
            pred = result

        pred = np.clip(pred,
                       self.current_lataccel - MAX_ACC_DELTA,
                       self.current_lataccel + MAX_ACC_DELTA)

        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.data['target_lataccel'][:, step_idx].copy()

        self.current_lataccel_history = np.concatenate(
            [self.current_lataccel_history, self.current_lataccel[:, None]],
            axis=1)

    # ── control_step  (mirrors lines 147-152) ────────────────

    def control_step(self, step_idx: int, actions: np.ndarray) -> None:
        """Accept externally-provided actions (N,), clip, append to history.

        Mirrors:
          if step < CONTROL_START: action = CSV steer
          action = clip(action, STEER_RANGE)
          action_history.append(action)
        """
        if step_idx < CONTROL_START_IDX:
            actions = self.data['steer_command'][:, step_idx].copy()
        actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history = np.concatenate(
            [self.action_history, actions[:, None]], axis=1)

    # ── step  (mirrors lines 167-174) ────────────────────────

    def step(self, step_idx: int, actions: np.ndarray) -> dict:
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

        # Append state to history (mirrors self.state_history.append(state))
        new_state = np.stack([roll_la, v_ego, a_ego], axis=-1)[:, None, :]
        self.state_history = np.concatenate(
            [self.state_history, new_state], axis=1)

        self.control_step(step_idx, actions)
        self.sim_step(step_idx)

        return dict(
            roll_lataccel=roll_la, v_ego=v_ego, a_ego=a_ego,
            target=target, future_plan=future_plan,
            current_lataccel=self.current_lataccel.copy(),
        )

    # ── rollout  (mirrors lines 195-213) ─────────────────────

    def rollout(self, controller_fn: Callable) -> Dict[str, np.ndarray]:
        """Run full rollout.

        controller_fn(step_idx, target, current_lataccel, state_dict, future_plan)
            → actions (N,)

        Returns dict with 'total_cost', 'lataccel_cost', 'jerk_cost' as (N,) arrays,
        plus 'controller_info' = list of dicts returned by controller_fn (if any).
        """
        for step_idx in range(CONTEXT_LENGTH, self.T):
            # Peek at this step's data for the controller (same as step() reads)
            roll_la, v_ego, a_ego, target, future_plan = \
                self.get_state_target_futureplan(step_idx)

            state_dict = dict(roll_lataccel=roll_la, v_ego=v_ego, a_ego=a_ego)

            actions = controller_fn(
                step_idx, target, self.current_lataccel.copy(),
                state_dict, future_plan)

            self.step(step_idx, actions)

        return self.compute_cost()

    # ── compute_cost  (mirrors lines 186-193) ────────────────

    def compute_cost(self) -> Dict[str, np.ndarray]:
        """Vectorized cost computation.  Returns dict with (N,) arrays."""
        target = self.data['target_lataccel'][:, CONTROL_START_IDX:COST_END_IDX]  # bounds-safe slice, like ref
        pred = self.current_lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = np.mean((target - pred)**2, axis=1) * 100
        jerk_cost = np.mean((np.diff(pred, axis=1) / DEL_T)**2, axis=1) * 100
        total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return {
            'lataccel_cost': lat_accel_cost,
            'jerk_cost': jerk_cost,
            'total_cost': total_cost,
        }
