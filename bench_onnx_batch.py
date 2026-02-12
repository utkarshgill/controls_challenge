"""Benchmark ONNX inference time vs batch size."""
import time
import numpy as np
import onnxruntime as ort

MODEL = 'models/tinyphysics.onnx'
CONTEXT_LENGTH = 20
VOCAB_SIZE = 256
STEPS = 100  # simulate 100 sequential steps

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])

for N in [1, 10, 25, 50, 100, 150, 200, 500, 1000]:
    states = np.random.randn(N, CONTEXT_LENGTH, 4).astype(np.float32)
    tokens = np.random.randint(0, VOCAB_SIZE, (N, CONTEXT_LENGTH)).astype(np.int64)

    # Warmup
    for _ in range(5):
        sess.run(None, {'states': states, 'tokens': tokens})

    t0 = time.perf_counter()
    for _ in range(STEPS):
        sess.run(None, {'states': states, 'tokens': tokens})
    elapsed = time.perf_counter() - t0

    per_call_ms = elapsed / STEPS * 1000
    per_episode_us = elapsed / STEPS / N * 1e6
    projected_1000ep = per_call_ms * 580 / 1000  # 580 steps, full rollout

    print(f"N={N:>4}  call={per_call_ms:6.2f}ms  per_ep={per_episode_us:6.1f}µs  "
          f"projected_1000ep_1worker={projected_1000ep/N:.1f}s  "
          f"projected_10workers={projected_1000ep/N/10:.1f}s")
