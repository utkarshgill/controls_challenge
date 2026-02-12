"""Benchmark actual wall-clock for different worker x chunk-size combos."""
import time
import multiprocessing as mp
import numpy as np
import onnxruntime as ort

MODEL = 'models/tinyphysics.onnx'
CONTEXT_LENGTH = 20
VOCAB_SIZE = 256
STEPS = 200  # enough to get stable timings

_sess = None
def init():
    global _sess
    _sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])

def worker(N):
    """Simulate N episodes for STEPS sequential ONNX calls."""
    states = np.random.randn(N, CONTEXT_LENGTH, 4).astype(np.float32)
    tokens = np.random.randint(0, VOCAB_SIZE, (N, CONTEXT_LENGTH)).astype(np.int64)
    for _ in range(STEPS):
        _sess.run(None, {'states': states, 'tokens': tokens})
    return N

TOTAL = 1000

if __name__ == '__main__':
  mp.set_start_method('fork')
  for n_workers in [1, 2, 4, 6, 8, 10, 12]:
    chunk = TOTAL // n_workers
    chunks = [chunk] * n_workers
    for i in range(TOTAL - chunk * n_workers):
        chunks[i] += 1

    pool = mp.Pool(n_workers, initializer=init)
    pool.map(worker, chunks)  # warmup

    t0 = time.perf_counter()
    pool.map(worker, chunks)
    elapsed = time.perf_counter() - t0

    projected = elapsed / STEPS * 580
    pool.terminate()
    pool.join()
    print(f"workers={n_workers:>2}  chunk~{chunk:>4}  "
          f"{STEPS}steps={elapsed:.2f}s  projected_580steps={projected:.1f}s")
