"""Profile: what fraction of the 7ms model call is actual GPU compute vs overhead?
Uses CUDA events for precise GPU timing."""

import os, sys, time, torch
from pathlib import Path
import numpy as np

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import LATACCEL_RANGE, VOCAB_SIZE, CONTEXT_LENGTH
from tinyphysics_batched import make_ort_session
import onnxruntime as ort

DEV = torch.device("cuda")
CL = CONTEXT_LENGTH

mdl_path = "models/tinyphysics.onnx"
ort_sess = make_ort_session(mdl_path)
out_name = ort_sess.get_outputs()[0].name

for NK in [640, 6400, 12800]:
    states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
    tokens = torch.randint(0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda")
    out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")

    io = ort_sess.io_binding()
    io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
    io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
    io.bind_output(
        out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
    )

    # Warm up
    for _ in range(5):
        ort_sess.run_with_iobinding(io)
    torch.cuda.synchronize()

    # GPU timing with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    N_ITERS = 50
    gpu_times = []
    wall_times = []

    for _ in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_event.record()
        ort_sess.run_with_iobinding(io)
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_times.append(start_event.elapsed_time(end_event))  # ms
        wall_times.append((t1 - t0) * 1000)  # ms

    gpu_mean = np.mean(gpu_times)
    wall_mean = np.mean(wall_times)
    overhead = wall_mean - gpu_mean

    print(
        f"NK={NK:>6d}  GPU={gpu_mean:.2f}ms  Wall={wall_mean:.2f}ms  "
        f"Overhead={overhead:.2f}ms ({overhead / wall_mean * 100:.0f}%)"
    )

    # Also measure: 10 back-to-back calls WITHOUT sync between them
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start_event.record()
    for _ in range(10):
        ort_sess.run_with_iobinding(io)
    end_event.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_10 = start_event.elapsed_time(end_event)
    wall_10 = (t1 - t0) * 1000
    print(
        f"         10x back-to-back: GPU={gpu_10:.1f}ms  Wall={wall_10:.1f}ms  "
        f"per_call: GPU={gpu_10 / 10:.2f}ms  Wall={wall_10 / 10:.2f}ms"
    )
