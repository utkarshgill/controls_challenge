"""Profile model call at different batch sizes + check if TRT is actually used."""

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

# Check what providers are ACTUALLY executing
ort_sess = make_ort_session(mdl_path)
print(f"Providers: {ort_sess.get_providers()}")

# Check per-node provider assignment
# This tells us if TRT actually compiled the graph or fell back to CUDA
out_name = ort_sess.get_outputs()[0].name

for NK in [64, 640, 6400, 12800, 32000, 64000]:
    states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
    tokens = torch.randint(0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda")

    io = ort_sess.io_binding()
    io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
    io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
    out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")
    io.bind_output(
        out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
    )

    # Warm up
    for _ in range(3):
        ort_sess.run_with_iobinding(io)
    torch.cuda.synchronize()

    N_ITERS = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        ort_sess.run_with_iobinding(io)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / N_ITERS

    throughput = NK / t
    print(
        f"  NK={NK:>6d}  {t * 1000:7.2f} ms  throughput={throughput / 1e6:.2f}M samples/s  per_sample={t / NK * 1e6:.2f} µs"
    )

# Also test WITHOUT TRT (pure CUDA EP)
print("\n--- Without TRT (pure CUDA EP) ---")
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.log_severity_level = 3
with open(mdl_path, "rb") as f:
    sess_cuda = ort.InferenceSession(
        f.read(), options, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
print(f"Providers: {sess_cuda.get_providers()}")

for NK in [64, 640, 6400, 12800, 32000, 64000]:
    states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
    tokens = torch.randint(0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda")

    io = sess_cuda.io_binding()
    io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
    io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
    out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")
    io.bind_output(
        out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
    )

    for _ in range(3):
        sess_cuda.run_with_iobinding(io)
    torch.cuda.synchronize()

    N_ITERS = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        sess_cuda.run_with_iobinding(io)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / N_ITERS

    throughput = NK / t
    print(
        f"  NK={NK:>6d}  {t * 1000:7.2f} ms  throughput={throughput / 1e6:.2f}M samples/s  per_sample={t / NK * 1e6:.2f} µs"
    )
