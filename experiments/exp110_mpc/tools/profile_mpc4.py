"""Deep profile: what's inside the 7ms TRT call?
Test with different batch sizes and model components."""

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

# Test 1: Scale NK while keeping total elements constant by adjusting CL
# This isolates batch vs sequence dimension
ort_sess = make_ort_session(mdl_path)
out_name = ort_sess.get_outputs()[0].name

print("=== TRT call time vs NK ===")
for NK in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
    try:
        states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
        tokens = torch.randint(
            0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda"
        )
        out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")

        io = ort_sess.io_binding()
        io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
        io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
        io.bind_output(
            out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
        )

        # Warm up
        for _ in range(3):
            ort_sess.run_with_iobinding(io)
        torch.cuda.synchronize()

        N_ITERS = 20
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(N_ITERS):
            ort_sess.run_with_iobinding(io)
        end_event.record()
        torch.cuda.synchronize()

        gpu_ms = start_event.elapsed_time(end_event) / N_ITERS
        throughput = NK / (gpu_ms / 1000)
        out_size_mb = NK * CL * VOCAB_SIZE * 4 / 1e6
        print(
            f"  NK={NK:>7d}  GPU={gpu_ms:8.2f}ms  throughput={throughput / 1e6:.2f}M/s  out_tensor={out_size_mb:.0f}MB"
        )
    except Exception as e:
        print(f"  NK={NK:>7d}  FAILED: {e}")

# Test 2: pure CUDA EP for comparison
print("\n=== CUDA EP (no TRT) vs NK ===")
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.log_severity_level = 3
with open(mdl_path, "rb") as f:
    sess_cuda = ort.InferenceSession(
        f.read(), options, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

for NK in [10, 100, 1000, 10000, 50000]:
    try:
        states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
        tokens = torch.randint(
            0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda"
        )
        out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")

        io = sess_cuda.io_binding()
        io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
        io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
        io.bind_output(
            out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
        )

        for _ in range(3):
            sess_cuda.run_with_iobinding(io)
        torch.cuda.synchronize()

        N_ITERS = 10
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(N_ITERS):
            sess_cuda.run_with_iobinding(io)
        end_event.record()
        torch.cuda.synchronize()

        gpu_ms = start_event.elapsed_time(end_event) / N_ITERS
        throughput = NK / (gpu_ms / 1000)
        print(
            f"  NK={NK:>7d}  GPU={gpu_ms:8.2f}ms  throughput={throughput / 1e6:.2f}M/s"
        )
    except Exception as e:
        print(f"  NK={NK:>7d}  FAILED: {e}")

# Test 3: How much is the output tensor? The lm_head produces (NK, 20, 1024)
# What if we only need the last timestep? Can we reduce by 20x?
print("\n=== Memory bandwidth estimate ===")
for NK in [6400, 12800, 64000]:
    out_bytes = NK * CL * VOCAB_SIZE * 4  # float32
    bw_tb = 2.0  # H100 HBM bandwidth TB/s
    min_time_ms = out_bytes / (bw_tb * 1e12) * 1000
    print(
        f"  NK={NK:>7d}  output={out_bytes / 1e6:.0f}MB  min_time@2TB/s={min_time_ms:.2f}ms"
    )
