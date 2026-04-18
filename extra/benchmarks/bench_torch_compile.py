"""Benchmark: ORT CUDA EP fp32 vs onnx2torch on GPU."""

import os, sys, time, numpy as np, torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import onnxruntime as ort
import onnx2torch
from tinyphysics import LATACCEL_RANGE, VOCAB_SIZE, CONTEXT_LENGTH

CL = CONTEXT_LENGTH

# ORT CUDA EP fp32 no TF32
options = ort.SessionOptions()
options.log_severity_level = 3
ort_sess = ort.InferenceSession(
    "models/tinyphysics.onnx",
    options,
    [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"],
)
out_name = ort_sess.get_outputs()[0].name

# onnx2torch on GPU
torch_model = onnx2torch.convert("models/tinyphysics.onnx").cuda().eval()
# Disable TF32 for PyTorch too
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

for NK in [32, 320, 3200, 6400]:
    states_t = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
    tokens_t = torch.randint(0, 1024, (NK, CL), dtype=torch.int64, device="cuda")
    states_np = states_t.cpu().numpy()
    tokens_np = tokens_t.cpu().numpy()

    out_gpu = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")
    io = ort_sess.io_binding()
    io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states_t.data_ptr())
    io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens_t.data_ptr())
    io.bind_output(
        out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_gpu.data_ptr()
    )

    # Warmup
    for _ in range(5):
        ort_sess.run_with_iobinding(io)
    for _ in range(5):
        with torch.no_grad():
            torch_model(states_t, tokens_t)
    torch.cuda.synchronize()

    N = 20

    # ORT
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        ort_sess.run_with_iobinding(io)
    torch.cuda.synchronize()
    t_ort = (time.perf_counter() - t0) / N

    # Torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            torch_model(states_t, tokens_t)
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / N

    speedup = t_ort / t_torch
    print(
        f"NK={NK:>5d}  ORT={t_ort * 1000:7.2f}ms  torch={t_torch * 1000:7.2f}ms  speedup={speedup:.2f}x"
    )
