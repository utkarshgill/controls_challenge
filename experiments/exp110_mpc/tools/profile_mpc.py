"""Profile the MPC inner loop bottleneck.
Run on GPU: python profile_mpc.py
"""

import os, sys, time, torch
from pathlib import Path
import numpy as np

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import LATACCEL_RANGE, VOCAB_SIZE, CONTEXT_LENGTH
from tinyphysics_batched import BatchedPhysicsModel, make_ort_session

DEV = torch.device("cuda")
BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=DEV)
CL = CONTEXT_LENGTH
NK = 6400  # 100 routes × 64 candidates

mdl_path = "models/tinyphysics.onnx"
ort_sess = make_ort_session(mdl_path)
phys = BatchedPhysicsModel(mdl_path, ort_session=ort_sess)

# Create dummy data
states = torch.randn(NK, CL, 4, dtype=torch.float32, device="cuda")
tokens = torch.randint(0, VOCAB_SIZE, (NK, CL), dtype=torch.int64, device="cuda")
rng_u = torch.rand(NK, dtype=torch.float64, device="cuda")

# Warm up TRT
for _ in range(3):
    phys._predict_gpu(
        {"states": states, "tokens": tokens}, temperature=0.8, rng_u=rng_u
    )
torch.cuda.synchronize()

# Profile individual components
N_ITERS = 50

# 1. Full predict call
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    phys._predict_gpu(
        {"states": states, "tokens": tokens}, temperature=0.8, rng_u=rng_u
    )
torch.cuda.synchronize()
t_full = (time.perf_counter() - t0) / N_ITERS
print(f"Full _predict_gpu:  {t_full * 1000:.2f} ms")

# 2. Just the ORT run_with_iobinding
io = phys._io
io.clear_binding_inputs()
io.clear_binding_outputs()
io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
out_name = ort_sess.get_outputs()[0].name
out_buf = torch.empty(NK, CL, VOCAB_SIZE, dtype=torch.float32, device="cuda")
io.bind_output(
    out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    ort_sess.run_with_iobinding(io)
torch.cuda.synchronize()
t_ort = (time.perf_counter() - t0) / N_ITERS
print(f"ORT run_with_iobinding only:  {t_ort * 1000:.2f} ms")

# 3. Just the binding setup
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    io.clear_binding_inputs()
    io.clear_binding_outputs()
    io.bind_input("states", "cuda", 0, np.float32, [NK, CL, 4], states.data_ptr())
    io.bind_input("tokens", "cuda", 0, np.int64, [NK, CL], tokens.data_ptr())
    io.bind_output(
        out_name, "cuda", 0, np.float32, [NK, CL, VOCAB_SIZE], out_buf.data_ptr()
    )
torch.cuda.synchronize()
t_bind = (time.perf_counter() - t0) / N_ITERS
print(f"IOBinding setup only:  {t_bind * 1000:.2f} ms")

# 4. Just softmax + sampling
logits = out_buf[:, -1, :]
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    probs = torch.softmax(logits / 0.8, dim=-1)
    cdf = torch.cumsum(probs, dim=1)
    cdf = cdf / cdf[:, -1:]
    u = rng_u.unsqueeze(1)
    samples = (
        torch.searchsorted(cdf.double(), u.double()).squeeze(1).clamp(0, VOCAB_SIZE - 1)
    )
torch.cuda.synchronize()
t_sample = (time.perf_counter() - t0) / N_ITERS
print(f"Softmax + CDF + searchsorted:  {t_sample * 1000:.2f} ms")

# 5. Tensor manipulation (cat, bucketize, etc.)
act_hk = torch.randn(NK, CL - 1, dtype=torch.float32, device="cuda")
cur_steer = torch.randn(NK, dtype=torch.float32, device="cuda")
st_hk = torch.randn(NK, CL, 3, dtype=torch.float32, device="cuda")
new_st = torch.randn(NK, 3, dtype=torch.float32, device="cuda")
pr_hk = torch.randn(NK, CL, dtype=torch.float32, device="cuda")
pred_la = torch.randn(NK, dtype=torch.float32, device="cuda")

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    a_ctx = torch.cat([act_hk, cur_steer.unsqueeze(1)], dim=1)
    s_ctx = torch.cat([st_hk[:, 1:], new_st.unsqueeze(1)], dim=1)
    fs = torch.empty(NK, CL, 4, dtype=torch.float32, device="cuda")
    fs[:, :, 0] = a_ctx
    fs[:, :, 1:] = s_ctx
    tk = (
        torch.bucketize(pr_hk.clamp(-5, 5), BINS, right=False)
        .clamp(0, VOCAB_SIZE - 1)
        .long()
    )
    new_pr = torch.cat([pr_hk[:, 1:], pred_la.unsqueeze(1)], dim=1)
    new_act = a_ctx[:, 1:]
torch.cuda.synchronize()
t_manip = (time.perf_counter() - t0) / N_ITERS
print(f"Tensor manipulation (cat/bucketize):  {t_manip * 1000:.2f} ms")

# 6. Pre-bound predict (no rebinding)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    # Just copy + run + softmax+sample, no rebinding
    ort_sess.run_with_iobinding(io)
    probs = torch.softmax(out_buf[:, -1, :] / 0.8, dim=-1)
    cdf = torch.cumsum(probs, dim=1)
    cdf = cdf / cdf[:, -1:]
    samples = torch.searchsorted(cdf.double(), rng_u.unsqueeze(1).double()).squeeze(1)
torch.cuda.synchronize()
t_prebound = (time.perf_counter() - t0) / N_ITERS
print(f"Pre-bound predict + sample:  {t_prebound * 1000:.2f} ms")

print(f"\n--- Summary for NK={NK} ---")
print(
    f"  ORT model call:     {t_ort * 1000:7.2f} ms  ({t_ort / t_full * 100:.0f}% of full)"
)
print(f"  IOBinding setup:    {t_bind * 1000:7.2f} ms  ({t_bind / t_full * 100:.0f}%)")
print(
    f"  Softmax+sampling:   {t_sample * 1000:7.2f} ms  ({t_sample / t_full * 100:.0f}%)"
)
print(f"  Tensor manipulation:{t_manip * 1000:7.2f} ms")
print(f"  Full predict:       {t_full * 1000:7.2f} ms")
print(f"  Pre-bound predict:  {t_prebound * 1000:7.2f} ms")
print(
    f"\n  Per control step (5 CEM × 50 H): {5 * 50 * t_full * 1000:.0f} ms = {5 * 50 * t_full:.2f} s"
)
print(
    f"  Per control step (pre-bound):     {5 * 50 * t_prebound * 1000:.0f} ms = {5 * 50 * t_prebound:.2f} s"
)
print(f"  400 steps total:                  {400 * 5 * 50 * t_full:.0f} s")
