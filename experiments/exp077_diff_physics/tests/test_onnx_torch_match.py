"""Test: exact match between ONNX Runtime CPU and onnx2torch PyTorch model.
Find and fix any discrepancies."""

import os, sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import onnxruntime as ort

ONNX_PATH = "models/tinyphysics.onnx"

# ===== ONNX Runtime CPU =====
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# Fixed inputs
np.random.seed(42)
states_np = np.random.randn(1, 20, 4).astype(np.float32)
tokens_np = np.random.randint(0, 1024, (1, 20)).astype(np.int64)

ort_out = sess.run(None, {"states": states_np, "tokens": tokens_np})[0]
print(f"ORT output shape: {ort_out.shape}")
print(f"ORT logits[0,-1,:5]: {ort_out[0, -1, :5]}")
print(f"ORT argmax: {ort_out[0, -1].argmax()}")

# ===== onnx2torch =====
try:
    import onnx2torch

    torch_model = onnx2torch.convert(ONNX_PATH)
    torch_model.eval()

    states_t = torch.from_numpy(states_np)
    tokens_t = torch.from_numpy(tokens_np)

    with torch.no_grad():
        torch_out = torch_model(states_t, tokens_t).numpy()

    print(f"\nonnx2torch output shape: {torch_out.shape}")
    print(f"onnx2torch logits[0,-1,:5]: {torch_out[0, -1, :5]}")
    print(f"onnx2torch argmax: {torch_out[0, -1].argmax()}")

    diff = np.abs(ort_out - torch_out)
    print(f"\nMax diff: {diff.max():.10f}")
    print(f"Mean diff: {diff.mean():.10f}")
    print(
        f"Diff at last position: max={diff[0, -1].max():.10f} mean={diff[0, -1].mean():.10f}"
    )

    # Check if the diff causes different token selection
    from tinyphysics import LATACCEL_RANGE, VOCAB_SIZE

    bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)

    def softmax(x, temp=0.8):
        x = x / temp
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    probs_ort = softmax(ort_out[0, -1])
    probs_torch = softmax(torch_out[0, -1])

    # For 100 random u values, check if token selection differs
    np.random.seed(0)
    mismatches = 0
    for _ in range(1000):
        u = np.random.rand()
        cdf_ort = np.cumsum(probs_ort)
        cdf_torch = np.cumsum(probs_torch)
        tok_ort = np.searchsorted(cdf_ort, u)
        tok_torch = np.searchsorted(cdf_torch, u)
        if tok_ort != tok_torch:
            mismatches += 1

    print(f"\nToken selection mismatches: {mismatches}/1000")

    # Expected value comparison
    ev_ort = (probs_ort * bins).sum()
    ev_torch = (probs_torch * bins).sum()
    print(
        f"Expected value: ORT={ev_ort:.8f} torch={ev_torch:.8f} diff={abs(ev_ort - ev_torch):.10f}"
    )

except ImportError:
    print("onnx2torch not installed. pip install onnx2torch")

# ===== Hand-written DifferentiablePhysics =====
try:
    sys.path.insert(0, "experiments/exp077_diff_physics")
    from diff_physics import DifferentiablePhysics

    weights_path = "experiments/exp077_diff_physics/tinyphysics_torch_weights.pt"
    if Path(weights_path).exists():
        dp = DifferentiablePhysics()
        dp.load_state_dict(torch.load(weights_path, weights_only=True))
        dp.eval()

        with torch.no_grad():
            dp_logits = dp.forward_logits(states_t, tokens_t).numpy()

        diff_dp = np.abs(ort_out - dp_logits)
        print(f"\nDifferentiablePhysics:")
        print(f"  logits[0,-1,:5]: {dp_logits[0, -1, :5]}")
        print(f"  Max diff from ORT: {diff_dp.max():.10f}")
        print(f"  Mean diff from ORT: {diff_dp.mean():.10f}")
    else:
        print(f"\nWeights not found at {weights_path}")
        print("Run: python experiments/exp077_diff_physics/extract_weights.py")
except Exception as e:
    print(f"\nDifferentiablePhysics error: {e}")
