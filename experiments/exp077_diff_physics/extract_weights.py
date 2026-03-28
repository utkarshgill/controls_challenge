"""Extract weights from tinyphysics.onnx into a PyTorch state_dict .pt file."""

import onnx
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
model = onnx.load(str(ROOT / "models" / "tinyphysics.onnx"))

# Build mapping from ONNX initializer names to numpy arrays
onnx_weights = {}
for init in model.graph.initializer:
    if init.data_type == 1:  # float32
        data = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
    elif init.data_type == 9:  # bool
        data = np.frombuffer(init.raw_data, dtype=np.bool_).reshape(init.dims)
    elif init.data_type == 7:  # int64
        data = np.frombuffer(init.raw_data, dtype=np.int64).reshape(init.dims)
    else:
        print(f"Skipping {init.name} with data_type={init.data_type}")
        continue
    onnx_weights[init.name] = data

# Map ONNX names to PyTorch state_dict keys
# Architecture (from tracing):
#   states (b,20,4) -> wt_embedding: Linear(4,64, bias=True) -> (b,20,64)
#   tokens (b,20)   -> wt2_embedding: Embedding(1024,64) -> (b,20,64)
#   concat -> (b,20,128) + wp_embedding(20,128) -> x (b,20,128)
#   4x TransformerBlock:
#     LayerNorm -> c_attn: Linear(128,384, bias=False) -> split(Q128,K128,V128)
#     Multi-head attention (4 heads, 32 per head), causal mask
#     c_proj: Linear(128,128, bias=False)
#     residual add
#     LayerNorm -> MLP: Linear(128,512,bias=False) -> Tanh -> Linear(512,128,bias=False)
#     residual add
#   LayerNorm_f -> lm_head: Linear(128,1024, bias=False)

# ONNX MatMul nodes (in order of appearance in weight list):
# onnx::MatMul_540: [4, 64]   -> wt_embedding.weight (transposed: ONNX uses x @ W, PyTorch uses x @ W.T so Linear.weight = W.T)
# onnx::MatMul_542: [128, 384] -> h.0.attn.c_attn.weight
# onnx::MatMul_546: [128, 128] -> h.0.attn.c_proj.weight
# onnx::MatMul_547: [128, 512] -> h.0.mlp.c_fc.weight
# onnx::MatMul_548: [512, 128] -> h.0.mlp.c_proj.weight
# ... repeat for h.1, h.2, h.3
# onnx::MatMul_570: [128, 1024] -> lm_head.weight

matmul_keys = [
    ("onnx::MatMul_540", "wt_embedding.weight"),  # [4, 64] -> Linear weight is [64, 4]
    ("onnx::MatMul_542", "h.0.attn.c_attn.weight"),  # [128, 384]
    ("onnx::MatMul_546", "h.0.attn.c_proj.weight"),  # [128, 128]
    ("onnx::MatMul_547", "h.0.mlp.c_fc.weight"),  # [128, 512]
    ("onnx::MatMul_548", "h.0.mlp.c_proj.weight"),  # [512, 128]
    ("onnx::MatMul_549", "h.1.attn.c_attn.weight"),
    ("onnx::MatMul_553", "h.1.attn.c_proj.weight"),
    ("onnx::MatMul_554", "h.1.mlp.c_fc.weight"),
    ("onnx::MatMul_555", "h.1.mlp.c_proj.weight"),
    ("onnx::MatMul_556", "h.2.attn.c_attn.weight"),
    ("onnx::MatMul_560", "h.2.attn.c_proj.weight"),
    ("onnx::MatMul_561", "h.2.mlp.c_fc.weight"),
    ("onnx::MatMul_562", "h.2.mlp.c_proj.weight"),
    ("onnx::MatMul_563", "h.3.attn.c_attn.weight"),
    ("onnx::MatMul_567", "h.3.attn.c_proj.weight"),
    ("onnx::MatMul_568", "h.3.mlp.c_fc.weight"),
    ("onnx::MatMul_569", "h.3.mlp.c_proj.weight"),
    ("onnx::MatMul_570", "lm_head.weight"),  # [128, 1024]
]

state_dict = {}

# MatMul weights: ONNX does x @ W, PyTorch Linear does x @ W.T
# So PyTorch Linear.weight = W.T
for onnx_name, pt_name in matmul_keys:
    w = onnx_weights[onnx_name]
    state_dict[pt_name] = torch.from_numpy(w.T.copy())  # transpose for Linear

# Bias for wt_embedding
state_dict["wt_embedding.bias"] = torch.from_numpy(
    onnx_weights["transformer.wt_embedding.bias"].copy()
)

# Embeddings
state_dict["wt2_embedding.weight"] = torch.from_numpy(
    onnx_weights["transformer.wt2_embedding.weight"].copy()
)
state_dict["wp_embedding.weight"] = torch.from_numpy(
    onnx_weights["transformer.wp_embedding.weight"].copy()
)

# Layer norms for each block
for i in range(4):
    for sub in ("attn", "mlp"):
        for param in ("weight", "bias"):
            onnx_key = f"transformer.h.{i}.{sub}.layer_norm.{param}"
            pt_key = f"h.{i}.{sub}.layer_norm.{param}"
            state_dict[pt_key] = torch.from_numpy(onnx_weights[onnx_key].copy())

# Final layer norm
state_dict["layer_norm_f.weight"] = torch.from_numpy(
    onnx_weights["transformer.layer_norm_f.weight"].copy()
)
state_dict["layer_norm_f.bias"] = torch.from_numpy(
    onnx_weights["transformer.layer_norm_f.bias"].copy()
)

# Causal mask (bool)
causal = onnx_weights["transformer.h.0.attn.bias"]  # (1,1,20,20) bool
state_dict["causal_mask"] = torch.from_numpy(causal.astype(np.bool_))

out_path = Path(__file__).parent / "tinyphysics_torch_weights.pt"
torch.save(state_dict, out_path)
print(f"Saved {len(state_dict)} tensors to {out_path}")
for k, v in sorted(state_dict.items()):
    print(f"  {k}: {list(v.shape)}")
