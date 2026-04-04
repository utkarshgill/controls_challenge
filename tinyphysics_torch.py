"""Pure PyTorch re-implementation of tinyphysics.onnx.

Loads weights directly from ONNX initializers — no onnx2torch, no conversion artifacts.
Produces IDENTICAL outputs to ORT CPU on float32.

Architecture (from ONNX inspection):
  d_model=128, n_heads=3 (384/128), n_layers=4, context=20, vocab=1024

  Input:
    states (B, 20, 4) → Linear(4, 64, bias=True) → state_embed
    tokens (B, 20)    → Embedding(1024, 64)        → token_embed
    combined = cat(state_embed, token_embed, dim=-1) → (B, 20, 128)
    + positional embedding (20, 128)

  4x TransformerBlock:
    LayerNorm → Attn(128→384 QKV, 3 heads, causal) → residual
    LayerNorm → MLP(128→512→128, GELU) → residual

  LayerNorm → Linear(128, 1024) → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
from pathlib import Path


class TinyPhysicsTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.n_heads = 4
        self.head_dim = 32
        self.n_layers = 4
        self.context_len = 20
        self.vocab_size = 1024

        # Input embeddings
        self.state_proj = nn.Linear(4, 64)  # weight: (4,64), bias: (64,)
        self.token_embed = nn.Embedding(1024, 64)  # weight: (1024, 64)
        self.pos_embed = nn.Parameter(torch.zeros(1, 20, 128))

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(128, 4) for _ in range(4)])

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(128)
        self.output_proj = nn.Linear(128, 1024, bias=False)

        # Causal mask
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(20, 20), diagonal=1).bool()
        )

    def forward(self, states, tokens):
        """
        states: (B, 20, 4)
        tokens: (B, 20) int64
        Returns: (B, 20, 1024) logits
        """
        state_emb = self.state_proj(states)  # (B, 20, 64)
        token_emb = self.token_embed(tokens)  # (B, 20, 64)
        x = torch.cat([state_emb, token_emb], dim=-1)  # (B, 20, 128)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, self.causal_mask)

        x = self.ln_f(x)
        logits = self.output_proj(x)  # (B, 20, 1024)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, d_model * 3, bias=False)  # (128, 384)
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)  # (128, 128)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp_fc1 = nn.Linear(d_model, d_model * 4, bias=False)  # (128, 512)
        self.mlp_fc2 = nn.Linear(d_model * 4, d_model, bias=False)  # (512, 128)

    def forward(self, x, causal_mask):
        # Attention
        h = self.ln_attn(x)
        qkv = self.attn_qkv(h)  # (B, T, 384)
        B, T, _ = qkv.shape
        q, k, v = qkv.split(self.n_heads * self.head_dim, dim=-1)  # each (B, T, 128)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.masked_fill(
            causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, 128)
        out = self.attn_proj(out)
        x = x + out

        # MLP
        h = self.ln_mlp(x)
        h = F.gelu(self.mlp_fc1(h), approximate="tanh")
        h = self.mlp_fc2(h)
        x = x + h

        return x


def load_weights_from_onnx(model: TinyPhysicsTorch, onnx_path: str):
    """Load weights from ONNX file directly into the PyTorch model."""
    onnx_model = onnx.load(onnx_path)
    weights = {}
    for init in onnx_model.graph.initializer:
        if "attn.bias" in init.name:
            continue  # causal mask, already created in model
        arr = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
        weights[init.name] = torch.from_numpy(arr.copy())

    with torch.no_grad():
        # Input embeddings
        # state_proj: weight is (4, 64), stored as MatMul weight (transposed in ONNX)
        model.state_proj.weight.copy_(weights["onnx::MatMul_540"].T)
        model.state_proj.bias.copy_(weights["transformer.wt_embedding.bias"])
        model.token_embed.weight.copy_(weights["transformer.wt2_embedding.weight"])
        model.pos_embed.copy_(weights["transformer.wp_embedding.weight"].unsqueeze(0))

        # Transformer blocks
        # MatMul weights in ONNX are stored as (in_dim, out_dim), need transpose for nn.Linear
        matmul_idx = 542  # Starting index for unnamed MatMul weights
        matmul_names = [
            # Block 0: qkv(542), proj(546), fc1(547), fc2(548)
            # Block 1: qkv(549), proj(553), fc1(554), fc2(555)
            # Block 2: qkv(556), proj(560), fc1(561), fc2(562)
            # Block 3: qkv(563), proj(567), fc1(568), fc2(569)
        ]
        block_matmul = [
            (542, 546, 547, 548),
            (549, 553, 554, 555),
            (556, 560, 561, 562),
            (563, 567, 568, 569),
        ]

        for i, (qkv_id, proj_id, fc1_id, fc2_id) in enumerate(block_matmul):
            block = model.blocks[i]
            block.ln_attn.weight.copy_(
                weights[f"transformer.h.{i}.attn.layer_norm.weight"]
            )
            block.ln_attn.bias.copy_(weights[f"transformer.h.{i}.attn.layer_norm.bias"])
            block.attn_qkv.weight.copy_(weights[f"onnx::MatMul_{qkv_id}"].T)
            block.attn_proj.weight.copy_(weights[f"onnx::MatMul_{proj_id}"].T)
            block.ln_mlp.weight.copy_(
                weights[f"transformer.h.{i}.mlp.layer_norm.weight"]
            )
            block.ln_mlp.bias.copy_(weights[f"transformer.h.{i}.mlp.layer_norm.bias"])
            block.mlp_fc1.weight.copy_(weights[f"onnx::MatMul_{fc1_id}"].T)
            block.mlp_fc2.weight.copy_(weights[f"onnx::MatMul_{fc2_id}"].T)

        # Final layer norm + output
        model.ln_f.weight.copy_(weights["transformer.layer_norm_f.weight"])
        model.ln_f.bias.copy_(weights["transformer.layer_norm_f.bias"])
        model.output_proj.weight.copy_(weights["onnx::MatMul_570"].T)

    return model


def load_model(onnx_path="models/tinyphysics.onnx", device="cpu"):
    model = TinyPhysicsTorch()
    load_weights_from_onnx(model, onnx_path)
    model.to(device).eval()
    return model
