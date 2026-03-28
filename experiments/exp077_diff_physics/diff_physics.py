"""Differentiable PyTorch reimplementation of tinyphysics.onnx.

This is a frozen copy of the ONNX model's weights in a fully differentiable
PyTorch graph. The key output is `expected_lataccel`: the probability-weighted
mean of the tokenizer bins, which is a smooth differentiable function of the
steer action input. This allows backpropagation of tracking loss through
the physics model.

Architecture (from ONNX inspection):
  - Input embedding: states(b,20,4)->Linear(4,64)+bias, tokens(b,20)->Embed(1024,64)
    Concat -> (b,20,128) + positional_embedding(20,128)
  - 4x TransformerBlock (pre-norm, causal, 4 heads, dim=128, mlp_dim=512, tanh activation)
  - Final LayerNorm -> Linear(128,1024) -> logits
  - Softmax(logits/temperature) -> probs -> E[lataccel] = sum(probs * bins)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
N_HEADS = 4
D_MODEL = 128
D_HEAD = D_MODEL // N_HEADS  # 32
D_MLP = 512
N_LAYERS = 4
LATACCEL_RANGE = (-5.0, 5.0)
MAX_ACC_DELTA = 0.5
TEMPERATURE = 0.8


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(D_MODEL)
        self.c_attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.c_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool)
            ).view(1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH),
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.layer_norm(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(D_MODEL, dim=-1)
        q = q.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (D_HEAD**-0.5)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return x + y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(D_MODEL)
        self.c_fc = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.c_proj = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x):
        h = self.layer_norm(x)
        h = torch.tanh(self.c_fc(h))
        h = self.c_proj(h)
        return x + h


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class DifferentiablePhysics(nn.Module):
    """Differentiable reimplementation of tinyphysics.onnx.

    All weights are frozen. The forward pass computes expected lataccel
    as a differentiable function of (states, tokens), allowing gradient
    flow back through the steer_action dimension of states.
    """

    def __init__(self, weights_path=None):
        super().__init__()

        # Input embeddings
        self.wt_embedding = nn.Linear(4, 64, bias=True)
        self.wt2_embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.wp_embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)

        # Transformer blocks
        self.h = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        # Output
        self.layer_norm_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Tokenizer bins for expected value computation
        bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(
            np.float32
        )
        self.register_buffer("bins", torch.from_numpy(bins))

        # Load weights if provided
        if weights_path is not None:
            self._load_weights(weights_path)

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def _load_weights(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)

        # Input embeddings
        self.wt_embedding.weight.data.copy_(sd["wt_embedding.weight"])
        self.wt_embedding.bias.data.copy_(sd["wt_embedding.bias"])
        self.wt2_embedding.weight.data.copy_(sd["wt2_embedding.weight"])
        self.wp_embedding.weight.data.copy_(sd["wp_embedding.weight"])

        # Transformer blocks
        for i in range(N_LAYERS):
            block = self.h[i]
            # Attention
            block.attn.layer_norm.weight.data.copy_(sd[f"h.{i}.attn.layer_norm.weight"])
            block.attn.layer_norm.bias.data.copy_(sd[f"h.{i}.attn.layer_norm.bias"])
            block.attn.c_attn.weight.data.copy_(sd[f"h.{i}.attn.c_attn.weight"])
            block.attn.c_proj.weight.data.copy_(sd[f"h.{i}.attn.c_proj.weight"])
            # MLP
            block.mlp.layer_norm.weight.data.copy_(sd[f"h.{i}.mlp.layer_norm.weight"])
            block.mlp.layer_norm.bias.data.copy_(sd[f"h.{i}.mlp.layer_norm.bias"])
            block.mlp.c_fc.weight.data.copy_(sd[f"h.{i}.mlp.c_fc.weight"])
            block.mlp.c_proj.weight.data.copy_(sd[f"h.{i}.mlp.c_proj.weight"])

        # Output
        self.layer_norm_f.weight.data.copy_(sd["layer_norm_f.weight"])
        self.layer_norm_f.bias.data.copy_(sd["layer_norm_f.bias"])
        self.lm_head.weight.data.copy_(sd["lm_head.weight"])

    def forward_logits(self, states, tokens):
        """Full transformer forward pass.

        Args:
            states: (B, T, 4) float — [steer_action, roll_lataccel, v_ego, a_ego]
            tokens: (B, T) long — tokenized past lataccel predictions

        Returns:
            logits: (B, T, 1024)
        """
        B, T = states.shape[:2]

        # Input embeddings
        h_state = self.wt_embedding(states)  # (B, T, 64)
        h_token = self.wt2_embedding(tokens)  # (B, T, 64)
        h = torch.cat([h_state, h_token], dim=-1)  # (B, T, 128)

        # Positional embedding
        pos = torch.arange(T, device=states.device)
        h = h + self.wp_embedding(pos)  # broadcast over batch

        # Transformer blocks
        for block in self.h:
            h = block(h)

        # Output
        h = self.layer_norm_f(h)
        logits = self.lm_head(h)  # (B, T, 1024)
        return logits

    def expected_lataccel(self, states, tokens, temperature=TEMPERATURE):
        """Compute E[lataccel] — the differentiable expected value.

        This is the key function for TBPTT. It computes:
            probs = softmax(logits[:, -1, :] / temperature)
            E[lataccel] = sum(probs * bins)

        The gradient d(E[lataccel])/d(steer_action) flows through softmax and
        the full transformer, providing an exact per-step gradient.

        Args:
            states: (B, T, 4) float
            tokens: (B, T) long

        Returns:
            expected: (B,) — expected lataccel values
        """
        logits = self.forward_logits(states, tokens)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        expected = (probs * self.bins.unsqueeze(0)).sum(dim=-1)
        return expected

    def tokenize(self, lataccel):
        """Differentiable-friendly tokenization.

        For the forward pass we need token indices (non-differentiable).
        This uses straight-through: detach the tokenization but keep
        the lataccel tensor in the graph for gradient flow through
        expected_lataccel.

        Args:
            lataccel: (B,) float tensor

        Returns:
            tokens: (B,) long tensor
        """
        clamped = lataccel.clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        # Bucketize: find nearest bin
        tokens = torch.bucketize(clamped, self.bins, right=False).clamp(
            0, VOCAB_SIZE - 1
        )
        return tokens
