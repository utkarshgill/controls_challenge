"""SteerGPT controller: autoregressive absolute-steer token prediction.

Architecture: plan encoder (bidirectional) + steer decoder (causal + cross-attn).
Mirrors tinyphysics.onnx but predicts steer instead of lataccel.
Decodes via expected value (softmax @ bin_centers) for sub-bin precision.
"""

import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, FUTURE_PLAN_STEPS

torch.set_num_threads(1)

# ── Constants (must match training) ───────────────────────────
N_BINS = 1024
STEER_LO, STEER_HI = -2.0, 2.0
BINS = np.linspace(STEER_LO, STEER_HI, N_BINS)
BINS_T = torch.from_numpy(BINS.astype(np.float32))

S_STEER = 2.0
S_LA = 5.0
S_ROLL = 2.0
S_V = 40.0
S_A = 4.0


def encode_steer(steer):
    clipped = np.clip(steer, STEER_LO, STEER_HI)
    return int(np.digitize(clipped, BINS, right=True).clip(0, N_BINS - 1))


# ── Model (must match training architecture) ──────────────────

class SteerGPT(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_enc=4, n_dec=4,
                 n_bins=N_BINS, d_state=6, d_plan=4, ctx=20, future_k=50, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.ctx = ctx

        half = d_model // 2
        self.plan_proj = nn.Linear(d_plan, d_model)
        self.plan_pos = nn.Parameter(torch.randn(1, future_k, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.plan_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        self.state_proj = nn.Linear(d_state, half)
        self.token_embed = nn.Embedding(n_bins, half)
        self.dec_pos = nn.Parameter(torch.randn(1, ctx, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(ctx, ctx), diagonal=1).bool())
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins, bias=False)

    def forward(self, states, tokens, plan):
        B, Kseq = tokens.shape
        plan_x = self.plan_proj(plan) + self.plan_pos[:, :plan.shape[1]]
        plan_memory = self.plan_encoder(plan_x)

        state_emb = self.state_proj(states)
        token_emb = self.token_embed(tokens)
        x = torch.cat([state_emb, token_emb], dim=-1)
        x = x + self.dec_pos[:, :Kseq]

        mask = self.causal_mask[:Kseq, :Kseq]
        decoded = self.decoder(x, plan_memory,
                               tgt_mask=mask, tgt_is_causal=True)
        return self.head(self.ln_f(decoded))


# ── Controller ────────────────────────────────────────────────

class Controller(BaseController):
    def __init__(self):
        ckpt_path = os.getenv("STEERGPT_CKPT", "").strip()
        if not ckpt_path:
            ckpt_path = str(Path(__file__).parent.parent /
                            "experiments/exp111_steergpt/checkpoints/best_model.pt")

        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        cfg = ckpt["config"]

        self.model = SteerGPT(
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_enc=cfg["n_enc"], n_dec=cfg["n_dec"],
            n_bins=cfg["n_bins"], d_state=cfg["d_state"],
            d_plan=cfg["d_plan"], ctx=cfg["ctx"],
            future_k=cfg["future_k"], dropout=0.0)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.K = cfg["ctx"]
        self.F = cfg["future_k"]

        self.steer_tokens = []  # absolute steer token indices
        self.state_hist = []    # (current_la, target_la, roll, v, a) per step
        self.prev_steer = 0.0
        self.n_calls = 0

    def _get_future(self, future_plan, attr, fallback, k=None):
        k = k or self.F
        vals = getattr(future_plan, attr, None)
        if vals is not None and len(vals) >= k:
            return np.asarray(vals[:k], np.float32)
        if vals is not None and len(vals) > 0:
            arr = np.asarray(vals, np.float32)
            return np.pad(arr, (0, k - len(arr)), mode="edge")
        return np.full(k, fallback, dtype=np.float32)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n_calls += 1

        obs = [current_lataccel, target_lataccel,
               state.roll_lataccel, state.v_ego, state.a_ego]
        self.state_hist.append(obs)

        n = len(self.steer_tokens)
        K_use = min(n, self.K)

        if K_use == 0:
            K_use = 1
            states_np = np.array([[
                current_lataccel / S_LA,
                target_lataccel / S_LA,
                state.roll_lataccel / S_ROLL,
                state.v_ego / S_V,
                state.a_ego / S_A,
            ]], dtype=np.float32)
            tokens_np = np.array([encode_steer(0.0)], dtype=np.int64)
        else:
            states_list = []
            for i in range(K_use):
                tok_step = n - K_use + i
                obs_step = tok_step  # same step as token (no offset)
                o = self.state_hist[obs_step] if obs_step < len(self.state_hist) else obs
                states_list.append([
                    o[0] / S_LA,
                    o[1] / S_LA,
                    o[2] / S_ROLL,
                    o[3] / S_V,
                    o[4] / S_A,
                ])
            states_np = np.array(states_list, dtype=np.float32)
            tokens_np = np.array(self.steer_tokens[-K_use:], dtype=np.int64)

        # Plan
        plan = np.stack([
            self._get_future(future_plan, "lataccel", target_lataccel) / S_LA,
            self._get_future(future_plan, "roll_lataccel", state.roll_lataccel) / S_ROLL,
            self._get_future(future_plan, "v_ego", state.v_ego) / S_V,
            self._get_future(future_plan, "a_ego", state.a_ego) / S_A,
        ], axis=-1).astype(np.float32)

        with torch.no_grad():
            s_t = torch.from_numpy(states_np).unsqueeze(0)
            t_t = torch.from_numpy(tokens_np).unsqueeze(0)
            p_t = torch.from_numpy(plan).unsqueeze(0)
            logits = self.model(s_t, t_t, p_t)
            # Expected value decode
            probs = torch.softmax(logits[0, -1], dim=-1)
            new_steer = float((probs * BINS_T).sum().item())

        new_steer = float(np.clip(new_steer, *STEER_RANGE))

        # Record token for next step
        self.steer_tokens.append(encode_steer(new_steer))
        self.prev_steer = new_steer
        return new_steer
