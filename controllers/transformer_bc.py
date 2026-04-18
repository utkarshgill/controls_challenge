"""Transformer BC controller: encoder-decoder with cross-attention.

Encoder (non-causal): 50-step future plan → representation of road ahead
Decoder (causal): K-step history → cross-attends to future → steer delta

No hand-engineered features. Raw physical quantities normalized to O(1).
"""

import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import deque

from . import BaseController
from tinyphysics import (
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    ACC_G,
    FUTURE_PLAN_STEPS,
)

# ── Must match training ──────────────────────────────────────
K_HIST = 20
D_MODEL = 64
N_HEADS = 4
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2
FUTURE_LEN = 50
HIST_DIM = 6
FUTURE_DIM = 4
DELTA_SCALE = 0.25
MAX_DELTA = 0.5

# Normalization (same as training)
S_LA = 5.0
S_V = 40.0
S_A = 4.0
S_ACT = 2.0

torch.set_num_threads(1)


class TransformerBC(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_enc=N_ENC_LAYERS,
        n_dec=N_DEC_LAYERS,
        future_len=FUTURE_LEN,
        k_hist=K_HIST,
        dropout=0.0,
    ):
        super().__init__()
        self.enc_proj = nn.Linear(FUTURE_DIM, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, future_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        self.dec_proj = nn.Linear(HIST_DIM, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, k_hist, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(k_hist, k_hist), diagonal=1).bool()
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )

    def forward(self, hist, future):
        mem = self.enc_proj(future) + self.enc_pos
        mem = self.encoder(mem)
        K = hist.shape[1]
        x = self.dec_proj(hist) + self.dec_pos[:, :K]
        mask = self.causal_mask[:K, :K]
        x = self.decoder(x, mem, tgt_mask=mask, tgt_is_causal=True)
        out = self.head(x[:, -1, :])
        return F.softplus(out[:, 0]) + 1.0, F.softplus(out[:, 1]) + 1.0


def _pad_future(vals, k, fallback):
    if vals is not None and len(vals) >= k:
        return np.asarray(vals[:k], np.float32)
    if vals is not None and len(vals) > 0:
        arr = np.asarray(vals, np.float32)
        return np.pad(arr, (0, k - len(arr)), mode="edge")
    return np.full(k, fallback, np.float32)


class Controller(BaseController):
    def __init__(self):
        ckpt_env = os.getenv("MODEL", "").strip()
        ckpt = ckpt_env or str(
            Path(__file__).parent.parent
            / "experiments"
            / "exp110_mpc"
            / "checkpoints"
            / "transformer_bc_model.pt"
        )
        self.model = TransformerBC()
        self.model.load_state_dict(
            torch.load(ckpt, weights_only=True, map_location="cpu")
        )
        self.model.eval()

        self.n = 0
        self.prev_action = 0.0
        # History buffer: deque of (action, target, current, roll, v_ego, a_ego) normalized
        self.hist = deque(maxlen=K_HIST)
        # Pre-fill with zeros
        for _ in range(K_HIST):
            self.hist.append(np.zeros(HIST_DIM, np.float32))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        warmup_n = CONTROL_START_IDX - CONTEXT_LENGTH

        if self.n <= warmup_n:
            # During warmup, still record history
            token = np.array(
                [
                    0.0 / S_ACT,
                    target_lataccel / S_LA,
                    current_lataccel / S_LA,
                    state.roll_lataccel / S_LA,
                    state.v_ego / S_V,
                    state.a_ego / S_A,
                ],
                np.float32,
            )
            self.hist.append(token)
            self.prev_action = 0.0
            return 0.0

        # Build future plan: (50, 4) normalized
        f_target = (
            _pad_future(
                getattr(future_plan, "lataccel", None), FUTURE_LEN, target_lataccel
            )
            / S_LA
        )
        f_roll = (
            _pad_future(
                getattr(future_plan, "roll_lataccel", None),
                FUTURE_LEN,
                state.roll_lataccel,
            )
            / S_LA
        )
        f_v = (
            _pad_future(getattr(future_plan, "v_ego", None), FUTURE_LEN, state.v_ego)
            / S_V
        )
        f_a = (
            _pad_future(getattr(future_plan, "a_ego", None), FUTURE_LEN, state.a_ego)
            / S_A
        )
        future = np.stack([f_target, f_roll, f_v, f_a], axis=-1)  # (50, 4)
        future = np.clip(future, -5.0, 5.0)

        # Build history: (K, 6)
        hist_arr = np.array(list(self.hist), np.float32)  # (K, 6)
        hist_arr = np.clip(hist_arr, -5.0, 5.0)

        with torch.no_grad():
            h_t = torch.from_numpy(hist_arr).unsqueeze(0)  # (1, K, 6)
            f_t = torch.from_numpy(future).unsqueeze(0)  # (1, 50, 4)
            alpha, beta = self.model(h_t, f_t)
            a_p, b_p = alpha.item(), beta.item()

        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = float(np.clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA))
        action = float(np.clip(self.prev_action + delta, *STEER_RANGE))

        # Record this step in history — action field is prev_action (matching
        # training data where obs[9] = prev_act at each step, i.e. the action
        # from the PREVIOUS step, not the one just decided)
        token = np.array(
            [
                self.prev_action / S_ACT,
                target_lataccel / S_LA,
                current_lataccel / S_LA,
                state.roll_lataccel / S_LA,
                state.v_ego / S_V,
                state.a_ego / S_A,
            ],
            np.float32,
        )
        self.hist.append(token)
        self.prev_action = action
        return action
