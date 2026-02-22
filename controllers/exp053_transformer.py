# exp053: Encoder-Decoder Transformer controller (discrete steer tokens, deterministic)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE, ACC_G

HIST_LEN   = 20
PLAN_LEN   = 50
D_MODEL    = 64
N_HEADS    = 4
N_ENC      = 2
N_DEC      = 2
D_FFN      = 256
STEER_BINS = 256

S_LAT  = 5.0
S_ROLL = 2.0
S_VEGO = 40.0
S_AEGO = 4.0


class SteerTokenizer:
    def __init__(self, n_bins=STEER_BINS, lo=STEER_RANGE[0], hi=STEER_RANGE[1]):
        self.n_bins = n_bins
        self.lo, self.hi = lo, hi
        self.bins = np.linspace(lo, hi, n_bins).astype(np.float32)

    def encode(self, value):
        value = np.clip(value, self.lo, self.hi)
        return int(np.digitize(value, self.bins, right=True))

    def decode(self, token):
        return float(self.bins[token])


class EncDecPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_proj = nn.Linear(4, D_MODEL)
        self.plan_pos = nn.Embedding(PLAN_LEN, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FFN,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_ENC)

        self.hist_proj = nn.Linear(5, D_MODEL)
        self.steer_embed = nn.Embedding(STEER_BINS, D_MODEL)
        self.hist_pos = nn.Embedding(HIST_LEN, D_MODEL)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FFN,
            dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=N_DEC)

        self.steer_head = nn.Linear(D_MODEL, STEER_BINS)
        self.value_head = nn.Linear(D_MODEL, 1)

        self.register_buffer('_causal_mask',
            nn.Transformer.generate_square_subsequent_mask(HIST_LEN), persistent=False)

    def forward_last(self, plan_feat, hist_feat, hist_tok):
        pos_p = torch.arange(PLAN_LEN, device=plan_feat.device)
        x_enc = self.plan_proj(plan_feat) + self.plan_pos(pos_p)
        memory = self.encoder(x_enc)

        pos_h = torch.arange(HIST_LEN, device=hist_feat.device)
        x_dec = self.hist_proj(hist_feat) + self.steer_embed(hist_tok) + self.hist_pos(pos_h)
        decoded = self.decoder(x_dec, memory, tgt_mask=self._causal_mask, tgt_is_causal=True)

        logits = self.steer_head(decoded[:, -1, :])
        return logits


class Controller(BaseController):
    def __init__(self):
        exp = Path(__file__).parent.parent / 'experiments' / 'exp053_transformer'
        ckpt = None
        for name in ('best_model.pt', 'final_model.pt'):
            p = exp / name
            if p.exists(): ckpt = str(p); break
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {exp}")

        self.model = EncDecPolicy()
        data = torch.load(ckpt, weights_only=False, map_location='cpu')
        key = 'model' if 'model' in data else 'ac'
        self.model.load_state_dict(data[key])
        self.model.eval()

        self.tok = SteerTokenizer()
        self.n = 0
        self._steer_tok_hist = [STEER_BINS // 2] * HIST_LEN
        self._cur_lat_hist = [0.0] * HIST_LEN
        self._target_hist = [0.0] * HIST_LEN
        self._roll_hist = [0.0] * HIST_LEN
        self._v_hist = [0.0] * HIST_LEN
        self._a_hist = [0.0] * HIST_LEN

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        step_idx = CONTEXT_LENGTH + self.n - 1

        if step_idx < CONTROL_START_IDX:
            self._steer_tok_hist = self._steer_tok_hist[1:] + [STEER_BINS // 2]
            self._cur_lat_hist = self._cur_lat_hist[1:] + [current_lataccel]
            self._target_hist = self._target_hist[1:] + [target_lataccel]
            self._roll_hist = self._roll_hist[1:] + [state.roll_lataccel]
            self._v_hist = self._v_hist[1:] + [state.v_ego]
            self._a_hist = self._a_hist[1:] + [state.a_ego]
            return 0.0

        # Build plan [1, 50, 4]
        plan = np.zeros((1, PLAN_LEN, 4), dtype=np.float32)
        fp_lat = getattr(future_plan, 'lataccel', []) or []
        fp_roll = getattr(future_plan, 'roll_lataccel', []) or []
        fp_v = getattr(future_plan, 'v_ego', []) or []
        fp_a = getattr(future_plan, 'a_ego', []) or []
        w = min(len(fp_lat), PLAN_LEN)
        if w > 0:
            plan[0, :w, 0] = np.array(fp_lat[:w], np.float32) / S_LAT
            plan[0, :w, 1] = np.array(fp_roll[:w], np.float32) / S_ROLL
            plan[0, :w, 2] = np.array(fp_v[:w], np.float32) / S_VEGO
            plan[0, :w, 3] = np.array(fp_a[:w], np.float32) / S_AEGO
            if w < PLAN_LEN:
                plan[0, w:] = plan[0, w-1:w]
        else:
            plan[0, :, 0] = target_lataccel / S_LAT
            plan[0, :, 1] = state.roll_lataccel / S_ROLL
            plan[0, :, 2] = state.v_ego / S_VEGO
            plan[0, :, 3] = state.a_ego / S_AEGO

        # Build hist features [1, 20, 5]
        hfeat = np.zeros((1, HIST_LEN, 5), dtype=np.float32)
        for j in range(HIST_LEN):
            hfeat[0, j, 0] = self._target_hist[j] / S_LAT
            hfeat[0, j, 1] = self._cur_lat_hist[j] / S_LAT
            hfeat[0, j, 2] = self._roll_hist[j] / S_ROLL
            hfeat[0, j, 3] = self._v_hist[j] / S_VEGO
            hfeat[0, j, 4] = self._a_hist[j] / S_AEGO

        # Hist tokens [1, 20]
        htok = np.array([self._steer_tok_hist], dtype=np.int64)

        with torch.inference_mode():
            logits = self.model.forward_last(
                torch.from_numpy(plan),
                torch.from_numpy(hfeat),
                torch.from_numpy(htok))
            tok = logits.argmax(dim=-1).item()

        action = float(np.clip(self.tok.decode(tok), STEER_RANGE[0], STEER_RANGE[1]))

        # Update histories
        self._steer_tok_hist = self._steer_tok_hist[1:] + [tok]
        self._cur_lat_hist = self._cur_lat_hist[1:] + [current_lataccel]
        self._target_hist = self._target_hist[1:] + [target_lataccel]
        self._roll_hist = self._roll_hist[1:] + [state.roll_lataccel]
        self._v_hist = self._v_hist[1:] + [state.v_ego]
        self._a_hist = self._a_hist[1:] + [state.a_ego]
        return action
