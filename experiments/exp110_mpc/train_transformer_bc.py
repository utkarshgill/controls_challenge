#!/usr/bin/env python3
"""Transformer BC: Encoder-decoder with cross-attention on MPC expert data.

ENCODER (future plan, non-causal):
  Input:  (50, 4) — [target, roll, v_ego, a_ego] for next 50 steps
  Output: (50, d) — rich representation of "the road ahead"

DECODER (history, causal):
  Input:  (K, 6) — [action, target, current, roll, v_ego, a_ego] for last K steps
  Q=history, KV=future → cross-attention
  Output: last token → MLP → Beta(α, β) for steer delta

No hand-engineered features. The transformer learns what matters from raw
physical quantities.

Usage:
  /venv/main/bin/python3 experiments/exp110_mpc/train_transformer_bc.py
"""

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
DEV = torch.device("cuda")

# ── Data layout in bc_data.npz obs (256-dim, from exp055) ────
# Stored SCALED. We undo the scaling to get raw values, then re-normalize.
# [0]*5   = target_lataccel    [1]*5   = current_lataccel
# [6]*40  = v_ego              [7]*4   = a_ego
# [8]*2   = roll_lataccel      [9]*2   = prev_action
# [16:36]*2 = h_act (20 steps) [36:56]*5 = h_lat (20 steps)
# [56:106]*5 = future target   [106:156]*2 = future roll
# [156:206]*40 = future v_ego  [206:256]*4 = future a_ego

N_ROUTES = 5000
N_CTRL = 400
DELTA_SCALE = 0.25

# ── Architecture ──────────────────────────────────────────────
K_HIST = int(os.getenv("K_HIST", "20"))
D_MODEL = int(os.getenv("D_MODEL", "64"))
N_HEADS = int(os.getenv("N_HEADS", "4"))
N_ENC_LAYERS = int(os.getenv("N_ENC", "2"))
N_DEC_LAYERS = int(os.getenv("N_DEC", "2"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

HIST_DIM = 6  # action, target, current, roll, v_ego, a_ego
FUTURE_DIM = 4  # target, roll, v_ego, a_ego
FUTURE_LEN = 50

# ── Training ──────────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "40"))
BC_LR = float(os.getenv("BC_LR", "3e-4"))
BC_LR_MIN = float(os.getenv("BC_LR_MIN", "1e-5"))
BC_BS = int(os.getenv("BC_BS", "512"))
GRAD_CLIP = 1.0
VAL_FRAC = 0.05
WD = float(os.getenv("WD", "1e-4"))

CKPT_DIR = Path(__file__).parent / "checkpoints"
BC_DATA_PATH = CKPT_DIR / "bc_data.npz"
OUT_PATH = CKPT_DIR / "transformer_bc_model.pt"

# ── Normalization constants (make features O(1)) ─────────────
# Applied during data extraction, stored here for the controller too.
S_LA = 5.0  # lataccel: target, current, roll
S_V = 40.0  # v_ego
S_A = 4.0  # a_ego
S_ACT = 2.0  # steer action


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════


class TransformerBC(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_enc=N_ENC_LAYERS,
        n_dec=N_DEC_LAYERS,
        future_len=FUTURE_LEN,
        k_hist=K_HIST,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_hist = k_hist

        # Encoder: future plan (non-causal)
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

        # Decoder: history (causal self-attn + cross-attn to future)
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

        # Causal mask for decoder self-attention
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(k_hist, k_hist), diagonal=1).bool()
        )

        # Output: last token → Beta params
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, hist, future):
        """
        hist:   (B, K, 6) — [action, target, current, roll, v_ego, a_ego]
        future: (B, 50, 4) — [target, roll, v_ego, a_ego]
        Returns: alpha (B,), beta (B,)
        """
        # Encode future
        mem = self.enc_proj(future) + self.enc_pos
        mem = self.encoder(mem)  # (B, 50, d)

        # Decode history with cross-attention to future
        K = hist.shape[1]
        x = self.dec_proj(hist) + self.dec_pos[:, :K]
        mask = self.causal_mask[:K, :K]
        x = self.decoder(x, mem, tgt_mask=mask, tgt_is_causal=True)  # (B, K, d)

        # Output from last token
        out = self.head(x[:, -1, :])  # (B, 2)
        alpha = F.softplus(out[:, 0]) + 1.0
        beta = F.softplus(out[:, 1]) + 1.0
        return alpha, beta


# ══════════════════════════════════════════════════════════════
#  Data extraction from flat bc_data.npz
# ══════════════════════════════════════════════════════════════


def extract_sequences(obs_flat, raw_flat):
    """Extract raw physical quantities from the 256-dim obs vectors.

    Returns per-route sequences:
      hist_data: (N, T, 6) — action, target, current, roll, v_ego, a_ego (normalized)
      future_data: (N, T, 50, 4) — target, roll, v_ego, a_ego (normalized)
      raw: (N, T) — target delta
    """
    N, T = N_ROUTES, N_CTRL
    obs = obs_flat.reshape(N, T, -1)
    raw = raw_flat.reshape(N, T)

    # Undo exp055 scaling to get raw values, then re-normalize simply
    target = obs[:, :, 0]  # already / S_LA
    current = obs[:, :, 1]  # already / S_LA
    roll = obs[:, :, 8]  # already / S_ROLL=2 → want / S_LA=5 → * 2/5
    v_ego = obs[:, :, 6]  # already / S_VEGO=40 → / S_V=40 → same
    a_ego = obs[:, :, 7]  # already / S_AEGO=4 → / S_A=4 → same
    prev_act = obs[:, :, 9]  # already / S_STEER=2 → / S_ACT=2 → same

    # For action at step t: prev_act + raw * DELTA_SCALE / S_ACT
    # Actually prev_act is scaled: real_prev = obs[:,t,9] * S_STEER
    # action_t = real_prev + raw_t * DELTA_SCALE
    # scaled_action_t = action_t / S_ACT = obs[:,t,9] + raw_t * DELTA_SCALE / S_ACT
    action = prev_act + raw * (DELTA_SCALE / S_ACT)

    # Normalize: roll is at /2 scale, want /5 scale → multiply by 2/5
    roll_norm = roll * (2.0 / S_LA)

    # History per step: (action, target, current, roll, v_ego, a_ego) all ~O(1)
    hist_data = np.stack(
        [action, target, current, roll_norm, v_ego, a_ego], axis=-1
    ).astype(np.float32)  # (N, T, 6)

    # Future plan at each step: (50, 4)
    # obs[56:106] = future target / S_LAT → keep as /S_LA
    # obs[106:156] = future roll / S_ROLL=2 → want /S_LA=5 → * 2/5
    # obs[156:206] = future v_ego / S_VEGO=40 → keep as /S_V
    # obs[206:256] = future a_ego / S_AEGO=4 → keep as /S_A
    f_target = obs[:, :, 56:106]  # (N, T, 50)
    f_roll = obs[:, :, 106:156] * (2.0 / S_LA)  # (N, T, 50)
    f_v = obs[:, :, 156:206]  # (N, T, 50)
    f_a = obs[:, :, 206:256]  # (N, T, 50)
    future_data = np.stack([f_target, f_roll, f_v, f_a], axis=-1).astype(
        np.float32
    )  # (N, T, 50, 4)

    return hist_data, future_data, raw


class SeqDataset(torch.utils.data.Dataset):
    """Yields windows of K consecutive steps from route sequences."""

    def __init__(self, hist, future, raw, k=K_HIST):
        # hist: (N, T, 6), future: (N, T, 50, 4), raw: (N, T)
        self.hist = hist
        self.future = future
        self.raw = raw
        self.k = k
        self.N, self.T = hist.shape[:2]
        self.per_route = self.T - k + 1  # valid start positions

    def __len__(self):
        return self.N * self.per_route

    def __getitem__(self, idx):
        ri = idx // self.per_route
        si = idx % self.per_route
        # History window: K consecutive steps
        h = self.hist[ri, si : si + self.k]  # (K, 6)
        # Future plan from the LAST step in the window (current decision point)
        f = self.future[ri, si + self.k - 1]  # (50, 4)
        # Target: raw delta at the last step
        r = self.raw[ri, si + self.k - 1]  # scalar
        return h, f, np.float32(r)


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print(
        f"Transformer BC: d={D_MODEL} heads={N_HEADS} "
        f"enc={N_ENC_LAYERS} dec={N_DEC_LAYERS} K={K_HIST}"
    )
    print("=" * 60)

    # Load and extract sequences
    print(f"\nLoading {BC_DATA_PATH}...")
    d = np.load(BC_DATA_PATH)
    hist, future, raw = extract_sequences(d["obs"], d["raw"])
    del d
    print(f"  hist: {hist.shape}, future: {future.shape}, raw: {raw.shape}")

    # Train/val split by route
    N = hist.shape[0]
    perm = np.random.RandomState(42).permutation(N)
    n_val = max(int(N * VAL_FRAC), 10)
    vi, ti = perm[:n_val], perm[n_val:]

    train_ds = SeqDataset(hist[ti], future[ti], raw[ti])
    val_ds = SeqDataset(hist[vi], future[vi], raw[vi])

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BC_BS,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=BC_BS, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"  Train: {len(train_ds):,} windows ({len(ti)} routes)")
    print(f"  Val:   {len(val_ds):,} windows ({n_val} routes)")

    # Model
    model = TransformerBC().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=BC_LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=BC_EPOCHS, eta_min=BC_LR_MIN
    )

    best_val = float("inf")
    best_ep = -1

    print(f"\nTraining {BC_EPOCHS} epochs, BS={BC_BS}")
    print("-" * 72)

    for ep in range(BC_EPOCHS):
        t0 = time.time()

        # ── Train ──
        model.train()
        tot, cnt = 0.0, 0
        for h, f, r in train_dl:
            h = h.to(DEV, non_blocking=True)
            f = f.to(DEV, non_blocking=True)
            r = r.to(DEV, non_blocking=True)

            alpha, beta = model(h, f)
            x = ((r + 1) / 2).clamp(1e-6, 1 - 1e-6)
            loss = -torch.distributions.Beta(alpha, beta).log_prob(x).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            tot += loss.item() * h.shape[0]
            cnt += h.shape[0]
        sched.step()
        train_loss = tot / cnt

        # ── Val ──
        model.eval()
        vt, vc = 0.0, 0
        with torch.no_grad():
            for h, f, r in val_dl:
                h = h.to(DEV, non_blocking=True)
                f = f.to(DEV, non_blocking=True)
                r = r.to(DEV, non_blocking=True)
                alpha, beta = model(h, f)
                x = ((r + 1) / 2).clamp(1e-6, 1 - 1e-6)
                loss = -torch.distributions.Beta(alpha, beta).log_prob(x).mean()
                vt += loss.item() * h.shape[0]
                vc += h.shape[0]
        val_loss = vt / vc

        # Diagnostics
        with torch.no_grad():
            mean_raw = (2 * alpha / (alpha + beta) - 1).mean().item()
            var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
            sigma_eff = (2 * torch.sqrt(var)).mean().item() * DELTA_SCALE

        dt = time.time() - t0
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            best_ep = ep
            torch.save(model.state_dict(), OUT_PATH)
            tag = " *BEST*"

        print(
            f"  Ep {ep:3d} | train {train_loss:.4f} | val {val_loss:.4f}{tag} "
            f"| mean {mean_raw:+.4f} σ_eff {sigma_eff:.4f} "
            f"| lr {opt.param_groups[0]['lr']:.1e} | {dt:.1f}s"
        )

    print("-" * 72)
    print(f"Best val: {best_val:.4f} @ epoch {best_ep}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
