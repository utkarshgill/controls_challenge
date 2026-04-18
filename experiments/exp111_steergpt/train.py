#!/usr/bin/env python3
"""SteerGPT: Autoregressive steer-token transformer trained on MPC expert data.

Architecture mirrors tinyphysics.onnx but predicts steer delta tokens:
  - Plan Encoder: bidirectional self-attention over 50-step future plan
  - Steer Decoder: causal self-attention + cross-attention to plan
  - Token: delta-steer discretized into 1024 bins over [-DELTA_MAX, DELTA_MAX]
  - State: [prev_steer, current_la, target_la, roll, v, a] at each position

Usage:
  python experiments/exp111_steergpt/train.py
  EPOCHS=60 BS=512 D_MODEL=128 python .../train.py
"""

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────
N_BINS = 1024
STEER_LO, STEER_HI = -2.0, 2.0
BINS = np.linspace(STEER_LO, STEER_HI, N_BINS)  # absolute steer bins
BINS_T = torch.from_numpy(BINS.astype(np.float32))

S_STEER = 2.0
S_LA = 5.0
S_ROLL = 2.0
S_V = 40.0
S_A = 4.0

# ── Hyperparameters ───────────────────────────────────────────
D_MODEL    = int(os.getenv("D_MODEL", "128"))
N_HEADS    = int(os.getenv("N_HEADS", "4"))
N_ENC      = int(os.getenv("N_ENC", "4"))
N_DEC      = int(os.getenv("N_DEC", "4"))
K          = int(os.getenv("K", "20"))
FUTURE_K   = int(os.getenv("FUTURE_K", "50"))
DROPOUT    = float(os.getenv("DROPOUT", "0.1"))
STRIDE     = int(os.getenv("STRIDE", "1"))
D_STATE    = 5    # [current_la, target_la, roll, v, a] — no prev_steer (redundant with token)
D_PLAN     = 4

EPOCHS     = int(os.getenv("EPOCHS", "60"))
LR         = float(os.getenv("LR", "3e-4"))
LR_MIN     = float(os.getenv("LR_MIN", "1e-5"))
BS         = int(os.getenv("BS", "512"))
GRAD_CLIP  = 1.0
VAL_FRAC   = 0.05
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N     = int(os.getenv("EVAL_N", "100"))

DATA_PATH  = Path(os.getenv("DATA_PATH",
    str(ROOT / "experiments/exp111_steergpt/data/steergpt_data.npz")))
CKPT_DIR   = Path(os.getenv("CKPT_DIR",
    str(ROOT / "experiments/exp111_steergpt/checkpoints")))
MODEL_PATH = str(ROOT / "models" / "tinyphysics.onnx")


# ══════════════════════════════════════════════════════════════
#  Tokenizer
# ══════════════════════════════════════════════════════════════

def encode_steer(steer):
    """Absolute steer value → token index."""
    clipped = np.clip(steer, STEER_LO, STEER_HI)
    return np.digitize(clipped, BINS, right=True).clip(0, N_BINS - 1)

def decode_steer(token):
    """Token index → absolute steer value."""
    return BINS[token]


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════

class SteerGPT(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_enc=N_ENC, n_dec=N_DEC,
                 n_bins=N_BINS, d_state=D_STATE, d_plan=D_PLAN,
                 ctx=K, future_k=FUTURE_K, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.ctx = ctx
        self.n_bins = n_bins

        self.plan_proj = nn.Linear(d_plan, d_model)
        self.plan_pos = nn.Parameter(torch.randn(1, future_k, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.plan_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        half = d_model // 2
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
        decoded = self.decoder(x, plan_memory, tgt_mask=mask, tgt_is_causal=True)
        return self.head(self.ln_f(decoded))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════
#  Dataset — precomputed tensors, zero per-item compute
# ══════════════════════════════════════════════════════════════

class SteerDataset(torch.utils.data.Dataset):
    """All windows precomputed as contiguous numpy arrays. __getitem__ is a slice."""

    def __init__(self, steer, current_la, target_la, roll_la, v_ego, a_ego,
                 route_indices, ctx=K, future_k=FUTURE_K):
        KK = ctx
        F = future_k
        N = len(route_indices)
        T = steer.shape[1]

        # Tokenize absolute steer values
        steer_tokens = encode_steer(steer).astype(np.int64)

        base_min = 1
        base_max = T - KK - F - 1
        bases = list(range(base_min, base_max + 1, STRIDE))
        n_per = len(bases)
        total = N * n_per

        # Pre-allocate big arrays
        all_states = np.empty((total, KK, D_STATE), dtype=np.float32)
        all_in_tok = np.empty((total, KK), dtype=np.int64)
        all_tgt_tok = np.empty((total, KK), dtype=np.int64)
        all_plan = np.empty((total, F, D_PLAN), dtype=np.float32)

        print(f"    Precomputing {total:,} windows...", end="", flush=True)
        t0 = time.time()
        idx = 0
        for ri, r in enumerate(route_indices):
            for base in bases:
                # State and token from SAME step (no offset)
                # current_la[t] reflects steer[t] (the token), NOT steer[t+1] (the target)
                all_states[idx, :, 0] = current_la[r, base:base+KK] / S_LA
                all_states[idx, :, 1] = target_la[r, base:base+KK] / S_LA
                all_states[idx, :, 2] = roll_la[r, base:base+KK] / S_ROLL
                all_states[idx, :, 3] = v_ego[r, base:base+KK] / S_V
                all_states[idx, :, 4] = a_ego[r, base:base+KK] / S_A
                all_in_tok[idx] = steer_tokens[r, base:base+KK]
                all_tgt_tok[idx] = steer_tokens[r, base+1:base+KK+1]
                ps = base + KK + 1
                pe = ps + F
                avail = min(pe, T) - ps
                all_plan[idx, :avail, 0] = target_la[r, ps:ps+avail] / S_LA
                all_plan[idx, :avail, 1] = roll_la[r, ps:ps+avail] / S_ROLL
                all_plan[idx, :avail, 2] = v_ego[r, ps:ps+avail] / S_V
                all_plan[idx, :avail, 3] = a_ego[r, ps:ps+avail] / S_A
                if avail < F:
                    all_plan[idx, avail:] = all_plan[idx, avail-1:avail]
                idx += 1
        print(f" {time.time()-t0:.1f}s", flush=True)

        self.states = all_states
        self.in_tok = all_in_tok
        self.tgt_tok = all_tgt_tok
        self.plan = all_plan

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.in_tok[idx], self.plan[idx], self.tgt_tok[idx]


# ══════════════════════════════════════════════════════════════
#  Batched sim eval (100 routes)
# ══════════════════════════════════════════════════════════════

def evaluate(model, eval_csvs, model_path):
    """Run SteerGPT controller on N routes using BatchedSimulator (GPU path)."""
    from tinyphysics_batched import BatchedSimulator
    import warnings
    warnings.filterwarnings("ignore")

    N = len(eval_csvs)
    sim = BatchedSimulator(model_path, csv_files=[str(f) for f in eval_csvs])
    T_max = sim.T

    # Buffers for autoregressive state (GPU)
    steer_hist = torch.zeros((N, T_max), dtype=torch.float64, device=DEV)
    steer_tok_hist = torch.zeros((N, T_max), dtype=torch.int64, device=DEV)
    cur_la_hist = torch.zeros((N, T_max), dtype=torch.float64, device=DEV)
    bins_gpu = torch.from_numpy(BINS.astype(np.float64)).to(DEV)

    # Fill warmup from sim (steps 0..CL already in sim buffers)
    steer_hist[:, :CONTROL_START_IDX] = sim.action_history[:, :CONTROL_START_IDX]
    cur_la_hist[:, :CONTROL_START_IDX] = sim.current_lataccel_history[:, :CONTROL_START_IDX]
    warmup_clamped = steer_hist[:, :CONTROL_START_IDX].clamp(STEER_LO, STEER_HI)
    steer_tok_hist[:, :CONTROL_START_IDX] = torch.bucketize(
        warmup_clamped, bins_gpu, right=False).clamp(0, N_BINS - 1)

    def controller_fn(step_idx, sim_ref):
        """GPU-path: controller_fn(step_idx, sim) -> actions tensor on GPU."""
        dg = sim_ref.data_gpu
        cur_la = sim_ref.current_lataccel  # (N,) current step

        # Track current_la for historical state features
        cur_la_hist[:, step_idx] = cur_la

        action_idx = step_idx - CONTROL_START_IDX
        if action_idx < 0:
            return torch.zeros(N, dtype=torch.float64, device=DEV)

        K_use = min(action_idx + 1, K)

        # Build state features with HISTORICAL per-position data
        # Training alignment: position i has state from step (tok_step + 1)
        # where tok_step = step_idx - K_use + i
        states_t = torch.zeros((N, K_use, D_STATE), dtype=torch.float32, device=DEV)
        for i in range(K_use):
            # Same step as token (no offset)
            obs_step = step_idx - K_use + i
            obs_step = max(obs_step, 0)
            states_t[:, i, 0] = cur_la_hist[:, obs_step].float() / S_LA
            states_t[:, i, 1] = dg["target_lataccel"][:, obs_step].float() / S_LA
            states_t[:, i, 2] = dg["roll_lataccel"][:, obs_step].float() / S_ROLL
            states_t[:, i, 3] = dg["v_ego"][:, obs_step].float() / S_V
            states_t[:, i, 4] = dg["a_ego"][:, obs_step].float() / S_A

        # Tokens: absolute steer tokens from history
        tok_end = step_idx
        tok_start = max(0, tok_end - K_use)
        tokens_t = steer_tok_hist[:, tok_start:tok_end]
        if tokens_t.shape[1] < K_use:
            pad = torch.zeros((N, K_use - tokens_t.shape[1]), dtype=torch.int64, device=DEV)
            tokens_t = torch.cat([pad, tokens_t], dim=1)

        # Plan: next FUTURE_K steps of (target, roll, v, a)
        ps = step_idx + 1
        pe = min(ps + FUTURE_K, T_max)
        F_use = pe - ps
        plan_t = torch.zeros((N, FUTURE_K, D_PLAN), dtype=torch.float32, device=DEV)
        if F_use > 0:
            plan_t[:, :F_use, 0] = dg["target_lataccel"][:, ps:pe].float() / S_LA
            plan_t[:, :F_use, 1] = dg["roll_lataccel"][:, ps:pe].float() / S_ROLL
            plan_t[:, :F_use, 2] = dg["v_ego"][:, ps:pe].float() / S_V
            plan_t[:, :F_use, 3] = dg["a_ego"][:, ps:pe].float() / S_A
            if F_use < FUTURE_K:
                plan_t[:, F_use:] = plan_t[:, F_use-1:F_use]

        with torch.no_grad():
            logits = model(states_t, tokens_t, plan_t)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            new_steer = (probs.double() @ bins_gpu).clamp(STEER_RANGE[0], STEER_RANGE[1])

        tok = torch.bucketize(new_steer.clamp(STEER_LO, STEER_HI),
                              bins_gpu, right=False).clamp(0, N_BINS - 1)
        steer_tok_hist[:, step_idx] = tok
        steer_hist[:, step_idx] = new_steer

        return new_steer

    costs = sim.rollout(controller_fn)
    return costs["total_cost"].mean(), costs["total_cost"].std()


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"SteerGPT: d_model={D_MODEL} heads={N_HEADS} enc={N_ENC} dec={N_DEC} K={K}")
    print(f"  bins={N_BINS} steer=[{STEER_LO},{STEER_HI}] eval_every={EVAL_EVERY} eval_n={EVAL_N}")
    print(f"  device={DEV}")
    print("=" * 70, flush=True)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...", flush=True)
    d = np.load(DATA_PATH)
    steer = d["steer"]
    current_la = d["current_la"]
    target_la = d["target_la"]
    roll_la = d["roll_la"]
    v_ego_arr = d["v_ego"]
    a_ego_arr = d["a_ego"]
    N, T = steer.shape
    print(f"  {N} routes, {T} steps each", flush=True)

    # Train/val split by route
    perm = np.random.RandomState(42).permutation(N)
    n_val = max(int(N * VAL_FRAC), 10)
    val_routes = perm[:n_val]
    train_routes = perm[n_val:]

    train_ds = SteerDataset(steer, current_la, target_la, roll_la, v_ego_arr, a_ego_arr,
                            train_routes, ctx=K, future_k=FUTURE_K)
    val_ds = SteerDataset(steer, current_la, target_la, roll_la, v_ego_arr, a_ego_arr,
                          val_routes, ctx=K, future_k=FUTURE_K)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BS, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4)

    print(f"  Train: {len(train_ds):,} windows ({len(train_routes)} routes)")
    print(f"  Val:   {len(val_ds):,} windows ({n_val} routes)", flush=True)

    # Model
    model = SteerGPT().to(DEV)
    bins_dev = BINS_T.to(DEV)  # (N_BINS,) bin centers on GPU
    print(f"  Params: {model.count_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_sim = float("inf")
    best_epoch = -1

    # Eval CSVs
    eval_csvs = sorted(Path(ROOT / "data").glob("*.csv"))[:EVAL_N]

    n_batches = len(train_loader)
    print(f"\nTraining for {EPOCHS} epochs, BS={BS}, {n_batches} batches/epoch")
    print("-" * 80, flush=True)

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_tokens = 0.0, 0, 0

        for batch_i, (states, in_tok, plan, tgt_tok) in enumerate(train_loader):
            states = states.to(DEV, non_blocking=True)
            in_tok = in_tok.to(DEV, non_blocking=True)
            plan = plan.to(DEV, non_blocking=True)
            tgt_tok = tgt_tok.to(DEV, non_blocking=True)

            logits = model(states, in_tok, plan)  # (B, K, N_BINS)
            loss = F.cross_entropy(logits.reshape(-1, N_BINS), tgt_tok.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            B = states.shape[0]
            n_tok = B * K
            total_loss += loss.item() * n_tok
            total_correct += (logits.argmax(-1) == tgt_tok).sum().item()
            total_tokens += n_tok

            if batch_i % 500 == 0 and batch_i > 0:
                print(f"    [{batch_i}/{n_batches}] loss={total_loss/total_tokens:.4f} "
                      f"acc={total_correct/total_tokens:.3f}", flush=True)

        sched.step()
        train_loss = total_loss / total_tokens
        train_acc = total_correct / total_tokens

        # Validation CE
        model.eval()
        val_loss_sum, val_correct, val_tokens = 0.0, 0, 0
        with torch.no_grad():
            for states, in_tok, plan, tgt_tok in val_loader:
                states = states.to(DEV, non_blocking=True)
                in_tok = in_tok.to(DEV, non_blocking=True)
                plan = plan.to(DEV, non_blocking=True)
                tgt_tok = tgt_tok.to(DEV, non_blocking=True)
                logits = model(states, in_tok, plan)
                loss = F.cross_entropy(logits.reshape(-1, N_BINS), tgt_tok.reshape(-1))
                n_tok = states.shape[0] * K
                val_loss_sum += loss.item() * n_tok
                val_correct += (logits.argmax(-1) == tgt_tok).sum().item()
                val_tokens += n_tok

        val_loss = val_loss_sum / val_tokens
        val_acc = val_correct / val_tokens
        elapsed = time.time() - t0

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "d_model": D_MODEL, "n_heads": N_HEADS,
                    "n_enc": N_ENC, "n_dec": N_DEC,
                    "n_bins": N_BINS, "steer_lo": STEER_LO, "steer_hi": STEER_HI,
                    "ctx": K, "future_k": FUTURE_K,
                    "d_state": D_STATE, "d_plan": D_PLAN,
                },
            }, CKPT_DIR / "best_model.pt")
            tag = " *BEST*"

        line = (f"  Ep {ep:3d} | train {train_loss:.4f} acc {train_acc:.3f} | "
                f"val {val_loss:.4f} acc {val_acc:.3f}{tag} | "
                f"lr {opt.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        # Sim eval every EVAL_EVERY epochs
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            sim_mean, sim_std = evaluate(model, eval_csvs, MODEL_PATH)
            sim_tag = ""
            if sim_mean < best_sim:
                best_sim = sim_mean
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "d_model": D_MODEL, "n_heads": N_HEADS,
                        "n_enc": N_ENC, "n_dec": N_DEC,
                        "n_bins": N_BINS, "steer_lo": STEER_LO, "steer_hi": STEER_HI,
                        "ctx": K, "future_k": FUTURE_K,
                        "d_state": D_STATE, "d_plan": D_PLAN,
                    },
                }, CKPT_DIR / "best_sim_model.pt")
                sim_tag = " ★"
            line += f"  sim={sim_mean:.1f}±{sim_std:.0f}{sim_tag}"

        print(line, flush=True)

    print("-" * 80)
    print(f"Best val CE: {best_val:.4f} at epoch {best_epoch}")
    print(f"Best sim cost: {best_sim:.1f}")
    print(f"Saved to {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
