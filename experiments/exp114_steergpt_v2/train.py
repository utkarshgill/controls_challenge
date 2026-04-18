#!/usr/bin/env python3
"""exp114 — SteerGPT v2: autoregressive steer prediction mirroring tinyphysics.onnx.

Architecture:
  tinyphysics.onnx:  state=[steer, roll, v, a](4)  token=lataccel  → next lataccel
  SteerGPT v2:       state=[target, roll, v, a](4)  token=steer     → next steer delta

  Plan encoder (bidirectional): future 50 steps of road → memory
  Steer decoder (causal + cross-attn): 20-step history of (state, steer) → Beta params

  No current_lataccel — noise response is captured autoregressively in steer sequence.
  Beta distribution output for continuous precision. Delta parameterization.

Usage:
  python experiments/exp114_steergpt_v2/train.py
"""

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, STEER_RANGE

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────
K = int(os.getenv("K", "20"))           # history context
FUTURE_K = int(os.getenv("FUTURE_K", "50"))
DELTA_SCALE = 0.25
D_STATE = 5   # [target_la, roll_la, v_ego, a_ego, current_la] — road + noise signal
D_PLAN = 4

# Normalization
S_LA = 5.0
S_ROLL = 2.0
S_V = 40.0
S_A = 4.0
S_STEER = 2.0

# ── Architecture ──────────────────────────────────────────────
D_MODEL = int(os.getenv("D_MODEL", "128"))
N_HEADS = int(os.getenv("N_HEADS", "4"))
N_ENC   = int(os.getenv("N_ENC", "4"))
N_DEC   = int(os.getenv("N_DEC", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# ── Training ──────────────────────────────────────────────────
EPOCHS    = int(os.getenv("EPOCHS", "200"))
LR        = float(os.getenv("LR", "3e-4"))
LR_MIN    = float(os.getenv("LR_MIN", "1e-5"))
BS        = int(os.getenv("BS", "2048"))
GRAD_CLIP = 1.0
VAL_FRAC  = 0.05
STRIDE    = int(os.getenv("STRIDE", "1"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "10"))
EVAL_N     = int(os.getenv("EVAL_N", "100"))

DATA_PATH = Path(os.getenv("DATA_PATH",
    str(ROOT / "experiments/exp111_steergpt/data/steergpt_data.npz")))
CKPT_DIR  = Path(os.getenv("CKPT_DIR",
    str(Path(__file__).parent / "checkpoints")))
MODEL_PATH = str(ROOT / "models" / "tinyphysics.onnx")


# ══════════════════════════════════════════════════════════════
#  Model — mirrors tinyphysics.onnx structure
# ══════════════════════════════════════════════════════════════

class SteerGPT(nn.Module):
    """
    Plan encoder (bidirectional) + Steer decoder (causal + cross-attn).
    Input per position: state(4) + steer(1) → Concat(Linear(4,d/2), Linear(1,d/2)) + pos
    Output: Beta distribution params for delta steer.
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_enc=N_ENC, n_dec=N_DEC,
                 ctx=K, future_k=FUTURE_K, dropout=DROPOUT):
        super().__init__()

        # Plan encoder
        self.plan_proj = nn.Linear(D_PLAN, d_model)
        self.plan_pos = nn.Parameter(torch.randn(1, future_k, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.plan_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        # Steer decoder: state(4) → d/2, steer(1) → d/2, concat → d
        half = d_model // 2
        self.state_proj = nn.Linear(D_STATE, half)
        self.steer_proj = nn.Linear(1, half)
        self.dec_pos = nn.Parameter(torch.randn(1, ctx, d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(ctx, ctx), diagonal=1).bool())

        # Output: Beta params
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, states, steers, plan):
        """
        states: (B, K, 4)  — [target_la, roll, v, a] per step (normalized)
        steers: (B, K, 1)  — steer value per step (normalized)
        plan:   (B, F, 4)  — future road features

        Returns: (B, K, 2) — Beta params at each position
        """
        B, Kseq, _ = states.shape

        # Encode plan
        plan_emb = self.plan_proj(plan) + self.plan_pos[:, :plan.shape[1]]
        plan_memory = self.plan_encoder(plan_emb)

        # Build decoder input: Concat(state_embed, steer_embed)
        state_emb = self.state_proj(states)      # (B, K, d/2)
        steer_emb = self.steer_proj(steers)      # (B, K, d/2)
        x = torch.cat([state_emb, steer_emb], dim=-1) + self.dec_pos[:, :Kseq]

        # Causal decode + cross-attention to plan
        mask = self.causal_mask[:Kseq, :Kseq]
        decoded = self.decoder(x, plan_memory, tgt_mask=mask, tgt_is_causal=True)

        return self.head(self.ln_f(decoded))  # (B, K, 2)

    def beta_params(self, states, steers, plan):
        out = self.forward(states, steers, plan)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════

class SteerDataset(torch.utils.data.Dataset):
    """Precomputed windows. Position i: state[t], steer[t]. Target: delta[t+1]."""

    def __init__(self, steer, current_la, target_la, roll_la, v_ego, a_ego,
                 route_indices, ctx=K, future_k=FUTURE_K):
        KK = ctx
        F = future_k
        N = len(route_indices)
        T = steer.shape[1]

        bases = list(range(1, T - KK - F, STRIDE))
        n_per = len(bases)
        total = N * n_per

        all_states = np.empty((total, KK, D_STATE), dtype=np.float32)
        all_steers = np.empty((total, KK, 1), dtype=np.float32)
        all_target = np.empty((total, KK), dtype=np.float32)  # raw delta targets
        all_plan = np.empty((total, F, D_PLAN), dtype=np.float32)

        print(f"    Precomputing {total:,} windows...", end="", flush=True)
        t0 = time.time()
        idx = 0
        for ri, r in enumerate(route_indices):
            for base in bases:
                # States: road features at steps [base, base+K)
                all_states[idx, :, 0] = target_la[r, base:base+KK] / S_LA
                all_states[idx, :, 1] = roll_la[r, base:base+KK] / S_ROLL
                all_states[idx, :, 2] = v_ego[r, base:base+KK] / S_V
                all_states[idx, :, 3] = a_ego[r, base:base+KK] / S_A
                all_states[idx, :, 4] = current_la[r, base:base+KK] / S_LA

                # Steer at steps [base, base+K)
                all_steers[idx, :, 0] = steer[r, base:base+KK] / S_STEER

                # Target: raw delta at steps [base+1, base+K+1)
                # raw = clip((steer[t+1] - steer[t]) / DELTA_SCALE, -1, 1)
                deltas = (steer[r, base+1:base+KK+1] - steer[r, base:base+KK]) / DELTA_SCALE
                all_target[idx] = np.clip(deltas, -1.0, 1.0)

                # Plan: future road from step base+K
                ps = base + KK
                pe = min(ps + F, T)
                avail = pe - ps
                all_plan[idx, :avail, 0] = target_la[r, ps:pe] / S_LA
                all_plan[idx, :avail, 1] = roll_la[r, ps:pe] / S_ROLL
                all_plan[idx, :avail, 2] = v_ego[r, ps:pe] / S_V
                all_plan[idx, :avail, 3] = a_ego[r, ps:pe] / S_A
                if avail < F:
                    all_plan[idx, avail:] = all_plan[idx, avail-1:avail]
                idx += 1
        print(f" {time.time()-t0:.1f}s", flush=True)

        self.states = all_states
        self.steers = all_steers
        self.target = all_target
        self.plan = all_plan

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.steers[idx], self.plan[idx], self.target[idx]


# ══════════════════════════════════════════════════════════════
#  Batched sim eval
# ══════════════════════════════════════════════════════════════

def evaluate(model, eval_csvs, model_path):
    from tinyphysics_batched import BatchedSimulator
    import warnings; warnings.filterwarnings("ignore")

    N = len(eval_csvs)
    sim = BatchedSimulator(model_path, csv_files=[str(f) for f in eval_csvs])
    T_max = sim.T
    dg = sim.data_gpu

    # Buffers
    steer_hist = torch.zeros((N, T_max), dtype=torch.float64, device=DEV)
    steer_hist[:, :CONTROL_START_IDX] = sim.action_history[:, :CONTROL_START_IDX]
    cur_la_hist = torch.zeros((N, T_max), dtype=torch.float64, device=DEV)
    cur_la_hist[:, :CONTROL_START_IDX] = sim.current_lataccel_history[:, :CONTROL_START_IDX]

    def controller_fn(step_idx, sim_ref):
        cur_la = sim_ref.current_lataccel
        cur_la_hist[:, step_idx] = cur_la  # track per-step noise

        action_idx = step_idx - CONTROL_START_IDX
        if action_idx < 0:
            return torch.zeros(N, dtype=torch.float64, device=DEV)

        K_use = min(action_idx + 1, K)

        # Build state with HISTORICAL per-position data
        states_t = torch.zeros((N, K_use, D_STATE), dtype=torch.float32, device=DEV)
        steers_t = torch.zeros((N, K_use, 1), dtype=torch.float32, device=DEV)
        for i in range(K_use):
            t = max(step_idx - K_use + i, 0)
            states_t[:, i, 0] = dg["target_lataccel"][:, t].float() / S_LA
            states_t[:, i, 1] = dg["roll_lataccel"][:, t].float() / S_ROLL
            states_t[:, i, 2] = dg["v_ego"][:, t].float() / S_V
            states_t[:, i, 3] = dg["a_ego"][:, t].float() / S_A
            states_t[:, i, 4] = cur_la_hist[:, t].float() / S_LA  # same step as steer
            steers_t[:, i, 0] = steer_hist[:, t].float() / S_STEER

        # Plan
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
            alpha, beta = model.beta_params(states_t, steers_t, plan_t)
            # Last position → deterministic mean
            a_last = alpha[:, -1]
            b_last = beta[:, -1]
            raw = 2.0 * a_last / (a_last + b_last) - 1.0

        delta = raw.double() * DELTA_SCALE
        prev = steer_hist[:, max(step_idx - 1, 0)]
        new_steer = (prev + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        steer_hist[:, step_idx] = new_steer
        return new_steer

    costs = sim.rollout(controller_fn)
    return costs["total_cost"].mean(), costs["total_cost"].std()


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"SteerGPT v2: d={D_MODEL} heads={N_HEADS} enc={N_ENC} dec={N_DEC} K={K}")
    print(f"  state=5(road+noise) token=steer(1) output=Beta(delta)")
    print(f"  stride={STRIDE} eval_every={EVAL_EVERY} device={DEV}")
    print("=" * 70, flush=True)

    d = np.load(DATA_PATH)
    steer = d["steer"]; current_la = d["current_la"]
    target_la = d["target_la"]; roll_la = d["roll_la"]
    v_ego_arr = d["v_ego"]; a_ego_arr = d["a_ego"]
    N, T = steer.shape
    print(f"  {N} routes, {T} steps", flush=True)

    perm = np.random.RandomState(42).permutation(N)
    n_val = max(int(N * VAL_FRAC), 10)
    val_routes, train_routes = perm[:n_val], perm[n_val:]

    train_ds = SteerDataset(steer, current_la, target_la, roll_la, v_ego_arr, a_ego_arr,
                            train_routes, ctx=K, future_k=FUTURE_K)
    val_ds = SteerDataset(steer, current_la, target_la, roll_la, v_ego_arr, a_ego_arr,
                          val_routes, ctx=K, future_k=FUTURE_K)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BS, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}", flush=True)

    model = SteerGPT().to(DEV)
    print(f"  Params: {model.count_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    eval_csvs = sorted(Path(ROOT / "data").glob("*.csv"))[:EVAL_N]
    best_val = float("inf")
    best_sim = float("inf")

    n_batches = len(train_loader)
    print(f"\n{EPOCHS} epochs, BS={BS}, {n_batches} batches/ep")
    print("-" * 80, flush=True)

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, n_samples = 0.0, 0

        for batch_i, (states, steers, plan, target) in enumerate(train_loader):
            states = states.to(DEV, non_blocking=True)
            steers = steers.to(DEV, non_blocking=True)
            plan = plan.to(DEV, non_blocking=True)
            target = target.to(DEV, non_blocking=True)

            out = model(states, steers, plan)  # (B, K, 2)
            alpha = F.softplus(out[..., 0]) + 1.0
            beta_p = F.softplus(out[..., 1]) + 1.0
            x = ((target + 1) / 2).clamp(1e-6, 1 - 1e-6)  # raw[-1,1] → Beta[0,1]
            loss = -torch.distributions.Beta(alpha, beta_p).log_prob(x).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item() * states.shape[0]
            n_samples += states.shape[0]

            if batch_i % 200 == 0 and batch_i > 0:
                print(f"    [{batch_i}/{n_batches}] loss={total_loss/n_samples:.4f}", flush=True)

        sched.step()
        train_loss = total_loss / n_samples

        # Val
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for states, steers, plan, target in val_loader:
                states = states.to(DEV, non_blocking=True)
                steers = steers.to(DEV, non_blocking=True)
                plan = plan.to(DEV, non_blocking=True)
                target = target.to(DEV, non_blocking=True)
                out = model(states, steers, plan)
                alpha = F.softplus(out[..., 0]) + 1.0
                beta_p = F.softplus(out[..., 1]) + 1.0
                x = ((target + 1) / 2).clamp(1e-6, 1 - 1e-6)
                loss = -torch.distributions.Beta(alpha, beta_p).log_prob(x).mean()
                val_sum += loss.item() * states.shape[0]
                val_n += states.shape[0]

        val_loss = val_sum / val_n
        elapsed = time.time() - t0

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "n_enc": N_ENC, "n_dec": N_DEC,
                "ctx": K, "future_k": FUTURE_K, "delta_scale": DELTA_SCALE,
            }}, CKPT_DIR / "best_model.pt")
            tag = " *BEST*"

        line = f"  Ep {ep:3d} | train {train_loss:.4f} | val {val_loss:.4f}{tag} | lr {opt.param_groups[0]['lr']:.1e} | {elapsed:.1f}s"

        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            sim_mean, sim_std = evaluate(model, eval_csvs, MODEL_PATH)
            sim_tag = ""
            if sim_mean < best_sim:
                best_sim = sim_mean
                torch.save({"model": model.state_dict(), "config": {
                    "d_model": D_MODEL, "n_heads": N_HEADS, "n_enc": N_ENC, "n_dec": N_DEC,
                    "ctx": K, "future_k": FUTURE_K, "delta_scale": DELTA_SCALE,
                }}, CKPT_DIR / "best_sim_model.pt")
                sim_tag = " ★"
            line += f"  sim={sim_mean:.1f}±{sim_std:.0f}{sim_tag}"

        print(line, flush=True)

    print("-" * 80)
    print(f"Best val: {best_val:.4f} | Best sim: {best_sim:.1f}")
    print(f"Saved to {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
