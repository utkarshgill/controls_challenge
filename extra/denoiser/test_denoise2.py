"""Can a transformer recover temp=0.1 observations from temp=0.8 noisy sim?

Run two sims on the same routes with zero actions:
  - sim_clean: temp=0.1 (nearly deterministic)
  - sim_noisy: temp=0.8 (stochastic)

Train a transformer on the noisy history to predict the clean lataccel.
"""

import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

SEQ_LEN = 20
TOKEN_DIM = 8
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MLP_RATIO = 2

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL = 2.0

N_ROUTES = int(os.getenv("N_ROUTES", "500"))
N_EPOCHS = int(os.getenv("N_EPOCHS", "50"))
BATCH_SIZE = 2048
LR = 3e-4


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * MLP_RATIO),
            nn.GELU(),
            nn.Linear(d_model * MLP_RATIO, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LataccelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TOKEN_DIM, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, tokens):
        x = self.proj(tokens) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, -1, :])).squeeze(-1)


def run_sim(csv_files, mdl_path, ort_sess, csv_cache, temperature):
    """Run sim with zero actions at given temperature. Returns lataccel history."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    sim.sim_model.sim_temperature = temperature
    N = sim.N

    def ctrl(step_idx, sim_ref):
        return torch.zeros(N, dtype=torch.float64, device="cuda")

    sim.rollout(ctrl)
    return (
        sim.current_lataccel_history.float(),
        sim.action_history.float(),
        sim.data_gpu,
    )


def build_sequences(la_hist, act_hist, dg, start, end):
    """Build (N*steps, SEQ_LEN, TOKEN_DIM) token sequences from histories."""
    N = la_hist.shape[0]
    all_seq = []
    all_targets = []

    for t in range(start, end):
        seq_start = max(t - SEQ_LEN, 0)
        L = t - seq_start
        pad = SEQ_LEN - L

        r = torch.arange(seq_start, t, device="cuda")
        targets_la = dg["target_lataccel"][:, r].float()
        currents = la_hist[:, r]
        actions = act_hist[:, r]
        rolls = dg["roll_lataccel"][:, r].float()
        vegos = dg["v_ego"][:, r].float()
        aegos = dg["a_ego"][:, r].float()
        errors = targets_la - currents

        if seq_start > 0:
            prev = la_hist[:, seq_start - 1 : t - 1]
        else:
            prev = torch.cat(
                [torch.zeros(N, 1, device="cuda"), currents[:, :-1]], dim=1
            )
        jerks = (currents - prev) / DEL_T

        tok = torch.stack(
            [
                targets_la / S_LAT,
                currents / S_LAT,
                errors / S_LAT,
                actions / S_STEER,
                rolls / S_ROLL,
                vegos / S_VEGO,
                aegos / S_AEGO,
                jerks / S_LAT,
            ],
            dim=-1,
        ).clamp(-5.0, 5.0)

        if pad > 0:
            tok = F.pad(tok, (0, 0, pad, 0))

        all_seq.append(tok)
        all_targets.append(la_hist[:, t])

    return torch.cat(all_seq, dim=0), torch.cat(all_targets, dim=0)


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    print(f"Running sims on {N_ROUTES} routes...")
    t0 = time.time()
    la_clean, act_clean, dg_clean = run_sim(all_csv, mdl_path, ort_sess, csv_cache, 0.1)
    la_noisy, act_noisy, dg_noisy = run_sim(all_csv, mdl_path, ort_sess, csv_cache, 0.8)
    print(f"  Done in {time.time() - t0:.1f}s")

    start, end = CONTROL_START_IDX, COST_END_IDX

    # Build sequences from noisy sim, with clean lataccel as labels
    print("Building sequences...")
    noisy_seq, noisy_la = build_sequences(la_noisy, act_noisy, dg_noisy, start, end)
    _, clean_la = build_sequences(la_clean, act_clean, dg_clean, start, end)
    print(f"  {len(noisy_seq)} samples")

    # Baselines
    noise_mse = F.mse_loss(noisy_la, clean_la).item()
    noise_mae = (noisy_la - clean_la).abs().mean().item()
    noise_corr = torch.corrcoef(torch.stack([noisy_la, clean_la]))[0, 1].item()
    print(f"\n  Noise gap (noisy vs clean lataccel):")
    print(f"    MSE={noise_mse:.6f}  MAE={noise_mae:.4f}  corr={noise_corr:.4f}")

    persist_mse = F.mse_loss(noisy_seq[:, -1, 1] * S_LAT, clean_la).item()
    print(f"  Persistence (last noisy → clean): MSE={persist_mse:.6f}")

    # Train/val split
    n = len(noisy_seq)
    perm = torch.randperm(n)
    n_val = n // 10
    tr_seq, tr_label = noisy_seq[perm[n_val:]], clean_la[perm[n_val:]]
    va_seq, va_label = noisy_seq[perm[:n_val]], clean_la[perm[:n_val]]
    va_noisy = noisy_la[perm[:n_val]]

    model = LataccelPredictor().to(DEV)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    print(f"\nTraining (predict clean from noisy)...")
    for ep in range(N_EPOCHS):
        model.train()
        total, nb = 0.0, 0
        for idx in torch.randperm(len(tr_seq), device="cuda").split(BATCH_SIZE):
            pred = model(tr_seq[idx]) * S_LAT
            loss = F.mse_loss(pred, tr_label[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            nb += len(idx)
        sched.step()

        if ep % 5 == 0 or ep == N_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vp = torch.cat(
                    [
                        model(va_seq[i : i + BATCH_SIZE]) * S_LAT
                        for i in range(0, len(va_seq), BATCH_SIZE)
                    ]
                )
                mse_clean = F.mse_loss(vp, va_label).item()
                mae_clean = (vp - va_label).abs().mean().item()
                mse_noisy = F.mse_loss(va_noisy, va_label).item()
            print(
                f"  E{ep:3d}  train={total / nb:.6f}  "
                f"val_mse={mse_clean:.6f}  val_mae={mae_clean:.4f}  "
                f"(noisy baseline={mse_noisy:.6f})"
            )

    print(f"\n  Summary:")
    print(f"    Raw noisy vs clean:         MSE={noise_mse:.6f}  MAE={noise_mae:.4f}")
    print(f"    Transformer vs clean:       MSE={mse_clean:.6f}  MAE={mae_clean:.4f}")
    imp = (1 - mse_clean / noise_mse) * 100 if noise_mse > 0 else 0
    print(f"    Improvement: {imp:.1f}% MSE reduction")
    if imp > 5:
        print(f"    → YES, transformer can recover clean signal from noisy")
    else:
        print(f"    → NO, clean signal is not recoverable from noisy history")


if __name__ == "__main__":
    main()
