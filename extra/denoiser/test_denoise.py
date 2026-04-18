"""Can a transformer learn to denoise lataccel history?

Uses BatchedSimulator with compute_expected=True. Stores expected values
during rollout via a simple wrapper. Then trains a transformer to predict
expected from the noisy sampled history.
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

# ── Transformer ──


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


# ── Data collection ──


def collect_data(csv_files, mdl_path, ort_sess, csv_cache):
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    sim.compute_expected = True
    N = sim.N
    dg = sim.data_gpu
    n_ctrl = COST_END_IDX - CONTROL_START_IDX

    # Store expected values as they're computed
    expected_buf = torch.zeros((N, n_ctrl), dtype=torch.float32, device="cuda")
    ctrl_si = [0]

    def ctrl(step_idx, sim_ref):
        return torch.zeros(N, dtype=torch.float64, device="cuda")

    sim.rollout(ctrl)

    # After rollout, sim has full histories:
    #   current_lataccel_history[:, t] for t in 0..T-1
    #   These are indexed directly by step_idx (0-based)
    # The sampled lataccel at control step t is:
    #   sim.current_lataccel_history[:, t]
    sampled_hist = sim.current_lataccel_history.float()  # (N, T)
    action_hist = sim.action_history.float()  # (N, T)

    # Sanity
    print(
        f"  sampled_hist shape={sampled_hist.shape}  action_hist shape={action_hist.shape}"
    )
    print(f"  state_hist shape={sim.state_history.shape}")
    print(f"  action_hist nan count: {action_hist.isnan().sum().item()}")
    print(f"  state_hist nan count: {sim.state_history.float().isnan().sum().item()}")
    print(f"  sampled_hist nan count: {sampled_hist.isnan().sum().item()}")
    # Find first NaN in action_hist
    nan_mask = action_hist.isnan()
    if nan_mask.any():
        first_nan = nan_mask.nonzero()[0]
        print(
            f"  first NaN in action_hist at route={first_nan[0].item()}, idx={first_nan[1].item()}"
        )
        # Also check what _hist_len ended at
        print(f"  sim._hist_len={sim._hist_len}")
    s_start = CONTROL_START_IDX
    s_end = COST_END_IDX
    sampled_ctrl = sampled_hist[:, s_start:s_end]
    print(f"  sampled ctrl range=[{sampled_ctrl.min():.3f}, {sampled_ctrl.max():.3f}]")

    # Recompute expected values using the EXACT same contexts the sim used.
    # The sim at internal hist index h used:
    #   states = state_history[:, h-CL+1:h+1, :]  (CL entries)
    #   actions = action_history[:, h-CL+1:h+1]    (CL entries)
    #   preds = current_lataccel_history[:, h-CL:h] (CL entries)
    from tinyphysics_batched import LATACCEL_RANGE, VOCAB_SIZE

    BINS = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=DEV)
    phys = sim.sim_model
    CL = CONTEXT_LENGTH
    state_hist = sim.state_history.float()  # (N, T, 3)

    expected_ctrl = torch.zeros((N, n_ctrl), dtype=torch.float32, device="cuda")
    print(f"  Recomputing expected values using sim histories...")
    for ci in range(n_ctrl):
        t = CONTROL_START_IDX + ci
        h = CL + t  # internal hist index = CL + step_idx

        # Use the exact same slices the sim used in sim_step
        sim_states = state_hist[:, h - CL + 1 : h + 1, :]  # (N, CL, 3)
        sim_actions = action_hist[:, h - CL + 1 : h + 1]  # (N, CL)
        sim_preds = sampled_hist[:, h - CL : h]  # (N, CL)

        # Build model input: states = [actions, sim_states]
        states_input = torch.zeros((N, CL, 4), dtype=torch.float32, device="cuda")
        states_input[:, :, 0] = sim_actions
        states_input[:, :, 1:] = sim_states

        clamped = sim_preds.clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1]).float()
        tokens = (
            torch.bucketize(clamped, BINS, right=False).clamp(0, VOCAB_SIZE - 1).long()
        )

        if ci == 0:
            print(
                f"    h={h} states_nan={states_input.isnan().any()} "
                f"actions_nan={sim_actions.isnan().any()} "
                f"preds_nan={sim_preds.isnan().any()}"
            )

        phys._predict_gpu(
            {"states": states_input, "tokens": tokens}, temperature=0.8, rng_u=None
        )
        probs = phys._last_probs_gpu
        expected_ctrl[:, ci] = (probs * BINS.float().unsqueeze(0)).sum(dim=-1)

        if ci == 0:
            print(
                f"    probs_nan={probs.isnan().any()} exp[0]={expected_ctrl[0, 0].item():.4f}"
            )

    print(
        f"  expected ctrl range=[{expected_ctrl.min():.3f}, {expected_ctrl.max():.3f}]"
    )
    print(
        f"  any NaN? sampled={sampled_ctrl.isnan().any()}  expected={expected_ctrl.isnan().any()}"
    )

    # Build token sequences
    all_seq = []
    all_sampled = []
    all_expected = []

    for ci in range(n_ctrl):
        t = CONTROL_START_IDX + ci
        # History window of SEQ_LEN steps ending at t (exclusive — these are the inputs)
        seq_end = t
        seq_start = max(seq_end - SEQ_LEN, 0)
        L = seq_end - seq_start
        pad = SEQ_LEN - L

        r = torch.arange(seq_start, seq_end, device="cuda")
        targets = dg["target_lataccel"][:, r].float()
        currents = sampled_hist[:, r]
        actions = action_hist[:, r]
        rolls = dg["roll_lataccel"][:, r].float()
        vegos = dg["v_ego"][:, r].float()
        aegos = dg["a_ego"][:, r].float()
        errors = targets - currents

        # Jerk
        if seq_start > 0:
            prev = sampled_hist[:, seq_start - 1 : seq_end - 1]
        else:
            prev = torch.cat(
                [
                    torch.zeros(N, 1, device="cuda", dtype=sampled_hist.dtype),
                    currents[:, :-1],
                ],
                dim=1,
            )
        jerks = (currents - prev) / DEL_T

        tok = torch.stack(
            [
                targets / S_LAT,
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
        all_sampled.append(sampled_ctrl[:, ci])
        all_expected.append(expected_ctrl[:, ci])

    seq_t = torch.cat(all_seq, dim=0)
    sampled_t = torch.cat(all_sampled, dim=0)
    expected_t = torch.cat(all_expected, dim=0)
    return seq_t, sampled_t, expected_t


def main():
    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))[:N_ROUTES]
    csv_cache = CSVCache([str(f) for f in all_csv])

    print(f"Collecting data from {N_ROUTES} routes...")
    t0 = time.time()
    seq_t, sampled_t, expected_t = collect_data(all_csv, mdl_path, ort_sess, csv_cache)
    print(f"  {len(seq_t)} samples in {time.time() - t0:.1f}s")

    baseline_mse = F.mse_loss(sampled_t, expected_t).item()
    baseline_mae = (sampled_t - expected_t).abs().mean().item()
    print(
        f"\n  Noise magnitude (sampled vs expected): MSE={baseline_mse:.6f}  MAE={baseline_mae:.4f}"
    )

    # Train/val split
    n = len(seq_t)
    perm = torch.randperm(n)
    n_val = n // 10
    train_seq, train_exp = seq_t[perm[n_val:]], expected_t[perm[n_val:]]
    val_seq, val_exp = seq_t[perm[:n_val]], expected_t[perm[:n_val]]
    val_sampled = sampled_t[perm[:n_val]]
    print(f"  Train: {len(train_seq)}  Val: {len(val_seq)}")

    model = LataccelPredictor().to(DEV)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    print(f"\nTraining...")
    for ep in range(N_EPOCHS):
        model.train()
        total, nb = 0.0, 0
        for idx in torch.randperm(len(train_seq), device="cuda").split(BATCH_SIZE):
            pred = model(train_seq[idx]) * S_LAT
            loss = F.mse_loss(pred, train_exp[idx])
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
                        model(val_seq[i : i + BATCH_SIZE]) * S_LAT
                        for i in range(0, len(val_seq), BATCH_SIZE)
                    ]
                )
                mse_exp = F.mse_loss(vp, val_exp).item()
                mae_exp = (vp - val_exp).abs().mean().item()
            print(
                f"  E{ep:3d}  train={total / nb:.6f}  val_mse={mse_exp:.6f}  val_mae={mae_exp:.4f}"
            )

    print(
        f"\n  Noise (sampled vs expected):        MSE={baseline_mse:.6f}  MAE={baseline_mae:.4f}"
    )
    print(
        f"  Transformer (predict expected):      MSE={mse_exp:.6f}  MAE={mae_exp:.4f}"
    )
    imp = (1 - mse_exp / baseline_mse) * 100 if baseline_mse > 0 else 0
    print(f"  Improvement: {imp:.1f}% MSE reduction")
    print(
        f"  → {'YES' if imp > 5 else 'NO'}, transformer {'can' if imp > 5 else 'cannot'} learn noise structure"
    )


if __name__ == "__main__":
    main()
