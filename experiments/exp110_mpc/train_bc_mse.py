"""BC with MSE loss: direct regression of raw delta from MPC expert data.

Usage:
  /venv/main/bin/python3 experiments/exp110_mpc/train_bc_mse.py

Outputs a single tanh-squashed scalar instead of Beta distribution params.
This avoids the mode-covering behavior of Beta NLL which adds noise.
"""

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DEV = torch.device("cuda")

# ── architecture ──────────────────────────────────────────────
STATE_DIM = 256
HIDDEN = int(os.getenv("HIDDEN", "256"))
N_LAYERS = int(os.getenv("N_LAYERS", "4"))

# ── BC hyperparams ────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "50"))
BC_LR = float(os.getenv("BC_LR", "3e-3"))
BC_LR_MIN = float(os.getenv("BC_LR_MIN", "1e-5"))
BC_BS = int(os.getenv("BC_BS", "4096"))
BC_GRAD_CLIP = 2.0
VAL_FRAC = 0.02
WEIGHT_DECAY = float(os.getenv("WD", "1e-4"))
LOSS_TYPE = os.getenv("LOSS", "huber")  # 'mse' or 'huber'

EXP_DIR = Path(__file__).parent
CKPT_DIR = EXP_DIR / "checkpoints"
BC_DATA_PATH = CKPT_DIR / "bc_data.npz"
OUT_MODEL_PATH = CKPT_DIR / "bc_mse_model.pt"


class MLPActor(nn.Module):
    """Simple MLP: obs -> single scalar (raw delta in [-1,1])."""

    def __init__(self, input_dim=STATE_DIM, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        # Initialize
        for m in self.net[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Last layer small init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


def main():
    print("=" * 60)
    print(f"BC MSE Training (loss={LOSS_TYPE}, hidden={HIDDEN}, layers={N_LAYERS})")
    print("=" * 60)

    # Load data
    print(f"\nLoading BC data from {BC_DATA_PATH} ...")
    d = np.load(BC_DATA_PATH)
    obs_np, raw_np = d["obs"], d["raw"]
    N = len(obs_np)
    print(f"  {N:,} samples")

    # Stats on raw targets
    print(
        f"  raw stats: mean={raw_np.mean():.4f} std={raw_np.std():.4f} "
        f"min={raw_np.min():.4f} max={raw_np.max():.4f}"
    )
    print(f"  |raw| > 0.9: {(np.abs(raw_np) > 0.9).mean() * 100:.1f}%")

    # Train/val split
    perm = np.random.RandomState(42).permutation(N)
    n_val = max(int(N * VAL_FRAC), 1000)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"  Train: {len(train_idx):,}, Val: {n_val:,}")

    obs_train = torch.from_numpy(obs_np[train_idx]).to(DEV)
    raw_train = torch.from_numpy(raw_np[train_idx]).to(DEV)
    obs_val = torch.from_numpy(obs_np[val_idx]).to(DEV)
    raw_val = torch.from_numpy(raw_np[val_idx]).to(DEV)
    del obs_np, raw_np, d

    # Model
    model = MLPActor().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {n_params:,}")

    # Loss
    if LOSS_TYPE == "huber":
        loss_fn = nn.HuberLoss(delta=0.1)
    else:
        loss_fn = nn.MSELoss()

    # Optimizer
    opt = optim.AdamW(model.parameters(), lr=BC_LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=BC_EPOCHS, eta_min=BC_LR_MIN
    )

    # Initial val loss
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i in range(0, n_val, BC_BS):
            j = min(i + BC_BS, n_val)
            pred = model(obs_val[i:j])
            loss = loss_fn(pred, raw_val[i:j])
            val_losses.append(loss.item() * (j - i))
        init_val_loss = sum(val_losses) / n_val
    print(f"  Initial val loss: {init_val_loss:.6f}")

    best_val = float("inf")
    best_epoch = -1
    n_train = len(train_idx)

    print(f"\nTraining for {BC_EPOCHS} epochs, BS={BC_BS}, LR={BC_LR}")
    print("-" * 70)

    for ep in range(BC_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, n_batch = 0.0, 0

        for idx in torch.randperm(n_train, device=DEV).split(BC_BS):
            pred = model(obs_train[idx])
            loss = loss_fn(pred, raw_train[idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), BC_GRAD_CLIP)
            opt.step()

            total_loss += loss.item() * idx.numel()
            n_batch += idx.numel()

        sched.step()
        train_loss = total_loss / n_batch

        # Val
        model.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, n_val, BC_BS):
                j = min(i + BC_BS, n_val)
                pred = model(obs_val[i:j])
                loss = loss_fn(pred, raw_val[i:j])
                val_losses.append(loss.item() * (j - i))
            val_loss = sum(val_losses) / n_val

            # Diagnostics
            pred_all = model(obs_val[:2000])
            mean_pred = pred_all.mean().item()
            std_pred = pred_all.std().item()
            mae = (pred_all - raw_val[:2000]).abs().mean().item()

        elapsed = time.time() - t0
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            torch.save(model.state_dict(), OUT_MODEL_PATH)
            improved = " *BEST*"

        print(
            f"  Ep {ep:3d} | train {train_loss:.6f} | val {val_loss:.6f}{improved} "
            f"| mae {mae:.4f} mean {mean_pred:+.4f} std {std_pred:.4f} "
            f"| lr {opt.param_groups[0]['lr']:.1e} | {elapsed:.1f}s"
        )

    print("-" * 70)
    print(f"Best val loss: {best_val:.6f} at epoch {best_epoch}")
    print(f"Saved to {OUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
