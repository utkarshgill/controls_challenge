"""BC fine-tuning: load exp055 checkpoint, train actor on MPC expert data.

Usage:
  /venv/main/bin/python3 experiments/exp110_mpc/train_bc.py

Loads bc_data.npz (2M obs/raw pairs from MPC expert), fine-tunes actor with
Beta NLL loss. Saves best model as bc_model.pt.
"""

import numpy as np, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DEV = torch.device("cuda")

# ── architecture (must match exp055) ──────────────────────────
STATE_DIM, HIDDEN = 256, 256
A_LAYERS, C_LAYERS = 4, 4
DELTA_SCALE_MAX = 0.25

# ── BC hyperparams ────────────────────────────────────────────
BC_EPOCHS = int(os.getenv("BC_EPOCHS", "30"))
BC_LR = float(os.getenv("BC_LR", "3e-3"))
BC_LR_MIN = float(os.getenv("BC_LR_MIN", "1e-5"))
BC_BS = int(os.getenv("BC_BS", "4096"))
BC_GRAD_CLIP = 2.0
VAL_FRAC = 0.02  # 2% holdout
WEIGHT_DECAY = 1e-4

EXP_DIR = Path(__file__).parent
CKPT_DIR = EXP_DIR / "checkpoints"
BC_DATA_PATH = CKPT_DIR / "bc_data.npz"
SRC_MODEL_PATH = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
OUT_MODEL_PATH = CKPT_DIR / "bc_model.pt"


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))
        self.actor = nn.Sequential(*a)

        c = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(C_LAYERS - 1):
            c += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        c.append(nn.Linear(HIDDEN, 1))
        self.critic = nn.Sequential(*c)

        for layer in self.actor[:-1]:
            _ortho(layer)
        _ortho(self.actor[-1], gain=0.01)
        for layer in self.critic[:-1]:
            _ortho(layer)
        _ortho(self.critic[-1], gain=1.0)

    def beta_params(self, obs):
        out = self.actor(obs)
        return F.softplus(out[..., 0]) + 1.0, F.softplus(out[..., 1]) + 1.0


def main():
    print("=" * 60)
    print("BC Fine-tuning from MPC Expert Data")
    print("=" * 60)

    # Load data
    print(f"\nLoading BC data from {BC_DATA_PATH} ...")
    d = np.load(BC_DATA_PATH)
    obs_np, raw_np = d["obs"], d["raw"]
    N = len(obs_np)
    print(f"  {N:,} samples, obs shape {obs_np.shape}, raw shape {raw_np.shape}")

    # Train/val split
    perm = np.random.RandomState(42).permutation(N)
    n_val = max(int(N * VAL_FRAC), 1000)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"  Train: {len(train_idx):,}, Val: {n_val:,}")

    obs_train = torch.from_numpy(obs_np[train_idx]).to(DEV)
    raw_train = torch.from_numpy(raw_np[train_idx]).to(DEV)
    obs_val = torch.from_numpy(obs_np[val_idx]).to(DEV)
    raw_val = torch.from_numpy(raw_np[val_idx]).to(DEV)
    del obs_np, raw_np, d  # free RAM

    # Convert raw to Beta target x in (0,1)
    x_train = ((raw_train + 1) / 2).clamp(1e-6, 1 - 1e-6)
    x_val = ((raw_val + 1) / 2).clamp(1e-6, 1 - 1e-6)

    # Load model
    ac = ActorCritic().to(DEV)
    if SRC_MODEL_PATH.exists():
        sd = torch.load(SRC_MODEL_PATH, map_location=DEV, weights_only=False)
        ac.load_state_dict(sd, strict=False)
        print(f"\nLoaded exp055 checkpoint from {SRC_MODEL_PATH}")
    else:
        print(
            f"\nWARNING: No exp055 checkpoint found at {SRC_MODEL_PATH}, training from scratch"
        )

    # Compute initial val loss
    ac.eval()
    with torch.no_grad():
        init_losses = []
        for i in range(0, n_val, BC_BS):
            j = min(i + BC_BS, n_val)
            out = ac.actor(obs_val[i:j])
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x_val[i:j]).mean()
            init_losses.append(loss.item() * (j - i))
        init_val_loss = sum(init_losses) / n_val
    print(f"  Initial val loss (NLL): {init_val_loss:.4f}")

    # Optimizer
    opt = optim.AdamW(ac.actor.parameters(), lr=BC_LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=BC_EPOCHS, eta_min=BC_LR_MIN
    )

    best_val = float("inf")
    best_epoch = -1
    n_train = len(train_idx)

    print(f"\nTraining for {BC_EPOCHS} epochs, BS={BC_BS}, LR={BC_LR}")
    print("-" * 60)

    for ep in range(BC_EPOCHS):
        t0 = time.time()
        ac.train()
        total_loss, n_batch = 0.0, 0

        for idx in torch.randperm(n_train, device=DEV).split(BC_BS):
            out = ac.actor(obs_train[idx])
            a_p = F.softplus(out[..., 0]) + 1.0
            b_p = F.softplus(out[..., 1]) + 1.0
            loss = -torch.distributions.Beta(a_p, b_p).log_prob(x_train[idx]).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), BC_GRAD_CLIP)
            opt.step()

            total_loss += loss.item() * idx.numel()
            n_batch += idx.numel()

        sched.step()
        train_loss = total_loss / n_batch

        # Val loss
        ac.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, n_val, BC_BS):
                j = min(i + BC_BS, n_val)
                out = ac.actor(obs_val[i:j])
                a_p = F.softplus(out[..., 0]) + 1.0
                b_p = F.softplus(out[..., 1]) + 1.0
                loss = -torch.distributions.Beta(a_p, b_p).log_prob(x_val[i:j]).mean()
                val_losses.append(loss.item() * (j - i))
            val_loss = sum(val_losses) / n_val

        # Diagnostics: check sigma
        with torch.no_grad():
            diag_out = ac.actor(obs_val[:1000])
            a_p = F.softplus(diag_out[..., 0]) + 1.0
            b_p = F.softplus(diag_out[..., 1]) + 1.0
            mean_raw = (2 * a_p / (a_p + b_p) - 1).mean().item()
            sigma_raw = (
                (2 * torch.sqrt(a_p * b_p / ((a_p + b_p) ** 2 * (a_p + b_p + 1))))
                .mean()
                .item()
            )
            sigma_eff = sigma_raw * DELTA_SCALE_MAX

        elapsed = time.time() - t0
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            torch.save(ac.state_dict(), OUT_MODEL_PATH)
            improved = " *BEST*"

        print(
            f"  Ep {ep:3d} | train {train_loss:.4f} | val {val_loss:.4f}{improved} "
            f"| mean_raw {mean_raw:+.4f} sigma_eff {sigma_eff:.4f} "
            f"| lr {opt.param_groups[0]['lr']:.1e} | {elapsed:.1f}s"
        )

    print("-" * 60)
    print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Saved to {OUT_MODEL_PATH}")
    print(f"\nInitial val loss: {init_val_loss:.4f} -> Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
