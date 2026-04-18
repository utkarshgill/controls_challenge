"""Test: does feeding denoised lataccel to the existing MLP policy help?

Three evaluations:
  A) Policy with normal noisy observations (baseline, should be ~36)
  B) Policy with denoised current_lataccel (pretrained transformer denoiser)
  C) Oracle: policy with temp=0.1 current_lataccel (impossible in practice)

If B < A: denoiser helps even without retraining the policy
If B > A: policy was trained on noisy obs, denoised input confuses it
If B ≈ C: denoiser perfectly recovers the clean signal
"""

import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

os.environ.setdefault("CUDA", "1")
os.environ.setdefault("TRT", "1")

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

# Import the EXISTING MLP policy from exp055
from experiments.exp055_batch_of_batch.train import (
    ActorCritic as MLPActorCritic,
    _precompute_future_windows,
    fill_obs,
    HIST_LEN,
    OBS_DIM,
    STEER_RANGE,
    DELTA_SCALE_MAX,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── Denoiser architecture (same as test_denoise2) ──
SEQ_LEN = 20
TOKEN_DIM = 8
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MLP_RATIO = 2

S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL = 2.0


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


class LataccelDenoiser(nn.Module):
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


def run_policy_sim(csv_files, ac, mdl_path, ort_sess, csv_cache, ds, sim_temp):
    """Run the MLP policy at a given temperature. Returns (la_history, act_history, dg)."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    sim.sim_model.sim_temperature = sim_temp
    N = sim.N
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel
        cur32 = current.float()
        error = (target - current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_err = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_err
        ei = err_sum * (DEL_T / HIST_LEN)
        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")
        fill_obs(
            obs_buf,
            target.float(),
            cur32,
            dg["roll_lataccel"][:, step_idx].float(),
            dg["v_ego"][:, step_idx].float(),
            dg["a_ego"][:, step_idx].float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )
        with torch.inference_mode():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])
        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action

    sim.rollout(ctrl)
    return sim.current_lataccel_history.float(), sim.action_history.float(), dg


def train_denoiser(csv_files, ac, mdl_path, ort_sess, csv_cache, ds):
    """Train denoiser on POLICY-generated trajectories at temp=0.1 and 0.8."""
    import torch.optim as optim

    N_ROUTES = len(csv_files)
    print(f"  Training denoiser on {N_ROUTES} routes with POLICY trajectories...")

    # Run policy at both temperatures
    print(f"    Running policy @ temp=0.8...")
    la_noisy, act_noisy, dg = run_policy_sim(
        csv_files, ac, mdl_path, ort_sess, csv_cache, ds, 0.8
    )
    print(f"    Running policy @ temp=0.1...")
    la_clean, act_clean, _ = run_policy_sim(
        csv_files, ac, mdl_path, ort_sess, csv_cache, ds, 0.1
    )

    N = la_noisy.shape[0]

    # Build sequences from noisy policy trajectory, labels from clean
    start, end = CONTROL_START_IDX, COST_END_IDX
    all_seq, all_label = [], []
    for t in range(start, end):
        s = max(t - SEQ_LEN, 0)
        L = t - s
        pad = SEQ_LEN - L
        r = torch.arange(s, t, device="cuda")
        tgts = dg["target_lataccel"][:, r].float()
        curs = la_noisy[:, r]
        acts = act_noisy[:, r]
        rolls = dg["roll_lataccel"][:, r].float()
        vs = dg["v_ego"][:, r].float()
        as_ = dg["a_ego"][:, r].float()
        errs = tgts - curs
        prev = (
            torch.cat([torch.zeros(N, 1, device="cuda"), curs[:, :-1]], dim=1)
            if s == 0
            else la_noisy[:, s - 1 : t - 1]
        )
        jerks = (curs - prev) / DEL_T
        tok = torch.stack(
            [
                tgts / S_LAT,
                curs / S_LAT,
                errs / S_LAT,
                acts / S_STEER,
                rolls / S_ROLL,
                vs / S_VEGO,
                as_ / S_AEGO,
                jerks / S_LAT,
            ],
            dim=-1,
        ).clamp(-5, 5)
        if pad > 0:
            tok = F.pad(tok, (0, 0, pad, 0))
        all_seq.append(tok)
        all_label.append(la_clean[:, t])

    seq_t = torch.cat(all_seq)
    label_t = torch.cat(all_label)
    noise_mse = F.mse_loss(
        torch.cat([la_noisy[:, t : t + 1] for t in range(start, end)], dim=1).reshape(
            -1
        ),
        label_t,
    ).item()
    print(f"  {len(seq_t)} samples, noise MSE={noise_mse:.4f}, training...")

    model = LataccelDenoiser().to(DEV)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for ep in range(30):
        model.train()
        total, nb = 0.0, 0
        for idx in torch.randperm(len(seq_t), device="cuda").split(2048):
            pred = model(seq_t[idx]) * S_LAT
            loss = F.mse_loss(pred, label_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            nb += len(idx)
        sched.step()
        if ep % 10 == 0 or ep == 29:
            print(f"    E{ep:2d} loss={total / nb:.6f}")

    model.eval()
    return model


def eval_policy(
    csv_files, ac, mdl_path, ort_sess, csv_cache, ds, denoiser=None, mode="normal"
):
    """Run the MLP policy. mode: 'normal', 'denoised', or 'clean_oracle'."""
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_sess, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")
    hist_head = HIST_LEN - 1

    # For denoiser: maintain token ring buffer
    seq_buf = torch.zeros((N, SEQ_LEN, TOKEN_DIM), dtype=torch.float32, device="cuda")
    seq_head = SEQ_LEN - 1
    prev_lataccel = torch.zeros(N, dtype=torch.float32, device="cuda")

    # For oracle: run a parallel clean sim
    if mode == "clean_oracle":
        data2, rng2 = csv_cache.slice(csv_files)
        sim_clean = BatchedSimulator(
            str(mdl_path), ort_session=ort_sess, cached_data=data2, cached_rng=rng2
        )
        sim_clean.sim_model.sim_temperature = 0.1
        # We need to step the clean sim in lockstep
        clean_la_history = []

    def ctrl(step_idx, sim_ref):
        nonlocal hist_head, err_sum, seq_head, prev_lataccel
        target = dg["target_lataccel"][:, step_idx]
        current = sim_ref.current_lataccel  # the NOISY lataccel (sim runs at 0.8)
        cur32 = current.float()

        # Build denoiser token
        if denoiser is not None or mode != "normal":
            next_sh = (seq_head + 1) % SEQ_LEN
            error_f = target.float() - cur32
            jerk_f = (cur32 - prev_lataccel) / DEL_T
            seq_buf[:, next_sh] = torch.stack(
                [
                    target.float() / S_LAT,
                    cur32 / S_LAT,
                    error_f / S_LAT,
                    h_act32[:, hist_head] / S_STEER,
                    dg["roll_lataccel"][:, step_idx].float() / S_ROLL,
                    dg["v_ego"][:, step_idx].float() / S_VEGO,
                    dg["a_ego"][:, step_idx].float() / S_AEGO,
                    jerk_f / S_LAT,
                ],
                dim=-1,
            ).clamp(-5, 5)
            seq_head = next_sh
            prev_lataccel = cur32

        # Determine what "current" to feed to the policy
        if mode == "denoised" and denoiser is not None:
            split = seq_head + 1
            if split >= SEQ_LEN:
                seq_ordered = seq_buf
            else:
                seq_ordered = torch.cat([seq_buf[:, split:], seq_buf[:, :split]], dim=1)
            with torch.no_grad():
                denoised = denoiser(seq_ordered) * S_LAT
            obs_current = denoised.to(current.dtype)
        elif mode == "clean_oracle":
            # This is approximate — we can't actually step a clean sim in lockstep
            # because the clean sim would need different actions. So this mode
            # isn't implementable cleanly. Skip it.
            obs_current = current
        else:
            obs_current = current

        obs_cur32 = obs_current.float()

        error = (target - obs_current).float()
        next_head = (hist_head + 1) % HIST_LEN
        old_err = h_error[:, next_head]
        h_error[:, next_head] = error
        err_sum = err_sum + error - old_err
        ei = err_sum * (DEL_T / HIST_LEN)

        if step_idx < CONTROL_START_IDX:
            h_act[:, next_head] = 0.0
            h_act32[:, next_head] = 0.0
            h_lat[:, next_head] = obs_cur32
            hist_head = next_head
            return torch.zeros(N, dtype=h_act.dtype, device="cuda")

        fill_obs(
            obs_buf,
            target.float(),
            obs_cur32,
            dg["roll_lataccel"][:, step_idx].float(),
            dg["v_ego"][:, step_idx].float(),
            dg["a_ego"][:, step_idx].float(),
            h_act32,
            h_lat,
            hist_head,
            ei,
            future,
            step_idx,
        )

        with torch.inference_mode():
            logits = ac.actor(obs_buf)
        a_p = F.softplus(logits[..., 0]) + 1.0
        b_p = F.softplus(logits[..., 1]) + 1.0
        raw = 2.0 * a_p / (a_p + b_p) - 1.0
        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = obs_cur32
        hist_head = next_head
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    return costs


def main():
    N_ROUTES = int(os.getenv("N_ROUTES", "100"))
    N_TRAIN = int(os.getenv("N_TRAIN", "500"))

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    eval_csv = all_csv[:N_ROUTES]
    train_csv = all_csv[N_ROUTES : N_ROUTES + N_TRAIN]
    csv_cache = CSVCache([str(f) for f in all_csv[: N_ROUTES + N_TRAIN]])

    # Load MLP policy
    ac = MLPActorCritic().to(DEV)
    ckpt = torch.load(
        ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt",
        weights_only=False,
        map_location=DEV,
    )
    ac.load_state_dict(ckpt["ac"])
    ac.eval()
    ds = float(ckpt.get("delta_scale", 0.25))

    # A) Normal evaluation
    print("A) Policy with noisy observations...")
    costs_A = eval_policy(
        eval_csv, ac, mdl_path, ort_sess, csv_cache, ds, mode="normal"
    )
    print(f"   mean={np.mean(costs_A):.1f}")

    # Train denoiser on separate routes using POLICY trajectories
    denoiser = train_denoiser(train_csv, ac, mdl_path, ort_sess, csv_cache, ds)

    # B) Denoised evaluation
    print("B) Policy with denoised observations...")
    costs_B = eval_policy(
        eval_csv,
        ac,
        mdl_path,
        ort_sess,
        csv_cache,
        ds,
        denoiser=denoiser,
        mode="denoised",
    )
    print(f"   mean={np.mean(costs_B):.1f}")

    print(f"\n  A (normal):   {np.mean(costs_A):.1f}")
    print(f"  B (denoised): {np.mean(costs_B):.1f}")
    delta = np.mean(costs_B) - np.mean(costs_A)
    print(f"  Δ = {delta:+.1f}  ({'better' if delta < 0 else 'worse'})")


if __name__ == "__main__":
    main()
