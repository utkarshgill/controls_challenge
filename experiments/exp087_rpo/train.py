# exp087 — RPO: Reparameterization Proximal Policy Optimization
#
# Combines PPO's stable sample reuse with differentiable physics gradients.
#
# Each iteration:
#   1. Collect trajectories via real batched sim (TRT, stochastic)
#   2. For each trajectory window of H steps, compute action-gradients
#      via BPTT through the differentiable physics model (expected-value)
#   3. Cache the action-gradients
#   4. M epochs of policy updates:
#      - Regenerate noise (inverse reparameterization)
#      - Backprop cached action-gradients through policy
#      - Weight by importance ratio, clip, add KL + entropy
#      - Update policy
#   5. Update value function via TD-lambda
#
# The policy and obs are identical to exp055. Only the training algorithm changes.

import numpy as np, os, sys, time, random
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tinyphysics import (
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    ACC_G,
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
)
from tinyphysics_batched import BatchedSimulator, CSVCache, make_ort_session

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEV = torch.device("cuda")

# ── architecture (match exp055) ───────────────────────────────
HIST_LEN, FUTURE_K = 20, 50
STATE_DIM, HIDDEN = 256, 256
A_LAYERS, C_LAYERS = 4, 4
DELTA_SCALE = float(os.getenv("DELTA_SCALE", "0.25"))
S_LAT, S_STEER = 5.0, 2.0
S_VEGO, S_AEGO = 40.0, 4.0
S_ROLL, S_CURV = 2.0, 0.02
C, H1, H2 = 16, 36, 56
F_LAT, F_ROLL, F_V, F_A = 56, 106, 156, 206
OBS_DIM = 256
TEMPERATURE = 0.8
BINS_GPU = torch.from_numpy(
    np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE).astype(np.float32)
).to(DEV)

# ── RPO config ────────────────────────────────────────────────
RPO_H = int(os.getenv("RPO_H", "16"))  # BPTT horizon (steps)
RPO_M = int(os.getenv("RPO_M", "5"))  # reuse epochs
PI_LR = float(os.getenv("PI_LR", "3e-4"))
VF_LR = float(os.getenv("VF_LR", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.95"))
LAMDA = float(os.getenv("LAMDA", "0.9"))
C_LOW = float(os.getenv("C_LOW", "0.8"))  # clipping: 1 - c_low
C_HIGH = float(os.getenv("C_HIGH", "1.0"))  # clipping: 1 + c_high
LAMBDA_KL = float(os.getenv("LAMBDA_KL", "0.3"))
LAMBDA_ENT = float(os.getenv("LAMBDA_ENT", "0.003"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "0.5"))
VF_EPOCHS = int(os.getenv("VF_EPOCHS", "4"))
MINI_BS = int(os.getenv("MINI_BS", "25000"))

# ── runtime ───────────────────────────────────────────────────
CSVS_EPOCH = int(os.getenv("CSVS", "5000"))
SAMPLES_PER_ROUTE = int(os.getenv("SAMPLES_PER_ROUTE", "10"))
MAX_EP = int(os.getenv("EPOCHS", "5000"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "5"))
EVAL_N = 100
RESUME = int(os.getenv("RESUME", "0"))

EXP_DIR = Path(__file__).parent
BEST_PT = EXP_DIR / "best_model.pt"
BASE_PT = os.getenv(
    "BASE_MODEL", str(ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt")
)


# ══════════════════════════════════════════════════════════════
#  Policy (Gaussian instead of Beta — RPO needs reparameterization inverse)
# ══════════════════════════════════════════════════════════════


def _ortho(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor outputs mean and log_std for Gaussian policy
        a = [nn.Linear(STATE_DIM, HIDDEN), nn.ReLU()]
        for _ in range(A_LAYERS - 1):
            a += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        a.append(nn.Linear(HIDDEN, 2))  # [mean_raw, log_std_raw]
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

    def forward_actor(self, obs):
        """Returns mean and std for Gaussian policy in raw delta space [-1, 1]."""
        out = self.actor(obs)
        mean = torch.tanh(out[..., 0])  # mean in [-1, 1]
        log_std = out[..., 1].clamp(-5, 0)  # std in [0.007, 1.0]
        std = log_std.exp()
        return mean, std

    def get_delta(self, obs):
        """Deterministic mean action."""
        mean, _ = self.forward_actor(obs)
        return mean

    def sample_action(self, obs):
        """Sample action, return (raw, logp, mean, std)."""
        mean, std = self.forward_actor(obs)
        eps = torch.randn_like(mean)
        raw = (mean + std * eps).clamp(-1, 1)
        # Log prob of Gaussian (before clamp — approximate)
        logp = (
            -0.5 * ((raw - mean) / std.clamp_min(1e-6)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        )
        return raw, logp, mean, std


# ══════════════════════════════════════════════════════════════
#  Differentiable physics (onnx2torch) for BPTT
# ══════════════════════════════════════════════════════════════


class DiffPhysics(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        import onnx2torch

        self.model = onnx2torch.convert(str(onnx_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer("bins", BINS_GPU)

    def expected_lataccel(self, states, tokens):
        """(B,20,4), (B,20) long -> (B,) expected lataccel, differentiable w.r.t. states."""
        logits = self.model(states, tokens)
        probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        return (probs * self.bins.unsqueeze(0)).sum(dim=-1)

    def tokenize(self, lataccel):
        clamped = lataccel.float().clamp(LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(clamped, self.bins, right=False).clamp(0, VOCAB_SIZE - 1)


# ══════════════════════════════════════════════════════════════
#  Obs builder (identical to exp055)
# ══════════════════════════════════════════════════════════════


def _precompute_future_windows(dg):
    def _w(x):
        x = x.float()
        shifted = torch.cat([x[:, 1:], x[:, -1:].expand(-1, FUTURE_K)], dim=1)
        return shifted.unfold(1, FUTURE_K, 1).contiguous()

    return {
        k: _w(dg[k]) for k in ("target_lataccel", "roll_lataccel", "v_ego", "a_ego")
    }


def _write_ring(dest, ring, head, scale):
    split = head + 1
    if split >= HIST_LEN:
        dest[:, :] = ring / scale
        return
    tail = HIST_LEN - split
    dest[:, :tail] = ring[:, split:] / scale
    dest[:, tail:] = ring[:, :split] / scale


def fill_obs(
    buf,
    target,
    current,
    roll_la,
    v_ego,
    a_ego,
    h_act,
    h_lat,
    hist_head,
    ei,
    future,
    step_idx,
):
    v2 = torch.clamp(v_ego * v_ego, min=1.0)
    k_tgt = (target - roll_la) / v2
    k_cur = (current - roll_la) / v2
    fp0 = future["target_lataccel"][:, step_idx, 0]
    fric = torch.sqrt(current**2 + a_ego**2) / 7.0
    pa = h_act[:, hist_head]
    pa2 = h_act[:, (hist_head - 1) % HIST_LEN]
    pl = h_lat[:, hist_head]
    buf[:, 0] = target / S_LAT
    buf[:, 1] = current / S_LAT
    buf[:, 2] = (target - current) / S_LAT
    buf[:, 3] = k_tgt / S_CURV
    buf[:, 4] = k_cur / S_CURV
    buf[:, 5] = (k_tgt - k_cur) / S_CURV
    buf[:, 6] = v_ego / S_VEGO
    buf[:, 7] = a_ego / S_AEGO
    buf[:, 8] = roll_la / S_ROLL
    buf[:, 9] = pa / S_STEER
    buf[:, 10] = ei / S_LAT
    buf[:, 11] = (fp0 - target) / DEL_T / S_LAT
    buf[:, 12] = (current - pl) / DEL_T / S_LAT
    buf[:, 13] = (pa - pa2) / DEL_T / S_STEER
    buf[:, 14] = fric
    buf[:, 15] = torch.clamp(1.0 - fric, min=0.0)
    _write_ring(buf[:, C:H1], h_act, hist_head, S_STEER)
    _write_ring(buf[:, H1:H2], h_lat, hist_head, S_LAT)
    buf[:, F_LAT:F_ROLL] = future["target_lataccel"][:, step_idx] / S_LAT
    buf[:, F_ROLL:F_V] = future["roll_lataccel"][:, step_idx] / S_ROLL
    buf[:, F_V:F_A] = future["v_ego"][:, step_idx] / S_VEGO
    buf[:, F_A:OBS_DIM] = future["a_ego"][:, step_idx] / S_AEGO
    buf.clamp_(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  Rollout: collect trajectories via real sim + compute action-gradients via BPTT
# ══════════════════════════════════════════════════════════════


def collect_and_compute_grads(
    csv_files, ac, diff_phys, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE
):
    """Phase 1+2: Collect trajectories from real sim, then compute action-gradients via BPTT.

    Returns dict with obs, raw_actions, logp, values, rewards, action_grads, means, stds.
    """
    # Phase 1: Real sim rollout (same as exp055 but with Gaussian policy)
    data, rng = csv_cache.slice(csv_files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
    )
    N = sim.N
    dg = sim.data_gpu
    S = COST_END_IDX - CONTROL_START_IDX  # 400
    future = _precompute_future_windows(dg)

    h_act = torch.zeros((N, HIST_LEN), dtype=torch.float64, device="cuda")
    h_act32 = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_lat = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    h_error = torch.zeros((N, HIST_LEN), dtype=torch.float32, device="cuda")
    err_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    obs_buf = torch.empty((N, OBS_DIM), dtype=torch.float32, device="cuda")

    all_obs = torch.empty((S, N, OBS_DIM), dtype=torch.float32, device="cuda")
    all_raw = torch.empty((S, N), dtype=torch.float32, device="cuda")
    all_logp = torch.empty((S, N), dtype=torch.float32, device="cuda")
    all_val = torch.empty((S, N), dtype=torch.float32, device="cuda")
    all_mean = torch.empty((S, N), dtype=torch.float32, device="cuda")
    all_std = torch.empty((S, N), dtype=torch.float32, device="cuda")

    si = 0
    hist_head = HIST_LEN - 1

    def ctrl(step_idx, sim_ref):
        nonlocal si, hist_head, err_sum
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
            mean, std = ac.forward_actor(obs_buf)
            val = ac.critic(obs_buf).squeeze(-1)
        eps = torch.randn_like(mean)
        raw = (mean + std * eps).clamp(-1.0, 1.0)
        log_std = std.clamp_min(1e-6).log()
        logp = (
            -0.5 * ((raw - mean) / std.clamp_min(1e-6)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        )

        delta = raw.to(h_act.dtype) * ds
        action = (h_act[:, hist_head] + delta).clamp(STEER_RANGE[0], STEER_RANGE[1])

        h_act[:, next_head] = action
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head

        if step_idx < COST_END_IDX:
            all_obs[si] = obs_buf
            all_raw[si] = raw
            all_logp[si] = logp
            all_val[si] = val
            all_mean[si] = mean
            all_std[si] = std
            si += 1
        return action

    costs = sim.rollout(ctrl)["total_cost"]
    S_actual = si

    # Compute rewards from sim histories
    start, end = CONTROL_START_IDX, CONTROL_START_IDX + S_actual
    pred = sim.current_lataccel_history[:, start:end].float()
    tgt = dg["target_lataccel"][:, start:end].float()
    lat_r = (tgt - pred) ** 2 * (100 * LAT_ACCEL_COST_MULTIPLIER)
    jerk = torch.diff(pred, dim=1, prepend=pred[:, :1]) / DEL_T
    rew = -(lat_r + jerk**2 * 100).float()
    dones = torch.zeros((N, S_actual), dtype=torch.float32, device="cuda")
    dones[:, -1] = 1.0

    # Phase 2: Compute action-gradients via BPTT through diff physics
    # For each H-step window, backprop the cost through the expected-value
    # physics model to get ∇_action(cost) for each step.
    H = RPO_H
    action_grads = torch.zeros((S_actual, N), dtype=torch.float32, device="cuda")

    # Process windows
    for win_start in range(0, S_actual, H):
        win_end = min(win_start + H, S_actual)
        win_len = win_end - win_start
        t_start = CONTROL_START_IDX + win_start

        # Get actions for this window (with gradient)
        win_raw = all_raw[win_start:win_end].T.clone()  # (N, win_len)
        win_raw.requires_grad_(True)

        # Get context from sim histories
        ctx_start = max(0, t_start - CONTEXT_LENGTH + 1)
        act_hist = sim.action_history[:, ctx_start:t_start].float()
        state_hist = sim.state_history[:, ctx_start : t_start + 1, :3].float()
        pred_hist = sim.current_lataccel_history[
            :, max(0, t_start - CONTEXT_LENGTH) : t_start
        ].float()

        CL = CONTEXT_LENGTH
        if act_hist.shape[1] < CL - 1:
            act_hist = F.pad(act_hist, (CL - 1 - act_hist.shape[1], 0), value=0.0)
        if state_hist.shape[1] < CL:
            state_hist = F.pad(
                state_hist, (0, 0, CL - state_hist.shape[1], 0), value=0.0
            )
        if pred_hist.shape[1] < CL:
            pred_hist = F.pad(pred_hist, (CL - pred_hist.shape[1], 0), value=0.0)

        cur_la = sim.current_lataccel_history[:, max(0, t_start - 1)].float()

        # Autoregressive rollout through diff physics
        win_cost = torch.zeros(N, device="cuda")
        prev_la = cur_la

        for h in range(win_len):
            t = t_start + h
            # Action: prev_steer + win_raw[:, h] * ds
            if h == 0:
                prev_act = sim.action_history[:, t - 1].float()
            else:
                prev_act = (
                    action_h.detach()
                )  # detach to avoid multi-step chain through tokenizer

            action_h = (prev_act + win_raw[:, h] * ds).clamp(
                STEER_RANGE[0], STEER_RANGE[1]
            )

            # Update action context
            act_ctx = torch.cat([act_hist[:, 1:], action_h.unsqueeze(1)], dim=1)

            # Update state context
            new_st = torch.stack(
                [
                    dg["roll_lataccel"][:, t].float(),
                    dg["v_ego"][:, t].float(),
                    dg["a_ego"][:, t].float(),
                ],
                dim=-1,
            )
            st_ctx = torch.cat([state_hist[:, 1:, :], new_st.unsqueeze(1)], dim=1)

            # Build model input
            full_states = torch.cat([act_ctx.unsqueeze(-1), st_ctx], dim=-1)
            tokens = diff_phys.tokenize(pred_hist)

            # Expected lataccel (differentiable w.r.t. action_h via full_states)
            pred_la = diff_phys.expected_lataccel(full_states, tokens)
            pred_la = torch.clamp(
                pred_la, cur_la - MAX_ACC_DELTA, cur_la + MAX_ACC_DELTA
            )

            # Per-step cost
            target_la = dg["target_lataccel"][:, t].float()
            win_cost += (target_la - pred_la) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
            win_cost += ((pred_la - prev_la) / DEL_T) ** 2 * 100

            # Advance context (detach pred for next step's tokenization)
            prev_la = pred_la
            act_hist = act_ctx.detach()
            state_hist = st_ctx.detach()
            pred_hist = torch.cat(
                [pred_hist[:, 1:], pred_la.detach().unsqueeze(1)], dim=1
            )
            cur_la = pred_la.detach()

        # Backprop to get ∇_raw(cost) for this window
        win_cost_mean = win_cost.mean()
        win_cost_mean.backward()

        if win_raw.grad is not None:
            # action_grads[t, n] = ∂cost/∂raw[n, t]
            action_grads[win_start:win_end] = win_raw.grad.T.detach()

    # Flatten for updates
    obs_flat = all_obs[:S_actual].permute(1, 0, 2).reshape(-1, OBS_DIM)
    raw_flat = all_raw[:S_actual].T.reshape(-1)
    logp_flat = all_logp[:S_actual].T.reshape(-1)
    val_2d = all_val[:S_actual].T
    mean_flat = all_mean[:S_actual].T.reshape(-1)
    std_flat = all_std[:S_actual].T.reshape(-1)
    agrads_flat = action_grads[:S_actual].T.reshape(-1)

    return dict(
        obs=obs_flat,
        raw=raw_flat,
        old_logp=logp_flat,
        val_2d=val_2d,
        rew=rew,
        done=dones,
        mean_old=mean_flat,
        std_old=std_flat,
        action_grads=agrads_flat,
        costs=costs,
    )


# ══════════════════════════════════════════════════════════════
#  RPO Update
# ══════════════════════════════════════════════════════════════


class RPO:
    def __init__(self, ac):
        self.ac = ac
        self.pi_opt = optim.AdamW(ac.actor.parameters(), lr=PI_LR, betas=(0.7, 0.95))
        self.vf_opt = optim.AdamW(ac.critic.parameters(), lr=VF_LR, betas=(0.7, 0.95))

    def _gae(self, rew, val, done):
        N, S = rew.shape
        adv = torch.empty_like(rew)
        g = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(S - 1, -1, -1):
            nv = val[:, t + 1] if t < S - 1 else g
            mask = 1.0 - done[:, t]
            g = (rew[:, t] + GAMMA * nv * mask - val[:, t]) + GAMMA * LAMDA * mask * g
            adv[:, t] = g
        return adv.reshape(-1), (adv + val).reshape(-1)

    def update(self, gd, ds=DELTA_SCALE):
        obs = gd["obs"]
        raw = gd["raw"]
        agrads = gd["action_grads"]
        mean_old = gd["mean_old"]
        std_old = gd["std_old"]

        # GAE for value function
        adv_t, ret_t = self._gae(gd["rew"], gd["val_2d"], gd["done"])

        # Batch-of-batch variance reduction
        if SAMPLES_PER_ROUTE > 1:
            n_total, S = gd["rew"].shape
            n_routes = n_total // SAMPLES_PER_ROUTE
            adv_2d = adv_t.reshape(n_routes, SAMPLES_PER_ROUTE, -1)
            adv_t = (adv_2d - adv_2d.mean(dim=1, keepdim=True)).reshape(-1)

        n_samples = len(obs)
        pi_sum, kl_sum, ent_sum, n_pi = 0.0, 0.0, 0.0, 0
        vf_sum, n_vf = 0.0, 0

        # M epochs of policy update with cached action-gradients
        for epoch in range(RPO_M):
            for idx in torch.randperm(n_samples, device="cuda").split(MINI_BS):
                mb_obs = obs[idx]
                mb_raw = raw[idx]
                mb_agrads = agrads[idx]
                mb_mean_old = mean_old[idx]
                mb_std_old = std_old[idx]

                # Current policy params
                mean_new, std_new = self.ac.forward_actor(mb_obs)

                # Inverse reparameterization: recover noise
                eps_reg = (mb_raw - mean_new) / std_new.clamp_min(1e-6)

                # Rebuild action: a = mean + std * eps_reg (differentiable w.r.t. theta)
                action_rebuilt = mean_new + std_new * eps_reg.detach()

                # Importance ratio
                old_logp = (
                    -0.5 * ((mb_raw - mb_mean_old) / mb_std_old.clamp_min(1e-6)) ** 2
                    - mb_std_old.clamp_min(1e-6).log()
                    - 0.5 * np.log(2 * np.pi)
                )
                new_logp = (
                    -0.5 * ((mb_raw - mean_new) / std_new.clamp_min(1e-6)) ** 2
                    - std_new.clamp_min(1e-6).log()
                    - 0.5 * np.log(2 * np.pi)
                )
                rho = (new_logp - old_logp).exp()

                # Asymmetric clipping mask
                mask = ((rho >= 1 - C_LOW) & (rho <= 1 + C_HIGH)).float()

                # RPG surrogate: ρ * ∇_θ(action) · cached_action_grad
                # action_rebuilt is differentiable w.r.t. θ
                # We want: loss = -ρ * action_rebuilt · action_grad (stop-grad on action_grad and ρ)
                # So that ∇_θ loss = -ρ * ∇_θ(action_rebuilt) * action_grad
                surrogate = -(
                    rho.detach() * mask.detach() * action_rebuilt * mb_agrads.detach()
                ).mean()

                # KL divergence: D_KL(π_old || π_new)
                kl = (
                    torch.log(std_new / mb_std_old.clamp_min(1e-6))
                    + (mb_std_old**2 + (mb_mean_old - mean_new) ** 2)
                    / (2 * std_new**2 + 1e-8)
                    - 0.5
                ).mean()

                # Entropy bonus
                ent = (
                    0.5 + 0.5 * np.log(2 * np.pi) + std_new.clamp_min(1e-6).log()
                ).mean()

                loss = surrogate + LAMBDA_KL * kl - LAMBDA_ENT * ent

                self.pi_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), GRAD_CLIP)
                self.pi_opt.step()

                bs = idx.numel()
                pi_sum += surrogate.item() * bs
                kl_sum += kl.item() * bs
                ent_sum += ent.item() * bs
                n_pi += bs

        # Value function update
        for _ in range(VF_EPOCHS):
            for idx in torch.randperm(n_samples, device="cuda").split(MINI_BS):
                val = self.ac.critic(obs[idx]).squeeze(-1)
                vf_loss = F.mse_loss(val, ret_t[idx])
                self.vf_opt.zero_grad(set_to_none=True)
                vf_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 1.0)
                self.vf_opt.step()
                vf_sum += vf_loss.item() * idx.numel()
                n_vf += idx.numel()

        with torch.no_grad():
            _, std_diag = self.ac.forward_actor(obs[:1000])
            sigma_eff = std_diag.mean().item() * ds

        return dict(
            pi=pi_sum / max(1, n_pi),
            vf=vf_sum / max(1, n_vf),
            kl=kl_sum / max(1, n_pi),
            ent=ent_sum / max(1, n_pi),
            σ=sigma_eff,
            lr=self.pi_opt.param_groups[0]["lr"],
        )


# ══════════════════════════════════════════════════════════════
#  Eval
# ══════════════════════════════════════════════════════════════


def evaluate(ac, files, mdl_path, ort_session, csv_cache, ds=DELTA_SCALE):
    data, rng = csv_cache.slice(files)
    sim = BatchedSimulator(
        str(mdl_path), ort_session=ort_session, cached_data=data, cached_rng=rng
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
            raw = ac.get_delta(obs_buf)
        delta = raw * ds
        action = (h_act[:, hist_head].float() + delta).clamp(
            STEER_RANGE[0], STEER_RANGE[1]
        )
        h_act[:, next_head] = action.double()
        h_act32[:, next_head] = action.float()
        h_lat[:, next_head] = cur32
        hist_head = next_head
        return action.double()

    costs = sim.rollout(ctrl)["total_cost"]
    return float(np.mean(costs)), float(np.std(costs))


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════


def train():
    ac = ActorCritic().to(DEV)

    # Load exp055 weights (Beta → Gaussian: map actor weights, reinit last layer)
    if RESUME and BEST_PT.exists():
        ckpt = torch.load(BEST_PT, weights_only=False, map_location=DEV)
        ac.load_state_dict(ckpt["ac"])
        print(f"Resumed from {BEST_PT}")
    else:
        ckpt = torch.load(BASE_PT, weights_only=False, map_location=DEV)
        # Load all layers except the last (different output format)
        state = ckpt["ac"]
        own_state = ac.state_dict()
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
        ac.load_state_dict(own_state)
        print(f"Loaded base from {BASE_PT} (partial — last actor layer reinitialized)")

    ds_ckpt = ckpt.get("delta_scale", None)
    if ds_ckpt is not None:
        global DELTA_SCALE
        DELTA_SCALE = float(ds_ckpt)
    ds = DELTA_SCALE

    print(f"  Actor params: {sum(p.numel() for p in ac.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in ac.critic.parameters()):,}")

    # Diff physics for BPTT
    diff_phys = DiffPhysics(ROOT / "models" / "tinyphysics.onnx").to(DEV)
    print(f"Loaded diff physics (onnx2torch)")

    rpo = RPO(ac)

    mdl_path = ROOT / "models" / "tinyphysics.onnx"
    ort_sess = make_ort_session(mdl_path)
    all_csv = sorted((ROOT / "data").glob("*.csv"))
    va_f = all_csv[:EVAL_N]
    csv_cache = CSVCache([str(f) for f in all_csv])

    vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
    best, best_ep = vm, "init"
    print(f"Baseline: {vm:.1f} ± {vs:.1f}")

    print(f"\nRPO: H={RPO_H}  M={RPO_M}  c_low={C_LOW}  c_high={C_HIGH}")
    print(f"  λ_kl={LAMBDA_KL}  λ_ent={LAMBDA_ENT}  pi_lr={PI_LR}  vf_lr={VF_LR}")
    print(f"  csvs={CSVS_EPOCH}  K={SAMPLES_PER_ROUTE}")
    print()

    def save_best():
        torch.save({"ac": ac.state_dict(), "delta_scale": ds}, BEST_PT)

    for epoch in range(MAX_EP):
        ac.train()
        t0 = time.time()
        n_routes = min(CSVS_EPOCH, len(all_csv)) // SAMPLES_PER_ROUTE
        batch = random.sample(all_csv, max(n_routes, 1))
        batch = [f for f in batch for _ in range(SAMPLES_PER_ROUTE)]

        res = collect_and_compute_grads(
            batch, ac, diff_phys, mdl_path, ort_sess, csv_cache, ds=ds
        )
        t1 = time.time()

        info = rpo.update(res, ds=ds)
        tu = time.time() - t1

        line = (
            f"E{epoch:3d}  train={np.mean(res['costs']):6.1f}  σ={info['σ']:.4f}"
            f"  π={info['pi']:.4f}  vf={info['vf']:.1f}  kl={info['kl']:.4f}"
            f"  H={info['ent']:.2f}  lr={info['lr']:.1e}  ⏱{t1 - t0:.0f}+{tu:.0f}s"
        )

        if epoch % EVAL_EVERY == 0:
            ac.eval()
            vm, vs = evaluate(ac, va_f, mdl_path, ort_sess, csv_cache, ds=ds)
            mk = ""
            if vm < best:
                best, best_ep = vm, epoch
                save_best()
                mk = " ★"
            line += f"  val={vm:5.1f}±{vs:4.1f}{mk}"
        print(line)

    print(f"\nDone. Best: {best:.1f} (epoch {best_ep})")


if __name__ == "__main__":
    train()
