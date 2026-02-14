# RL for the comma Controls Challenge v2

## The problem

comma's [Controls Challenge v2](https://github.com/commaai/controls_challenge) asks you to build a lateral acceleration controller. You get the car's state (speed, acceleration, road roll) and a future plan of target lateral accelerations. You output steering torque. The simulator is a GPT-based ONNX model trained on real car data — not a physics equation, so you can't derive a transfer function.

The dataset has 20,000 driving segments, each about 600 steps at 10Hz. Your controller takes over after step 100. Cost is `50 * mean(tracking_error²) + mean(jerk²)`, averaged across segments. Lower is better.

comma's own [blog post](https://blog.comma.ai/rlcontrols/) found that PPO doesn't converge reliably on this problem due to the noisy GPT simulator, and that evolutionary methods (CMA-ES) work better. Their best result was CMA-ES optimizing 6 parameters of a hand-designed controller to a cost of 48.

I spent 18 months and 50 experiments trying to make PPO work. It does. Here's what mattered and what didn't.

<!-- TODO: insert final score screenshot -->

---

## Phase 1: Can a neural net even do this? (exp014–021)

### One neuron (exp016–017)

I started with the simplest possible thing: a single neuron with 3 weights, trained by behavioral cloning from the stock PID controller. Input: `[error, error_integral, v_ego]`. Output: steer torque. The network recovered the PID gains exactly: `P=0.195, I=0.100, D=-0.053`. Cost ~120 — worse than PID (~84) because BC from a mediocre teacher gives you a mediocre student.

But it proved neural nets can learn control from demonstrations. That was enough to keep going.

### First PPO attempts (exp021)

Standard PPO on top of the BC policy. BC cost ~75. Added exploration noise (σ=0.007). Cost immediately jumped to 87. After PPO updates: 76. Worse than BC.

The finding: **control is extremely sensitive to perturbation**. Even tiny exploration noise destroys tracking. Failed rollouts produce harmful gradients that push the policy further from the BC baseline. Every PPO experiment ended the same way — the policy degrades from BC then never recovers.

Tried: clipping exploration, BC regularization (too strong = frozen policy, too weak = degradation), reward shaping. Nothing worked. I concluded what comma's blog post said: PPO doesn't converge on this problem.

---

## Phase 2: Understanding why PPO fails (exp031–038)

### Residual PPO (exp032)

Instead of replacing PID, add a tiny correction on top: `u = u_PID + 0.1 * π(s)`. The policy explores in the "nullspace" of PID — small perturbations that PID doesn't care about.

Training reward looked fine. Eval cost: 75,000. The policy learned to correlate its output with the tracking error — effectively fighting PID instead of helping it. The residual correction was small enough not to matter during training but catastrophic during evaluation because it accumulated over 500 steps.

Finding: **the policy and the PID must agree on what to do, or they fight**.

### Pure feedforward (exp033)

Removed error from the observation entirely. State = `[v_ego, a_ego, roll, future_curvatures]`. The policy can only do feedforward — anticipate, not react. By construction, `Cov(π, error) = 0`.

This never worked well, but the idea — separating feedforward from feedback — shaped everything that followed.

### Physics analysis (exp038)

Stopped training. Spent time analyzing the ONNX model: 20-step context window of `[steer, roll, v_ego, a_ego]` plus tokenized lataccel history. 1024-bin output distribution over [-5, 5]. The plant gain scales as v². The model has memory and inertia — it's not a static map.

Key insight: **the ONNX model IS a physics simulator**. The `set_model` hook in the evaluation framework lets the controller query it. This opened the door to model-based corrections at inference time.

### Linear preview (exp034) and temporal probes (exp037)

Tested whether future target information helps. Linear preview (3 weights × 3 horizon features): cost 102. Temporal probes (7 orthogonal features): similar. The preview information was there but a linear model couldn't extract it.

Finding: **preview needs nonlinearity and delta encoding**. Raw future targets aren't useful — the network needs to see rates of change.

---

## Phase 3: The delta-action breakthrough (exp047–048)

### Neural atom (exp047)

Grew a controller from scratch. BC from best feedforward controller: 50 weights, cost 82.89. Only `w[0] = 0.267` mattered — the first lataccel value in the history. Grid search over exponential decay: `w[i] = 0.267 * exp(-i/τ)`, optimal `τ ≈ 0.9`. Cost 81.46. Effective horizon: ~0.3 seconds.

This told me the controller mostly needs the immediate past, not deep history.

### Delta-action PPO (exp048)

The key idea: **don't output steer directly. Output the change in steer.**

```
steer_t = steer_{t-1} + network_output * scale
```

A zero output holds steady. Large jerk requires large network outputs, which the bounded action space resists. Smoothness by construction. This is velocity-form control — the same idea behind bumpless transfer in industrial PID.

BC from PID with delta encoding: cost ~85. PPO from there got to ~60 in 50 epochs. PPO was finally learning, not degrading. The delta formulation solved the exploration sensitivity that killed every previous PPO attempt — exploration noise produces smooth steering variations instead of random jitter.

---

## Phase 4: Refining to sub-50 (exp050)

### Observation space: 256 dimensions

The observation vector is 256 dimensions. Each feature is hand-scaled to roughly [-1, 1]:

- **16 core features**: target, current, error, error_integral, curvatures (κ_target, κ_current, κ_error), v_ego, a_ego, roll, prev_action, target_rate, current_rate, steer_rate, friction_circle, grip_headroom
- **40 history features**: 20 steps of steer commands + 20 steps of lataccel response
- **200 future features**: 50 steps × 4 channels (target_lataccel, roll, v_ego, a_ego)

This was arrived at by surgery. Started at 381 dims (with v/a/roll history, innovation history, future curvatures). Systematically removed features that didn't help BC loss. The 381→256 cut improved BC baseline from ~85 to ~75. Fewer dimensions = sharper gradients on the features that matter.

The 5 physics features (target_rate, current_rate, steer_rate, friction_circle, grip_headroom) were added because they involve multiplications and square roots that a ReLU network can't easily derive from raw inputs.

### Beta distribution over Gaussian

Bounded action space needs a bounded distribution. Gaussian + clipping creates probability mass spikes at the edges that distort the policy gradient. Beta naturally lives on [0,1], maps to [-1,1]. Cleaner gradients.

### NLL behavioral cloning

The biggest training speedup came from a subtle BC change. The original MSE loss trained only the Beta mean — the network learned where to steer but had no idea how certain it was (σ=0.45, nearly uniform). PPO then wasted 50+ epochs just collapsing σ.

Switching to negative log-likelihood forced the Beta concentration up during pretraining. σ dropped to 0.05-0.10 at the end of BC. PPO started from a concentrated, calibrated policy and improved immediately — validation decreased monotonically from epoch 1.

### PPO stabilization

The recipe that made PPO stable on this noisy plant:

- **GAMMA=0.95, LAMBDA=0.9**: shorter horizon, lower-variance advantages. The ONNX model's stochastic sampling makes long-horizon returns very noisy.
- **Reward normalization**: divide rewards by running std (Welford's algorithm). Keeps value function targets stable.
- **Per-minibatch advantage normalization**: every minibatch gets zero-mean, unit-variance advantages.
- **AR(1) colored noise** (ρ=0.5): temporally correlated exploration. Smooth steering variations instead of i.i.d. jitter.
- **Huber loss for the critic** (δ=50): MSE for normal errors, L1 for outliers. Prevents gradient explosions from reward normalization edge cases.
- **Checkpoint all training state**: model weights, optimizer state, AND reward normalization statistics. Missing any one of these causes divergence on resume.

### Newton correction at inference

The ONNX model is available at test time via `set_model`. One ONNX call per step:

1. Policy outputs a steering command
2. Run the model forward: predict next lataccel
3. Compare prediction to the cost-optimal 1-step target: `y* = (target + 2*current) / 3`
4. Correct: `steer -= K * (predicted - y*)`

The `(target + 2*current) / 3` comes from minimizing the per-step cost `5000*(target - y)² + 10000*(y - current)²`. Differentiating gives `y* = (target + 2*current) / 3` — only 1/3 of the way toward the target, because the jerk penalty is 2x heavier than tracking.

This dropped cost by ~2.5 points. The policy handles the big picture (anticipation, planning). The correction handles the fine details (plant mismatch, one-step precision).

---

## Results

<!-- TODO: fill in once training completes -->

| Stage | Cost | Notes |
|-------|------|-------|
| PID baseline | ~84 | Stock PID |
| One neuron BC | ~120 | 3 weights from PID |
| exp021 PPO | ~76 | Couldn't beat BC |
| exp048 delta PPO | ~60 | Delta-action breakthrough |
| exp050 BC (NLL) | ~75 | 256-dim, Beta, NLL |
| exp050 PPO (tuned) | ~47 | After ~1000 epochs |
| + Newton correction | ~46 | 1-step predict-and-correct |
| **5000-seg eval** | **XX.X** | **Official score** |

<!-- TODO: training curves, compute details -->

---

## What didn't work

**Standard PPO on absolute actions** (exp021): exploration noise destroys control. The policy degrades from BC and never recovers.

**Residual PPO on top of PID** (exp032): the policy correlates with error and fights PID. 75,000 cost.

**Observation normalization** (exp050): RunningMeanStd on observations. Regressed baseline from 57 to 540. Hand-scaled features beat learned normalization because they preserve semantic structure.

**Synthetic tracking error in BC** (exp050): added noise to `current_lataccel` during BC so the policy would see errors. But the expert action was computed for the clean state. Incoherent (obs, action) pairs.

**Multi-step MPC** (exp050): PPO policy proposes candidates, ONNX model simulates 5 steps forward. Consistently worse than the policy alone — MPC overrides the policy's long-horizon plan with myopic 5-step optimization, introducing jerk.

**Multi-step optimal target** (exp050): solved the Euler-Lagrange conditions for a jerk-minimizing trajectory. Cost: 1054. Even after fixing bugs, the ONNX model can't execute arbitrary acceleration profiles.

**MSE value loss without safety** (exp050): unbounded gradients from outlier returns corrupted the critic, which propagated to the actor via the shared loss → NaN.

**Missing checkpoint state** (exp050): reward normalization statistics weren't saved. On resume, stats reset, reward scale shifts, policy loss goes 0.003 → 16.7 → NaN.

---

## What I learned

**Delta-action formulation is the foundation.** Nothing else matters if the action space doesn't encode smoothness. Every absolute-action PPO attempt failed. Every delta-action attempt worked.

**Hand-crafted features beat learned normalization.** When you understand the physics, encoding it directly (curvatures, rates, friction circle) is more sample-efficient than letting the network figure it out from raw data.

**BC quality determines PPO ceiling.** MSE BC trained the mean but left σ=0.45 (uniform). NLL BC trained both mean and concentration. PPO went from "can't beat BC in 200 epochs" to "monotonically improving from epoch 1."

**The ONNX model is for correction, not planning.** MPC (multi-step search) failed. Newton (1-step correction) worked. The policy handles the plan. The model handles precision.

**Every piece of training state must be checkpointed.** Model weights, optimizer momentum, reward normalization statistics. Skip any one and training is non-reproducible on resume.

---

Code: `controllers/exp050_rich.py` (controller), `experiments/exp050_rich_ppo/train.py` (training).
