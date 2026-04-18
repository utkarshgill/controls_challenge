# Controls Challenge

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls.

That's comma's own framing. The [controls challenge](https://github.com/commaai/controls_challenge) is a GPT-based car simulator — an ONNX model that takes steering commands and spits out lateral acceleration. It's autoregressive: each prediction feeds back as input to the next step. 600 timesteps per route. 20,000 routes. The job is to track a target lateral acceleration profile.

The cost function:

```
total_cost = lataccel_cost * 50 + jerk_cost
```

where lataccel_cost is mean squared tracking error (×100) and jerk_cost penalizes jerky steering. The 50× multiplier means tracking matters way more than smoothness. The original prize winner scored under 45. PID baseline sits around 85.

The controller sees four things each step: the target lateral acceleration, the current lateral acceleration, the car state (velocity, forward acceleration, road roll), and a future plan — 50 timesteps of where the road is going.

```python
def update(self, target_lataccel, current_lataccel, state, future_plan):
    return steer_action  # a number between -2 and 2
```

---

## ChatGPT couldn't write a controller

Before I touched any code, I asked ChatGPT about the No Free Lunch theorem. It gave me a line I kept coming back to:

> There is no free lunch — but the world hands out cheap meals if you guess its structure correctly.

Then I pasted the README and asked it to write an MPC controller. What followed was five rounds of broken code. The dynamics model in every version was the same wrong thing:

```python
def predict_lataccel(self, current_lataccel, u):
    return current_lataccel + u * self.dt  # wrong
```

This treats the steering command as something that adds directly to lateral acceleration. But the ONNX simulator is a black box — it maps (states, tokens) → next_lataccel through a learned neural network. There's no closed-form dynamics. The relationship between steer input and lataccel output is nonlinear, speed-dependent, and history-dependent.

ChatGPT's fix for the oscillating MPC was always the same: increase the smoothing weight, decrease the learning rate, try again. Five rounds of knob-turning with zero insight into why it was oscillating. The dynamics model was wrong the entire time.

This taught me something: a generic AI assistant with the full spec still can't solve this. I had to understand the structure of what I was controlling.

---

## LunarLander first

When I started the controls challenge, there were too many unknowns. Bad results could mean broken PPO, wrong architecture, or insufficient data. Three suspects, entangled.

So I stopped. Went and solved LunarLander instead.

Not because LunarLander matters, but because it isolates PPO as a variable. If I can solve LunarLander in 100 epochs with clean code, PPO is no longer the suspect. The whole thing is 304 lines — `archive/beautiful_lander.py`. What transferred directly to the controls challenge:

- Separate actor/critic optimizers (pi_lr=3e-4, vf_lr=1e-3)
- Learnable state-independent log_std
- GAE with advantage normalization
- Gradient clipping at 0.5

The principle: **fix one invariant at a time.** This became the throughline of the whole project. LunarLander fixes PPO skills. PID baseline fixes the floor. A one-neuron experiment (exp016) fixes "can learning work at all" — yes, 3 weights recover PID gains. Each fix eliminates one suspect.

---

## The representation breakthrough

Two changes took me from 130 to 43. Both are about representing the problem correctly.

### Curvature encoding

The future plan has 50 timesteps of target_lataccel, roll_lataccel, v_ego, and a_ego. That's 200 numbers. The PID controller ignores all of them.

The raw numbers are garbage for an MLP. target_lataccel at 30 km/h and 120 km/h means completely different things for the steering wheel. What stays constant is curvature — the geometry of the road:

```python
def _curv(lat, roll, v):
    return (lat - roll) / max(v * v, 1.0)
```

This compresses 4×50 future plan values into 50 curvatures. Speed-invariant. The road geometry doesn't change with speed, just the lateral force required to follow it.

The moment I switched from raw future plan to curvature, BC-from-PID dropped from 130 to 75. One line of physics. The NFL theorem was right — the world hands out cheap meals if you guess its structure correctly. Curvature was the cheap meal.

### Delta actions

The other critical change: instead of outputting a steering angle directly, the network outputs a *change* in steering.

```python
delta = clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)  # DELTA_SCALE=0.25, MAX_DELTA=0.5
action = clip(prev_action + delta, -2, 2)
```

Why this matters: the simulator is autoregressive. Exploring with noise on absolute actions means a single noisy action at step 100 corrupts the state, which corrupts step 101, which corrupts step 102... the whole rollout goes bad. But with noise on *delta* actions, a noisy step just wiggles the steering slightly — the previous action anchors it. Noise stays local.

This wasn't derived from theory. The commit messages tell the real story: "revert to 47 config" after trying other values. DELTA_SCALE=0.25 and MAX_DELTA=0.5 survived because other values didn't work.

---

## The debugging grind

PPO couldn't beat BC for months. The pattern was always the same: BC gives 75, start PPO fine-tuning, cost immediately jumps to 100, 150, 300.

The causes were all silent failures. No exceptions. No NaNs. The network just quietly learned nothing or the wrong thing:

**EPS_CLIP was 0.** The PPO clipping range was set to [1.0, 1.0]. Every policy update was a no-op. The ratio was always clamped to exactly 1. PPO was literally doing nothing while appearing to train normally. Loss went down because the critic learned, but the actor never moved.

**Reward function bugs.** The training reward didn't match the eval cost. Missing the 50× lataccel multiplier. Jerk computed as `(cur - prev)²` instead of `((cur - prev)/0.1)²` — off by 100×. The network was optimizing the wrong objective.

**BC trained the mean but not the concentration.** With a Beta distribution, MSE loss only matches α/(α+β) to the target. There's zero gradient incentive for the distribution to get sharper. σ stays at 0.45 after BC instead of shrinking. Fix: NLL loss that trains both α and β.

**State-dependent sigma.** Sounds sophisticated — let the network modulate exploration per state. In practice it learned to avoid hard states by inflating σ there. Training looked great. Eval was terrible because E[tanh(μ + σε)] ≠ tanh(μ). The network compensated by learning extreme μ values, knowing noise + squashing would pull them back. Remove the noise at eval time and the actions are too aggressive.

Each of these took days to find. Together they explain months of "PPO can't beat BC."

---

## The sprint

Everything came together in one week.

```
Feb 9:   "ppo shows promise"           first commit in 37 days
Feb 10:  88 → 80 → 79                  PPO from scratch, no BC, beating PID
Feb 11:  96 → 72 → 66                  256-dim obs, Beta distribution
         "65.4 validation is a real breakthrough after months of work"
Feb 12:  112 → 63.8 in 11 epochs       NLL BC + reward normalization
Feb 13:  57 → 50 → 47                  cosine LR decay
         "I had to start it 8-9 times to reach 47 cost"
Feb 14:  47 → 43                        16 git commits in one day
         "fix ratio collapse" → "new best 43"
Feb 15:  42.5 on 5000 routes            #3 on leaderboard
```

The final architecture: 256-dim observation (16 core features + history buffers + future curvatures), 4-layer actor, 4-layer critic, 256 hidden, Beta distribution, delta actions, NLL behavioral cloning pretrain followed by PPO fine-tuning with cosine LR decay.

The reward function in training, matching eval exactly:

```python
def compute_rewards(traj):
    lat  = (tgt - cur)**2 * 100 * LAT_ACCEL_COST_MULTIPLIER  # 50×
    jerk = np.diff(cur, prepend=cur[0]) / DEL_T
    return (-(lat + jerk**2 * 100) / 500.0)
```

Half of Feb 15 was a GPU optimization blitz. The bottleneck was never the algorithm — it was how fast I could run rollouts. Batched ONNX inference (1000 routes in parallel instead of 1 at a time). Three MacBooks connected via USB-C for distributed rollouts. Then Google Cloud 48-core CPU. Then a Vast.ai GPU with TensorRT. 375× total speedup. Not smarter rollouts — faster rollouts. The bitter lesson.

The last piece: hybrid policy-MPC at inference. Sample 16 action candidates from the trained policy's Beta distribution. Roll each forward 5 steps through the ONNX simulator. Score each trajectory with the real cost function. Apply the best first action. Slot 0 is always the policy mean, so it never picks something worse than the pure policy. 2-4 point improvement for free.

---

## What I still don't know

**a_ego.** Curvature κ = (lataccel - roll) / v² encodes where the road bends but not when the bend arrives. That's a_ego — the rate of change of velocity. The simulator takes it as an explicit input. I never figured out the right way to use it. "Curvature tells you where the road bends. a_ego tells you when the bend arrives."

**Why PID works so well.** A 3-parameter controller with no future information scores 85. My 4000-parameter neural network with 50 steps of future plan and months of work scores 42.5. That's not even a 2× improvement. PID is doing something right that I don't fully understand.

**The gap from 42.5 to 35.9.** The #1 solution (haraschax, 35.97) uses heavier MPC compute — more candidates, longer horizons, more ONNX calls per step. The policy gets to ~45. The ONNX model as a forward simulator gets to ~36. Both are needed. I had the policy. I didn't push MPC hard enough.

The controls challenge is still a good benchmark of what ML can and can't do. PPO learned to anticipate curves, steer early, and trade tracking error against jerk in a way PID can't. But it took months of debugging silent failures, one critical representation insight (curvature), and a 375× speedup to get there. Modern RL is not plug and play. The reward function has to match eval exactly. The action space has to respect the simulator's dynamics. The observation has to encode the physics, not just concatenate numbers.

There is no free lunch. But the world does hand out cheap meals if you guess its structure correctly.
