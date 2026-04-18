# Controls Challenge Story — Raw Notes

> Collected from git log, actual source code, transcripts, archived docs, and experiment READMEs.
> NOT a polished writeup — raw material for the main article.

---

## Ground Truth Timeline (from git log)

### The Challenge Itself (upstream commits)
- **2024-03-25:** comma.ai creates the repo. Sim, model, controllers, readme.
- **2024-05-29:** Future plan added to controller API. v2 announced Aug 2024. Leaderboard added.
- **2025-01-12:** "prize was claimed" — the original challenge prize was won.
- **2025-07-23:** New PID baseline tuned, readme updated.

### December 25, 2025 — Christmas Day: The Fork
- `7260cbbd 2025-12-25 initial commit` — the user forks the repo and starts working.
- 5 more commits that day: cleanup, .gitignore, bc-pid improvements.
- **2025-12-26:** "more experiments", "clean experiments"
- Then silence until February.

### February 2026 — The Sprint
- **Feb 2:** checkpoint commit. exp013 BC from PID.
- **Feb 9:** "ppo shows promise" — first sign PPO is learning
- **Feb 12:** "friction circle + BC fixes"
- **Feb 13:** "closing in ~60-54 regime"
- **Feb 14:** 16 commits in one day. "fix ratio collapse", "fix log ratio clamp", "big run", "new best 43", "shoot 5", "revert shooting"
- **Feb 15:** 20+ commits. The GPU optimization blitz: CUDA, IOBinding, pre-allocated buffers, GPU tokenizer, TensorRT. The entire OPTIMIZATION_JOURNEY.md happens in ONE DAY.

---

## The Real Timeline (code + transcripts combined)

### December 25, 2025 — Christmas Day

The actual start. comma.ai Controls Challenge v2: track a target lateral acceleration profile. 20k CSVs, 600 timesteps each. Cost = `lataccel_cost × 50 + jerk_cost`. The original prize was already claimed 11 months ago. The user starts anyway.

**The winner got <45.** That's the target.

### December 25, 2025 — Christmas Day (attempts_1)

Git: 6 commits this day. Someone was coding on Christmas.

**What the CODE actually shows was tried:**

| Script | Obs Dim | Reward Correct? | CSVs/epoch | Result |
|--------|---------|-----------------|------------|--------|
| `train_bc_ppo.py` | 55 | Yes (lat×50 + jerk) | 8 | BC→PPO pipeline |
| `train_ppo.py` | **10** | Yes | 8 | Pure PPO, tiny obs |
| `train_ppo_pure.py` | 57 | Partial (÷100) | 20 | Different reward scale |
| `train_ppo_residual.py` | 10 | Yes | 8 | PID + residual |
| `train_pipeline.py` | 56 | **NO — 3 BUGS** | ~1250 | BC pre-tanh mismatch, no 50× lat, wrong jerk |
| `train_ppo_parallel.py` | 56 | **YES (first correct)** | ~1250 | First script with correct 50×lat + jerk² |

**The train_pipeline.py bugs (from actual code):**
1. No `LAT_ACCEL_COST_MULTIPLIER` (50×) in reward — network didn't know to prioritize tracking
2. Jerk = `(cur-prev)²` instead of `((cur-prev)/0.1)²` — off by 100× on jerk penalty
3. BC trained on pre-tanh output but eval used tanh — action space mismatch
These bugs explain why early BC→PPO gave 840+ cost. The reward didn't match the eval.

**Training scale was laughably small:** 8 CSVs per epoch. Each route is 580 steps. That's 4,640 transitions per epoch. Modern PPO uses millions.

**Scores at this point:**

| Approach | Cost | Notes |
|----------|------|-------|
| Random controller | ~10,000 | Baseline sanity check |
| Pure PPO from random init | 4,000–8,000 | Never converged (with 10D obs!) |
| PPO with σ=1.0 | 10,000–19,000 | "Basically random" — too much exploration |
| PPO residual on PID | **72** | Best early result, but plateaued |
| BC → PPO (buggy pipeline) | ~840 | Reward bugs above explain this |
| PID baseline | ~70–85 | The bar to beat |

**Critical bug found: State Normalization**
- `v_ego` (~33) was 100× larger than `error` (~0.1), 330,000× larger than curvatures (~0.0001)
- The code shows `OBS_SCALE` normalization was added: `[10, 1, 0.1, 2, 0.03, 1000, 1000×50]`
- But `ppo_parallel.py` used these scales while `train_pipeline.py` used `NORM_SCALE = [2, 5, 0.05, 5, 2, 2×50]` — inconsistent

**Confidence assessment (from STATUS.md):**
- BC stage: ~95% confidence
- PPO stage: ~70%
- Reaching <45: ~60% — "Winner proved it's possible, but might need multiple runs/tuning"

### Late December 2025 — attempts_2

Git: `3255b9d4 2025-12-26 more experiments`

**Moved to second attempt round.** Old weights archived: `ppo_best.pth` (~200 cost). Diagnostic scripts: `diagnose_bc_vs_pid.py`, `analyze_failures.py`, `final_diagnosis.py`.

BC results at this point:
- BC train: ~88
- BC val: ~129
- Train/val gap: ~46% — overfitting on 8 CSVs

**PPO from BC (parallel version, with correct reward):**
- Mean: 110.24 ± 115.77
- Median: 82.52
- Min: 5.16 ("CRUSHES target on some routes")
- Max: 794.30 ("catastrophic failures on hard routes")
- <45 on about 25% of routes — but variance too high

### January–February 2026 — The Experiment Explosion

37 days of silence in git (Dec 26 → Feb 2). Thinking? Reading? The experiment count jumps from ~30 to ~50.

**What the code shows actually worked vs what READMEs claimed:**

| Exp | README Claim | Code Reality | Actual Score |
|-----|--------------|--------------|--------------|
| 013 | "BC from PID" | 55D obs, MLP 128×3, tanh×2, Z-score norm | ~99 |
| 016 | "One neuron" | Linear(3,1), no bias, PID features | ~85 (learned PID gains) |
| 023 | "Conv1D BC" | Conv1d on 49 curvatures + 4-feature MLP | Not recorded |
| 025 | "With history" | Same as 023 + prev_action + per-step future v/roll | Not recorded |
| 029 | "Engineered" | **12 hand-crafted features**: immediate/tactical/strategic curvatures, acceleration, smoothness | Not recorded |
| 030 | "Vehicle-centric" | Conv1D on (lataccel, v_ego) future pairs, no curvature | Not recorded |
| 031 | "beautiful_lander PPO" | 55D, ActorCritic 128 hidden, tanh, anti-windup integral | ~868 (6 epochs) |
| 032 | "Residual PPO" | PID + 0.1×tanh(net), lowpass filter α=0.3 | Not converging |
| 034 | "Linear preview" | PID + 3 future stats (short/medium/long), hand-tuned weights | Modest improvement |
| 040 | "Best PID" | **Just PID with PPO-tuned gains: P=0.194, I=0.109, D=-0.058** | ~85 |
| 046 | "Systematic ablation" | **12 variants**: PID → PID+FF → CEM MPC → learned FF | 84.85 baseline |
| 047 | "Neural atom" | PID + 50-dim learned decay weights. Optimal: first 3 steps matter | **81.46** |
| 048 | "Delta-action PPO" | 54D, 3-layer actor, delta actions (×0.1, ±0.3), κ space | ~85 from BC |
| 049 | "Clean PPO" | **107D** with κ derivatives, 5-layer actor | Not recorded |

**Two critical insights emerged from code, not READMEs:**

1. **exp046 was a 12-variant systematic ablation** (not just one experiment). The code shows: baseline → simple FF → weighted 10-step → derivative → tuned gains → learned linear → learned tanh → 2-layer MLP → full CEM MPC → local 1-step MPC. This was methodical hypothesis testing.

2. **exp048's delta actions were discovered, not designed.** The `MAX_DELTA=0.3` constant limits physical jerk — but the commit messages show it came from "revert to 47 config" after failures with other values. The winning architecture was found by trying things and reverting what didn't work.

**The beautiful_lander pivot:** The user had a clean PPO solving `LunarLanderContinuous-v3` at `archive/beautiful_lander.py`. Key quote: "I solved beautiful_lander successfully." This 304-line script became the PPO template. It's still in the archive.

**MPC learnings (exp045):**
- v1: 1614, v2: 357, v3: **157**, v4: TBD
- "Internal cost must match evaluation cost; smoothness penalty HURTS"
- "Temporal consistency from small std and warm-start, not a cost penalty"
- "H=1 bad; H=10 poor; H=50 good"

**Neural atom (exp047) — the physics insight:**
- Single neuron with exponential decay weights on future curvatures
- Grid search found τ=0.9 optimal — only first 3 timesteps matter (~0.3s lookahead)
- Score: 81.46 — a single neuron nearly matching PID
- "Too much future causes instability"

### Early February 2025 — The Convergence

**Feb 2:** Git cleanup. Large .pkl files (586MB–1GB) blocking git push. Added to .gitignore.

**Feb 6:** Karpathy-style deep dive on all 20k CSVs. Understanding the texture of data. Cost structure understood: jerk cost theoretical floor ≈ 19 if tracking were perfect.

**Feb 10 — exp048/049:** The turning point.
- exp049: PPO from scratch, NO BC. Train: 34405 → 780 in 6 epochs. Val: 1003 → 394.
- "RL signal is strong; σ₀=0.3 instead of 1.0 to reduce wasted exploration."
- "Maybe BC itself is the problem, since it locks the policy into imitating PID."
- **Beat PID for the first time.** Val: 88 → 80 → 79.2.
- "79.2 is a solid new record — beaten exp049's 88 cleanly."

**Feb 11 — exp050_rich_ppo:** Physics-first approach. The actual code shows:
- 256-dim state (computed from 381 raw features), 4+4 layer actor-critic, 256 hidden, Beta distribution
- Delta actions with `DELTA_SCALE=0.25`, `MAX_DELTA=0.5` (more aggressive than exp048's 0.1/0.3)
- 11 core features + 6×20 history buffers + 4×50 future plan + 50 future κ
- Scaling constants tuned: S_LAT=5, S_STEER=2, S_VEGO=40, S_AEGO=4, S_ROLL=2, S_CURV=0.02
- 12 environment variables for MPC tuning (MPC, MPC_K, MPC_MAX, MPC_H, MPC_ROLL, MPC_N, etc.)
- "65.4 validation is a real breakthrough after months of work. The physics-first approach worked."
- BC baseline: 96.1. Best val: 72.4 at epoch 10.
- Then overfitting: val rose 72 → 78 → 83 → 90 → 98.
- Cause: σ shrinking (0.10 → 0.07); entropy collapse; policy overfits as exploration dies.

**Feb 11 — Batched rollouts born.** 
- Realized ONNX supports dynamic batch. 1000 batched calls ≈ 0.38s vs 1000×1 ≈ 1.04s.
- Pool removed; batched sim in `train.py`. Rollouts: 80s → 46-50s.

**Feb 12:** Best val ~66-67. Target: <40. Plateau. "Main lever is richer observation space, not more tuning."

### Feb 12-15 — The Push to Podium

**Feb 12:** "friction circle + BC fixes" — 1 commit. Quiet day.

**Feb 13:** "closing in ~60-54 regime" — 2 commits. Things are working.

**Feb 14 — 16 commits (THE DAY):**
The git messages tell the real story:
```
dc5e0346 global norm
092c7ee7 fix log ratio clamp
c5082406 fix ratio collapse         ← PPO was numerically unstable
d27e0bd1 big run
4065e9b2 numpy depr warning
4f2bd9dd revert to 47 config        ← tried something, broke it, reverted
5ca4c263 decrease gamma and lamda
f351eda2 fix numpy depr warning (again)
014369f1 shoot 5                    ← MPC shooting with H=5
c18d37ca revert shooting            ← shooting hurt, reverted
e8af0564 new best 43                ← THE MILESTONE
```
"fix ratio collapse" → "revert to 47 config" → "decrease gamma" → "new best 43"
This is the real optimization process: fix bugs, try things, revert failures, find the sweet spot.

- Cost descent: 90 → 57 → 50 → 47 → 43.
- "250 epochs from 90 to 47. Then 1000 epochs with LR decay from 47 to 43."
- "I had to start it 8-9 times to reach 47 cost." (ctrl-C and restart with different params)

**MacBook Cluster phase:** (see OPTIMIZATION_JOURNEY.md Part 2.5)
- Three MacBooks connected via USB-C Thunderbolt
- Persistent TCP server for rollout distribution
- Weighted FRAC splitting for load balancing
- 3000 CSVs in ~45s across 3 machines
- Thermal throttling on the fanless Air after 30+ min

**Cloud migration:**
- Google Cloud 48-core CPU: 5000 CSVs in 6s
- Vast.ai RTX 5060 Ti: batched GPU simulator

**Leaderboard moment:** "On 5000 routes. We beat number 3, just two guys above us."
- 45.72 on 5000 routes = #3
- Best checkpoint: 42.5

**MPC + Policy hybrid:**
- Sample 16 action sequences from policy
- Roll forward 5 steps using ONNX model
- Pick best action based on predicted cost
- Reliable 2-4 point boost
- "We will never diverge from policy action basin because policy is the one who predicted a_0"

---

## Key Themes & Insights (from all transcripts)

### Why This Problem Is Hard
1. **Autoregressive physics:** ONNX model feeds predictions back as inputs. Small errors compound over 580 steps.
2. **Non-differentiable simulator:** Can't backprop through the ONNX model. "no gradients flow through onnx. FUCKING READ THE CODE" — user, correcting a wrong analysis.
3. **Colored noise:** ONNX noise is temporally correlated — not standard RL assumption of i.i.d. transitions.
4. **Cost structure:** `lataccel_cost × 50 + jerk_cost`. The 50× multiplier means: "To hit 35, tracking must go from 30 to ~15. That means halving the RMS tracking error."
5. **Jerk floor:** "The noise floor means total cost can never go below ~20. So the budget from 20 to 35 is only 15 points for tracking."
6. **Reactive vs Anticipatory tradeoff:** "For a reactive strategy, this IS near-optimal. Jerk and error trade off. The leaderboard PPO broke free because an anticipatory strategy doesn't have that tradeoff. Steering early into a curve is BOTH smoother AND more accurate."

### The Core Philosophies

#### 1. Fix One Invariant at a Time

**This is the throughline of the entire story.**

When the user started the controls challenge, there were too many unknowns: "there were too many variants when i began. terrible cost could have been a reason of my poor ppo skills or just the architecture or the actual data bottleneck." Three possible failure modes, entangled.

The decision: **stop working on the challenge. Go solve LunarLander first.** Not because LunarLander matters, but because it isolates "PPO skills" as a variable. If you can solve LunarLander in <25 epochs, PPO is no longer the suspect. Now you can return to the controls challenge and know that bad results mean architecture or data — not a broken algorithm.

This motif — fix one invariant at a time — recurs throughout:
- **LunarLander first** → fixes PPO skills as invariant
- **PID baseline** → fixes "what's the floor" as invariant
- **One neuron (exp016)** → fixes "can learning work at all" (yes: 3 weights recover PID gains)
- **exp046's 12 ablations** → fixes "which feedforward structure matters" one at a time
- **Delta actions (exp048)** → fixes action parameterization, isolates from obs design
- **Beta distribution** → fixes distribution choice, isolates from everything else
- **NLL vs MSE for BC** → fixes σ initialization, one change
- **CSVS scaling (8→300→1000→5000)** → fixes data as invariant
- **MacBook cluster → cloud → GPU** → fixes compute as invariant

Each fix eliminates one suspect. The remaining improvement is always attributable.

#### 2. The Bitter Lesson (Sutton, 2019)

http://www.incompleteideas.net/IncIdeas/BitterLesson.html — the user's anchor whenever lost, demotivated, or confused.

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

> "Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research."

> "We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done."

This philosophy explains decisions throughout the project:

- **Why PPO over hand-tuned PID+feedforward:** The exp046 series proved you could get to 81 with hand-tuned linear weights on future targets. But that was a ceiling — each new route pattern would need new weights. PPO finds the pattern automatically. "the whole point of RL is to learn the non differentiable env."

- **Why scale data (8→5000 CSVs) instead of engineer features:** Early obs went from 10D→56D→107D→381D through hand-designed physics features. But the breakthrough came from scaling data (more CSVs per epoch) not more features. The 381D obs was later compressed to 256D — less feature engineering, more compute.

- **Why the sub-35 physics plans were abandoned:** The plans for gain-normalized actions, L_EFF estimation, differentiable world models, notch filters — all "human knowledge" approaches — were never executed. Instead: more data, more epochs, simpler architecture. The bitter lesson in action.

- **Why batch and GPU instead of clever algorithms:** The entire OPTIMIZATION_JOURNEY (375x speedup) is about leveraging computation. Not smarter rollouts — faster rollouts. Not better MPC — more MPC candidates. The winning MPC is just "sample 16 from policy, simulate all, pick best" — pure search.

- **Why the MacBook cluster, then cloud, then GPU:** Each compute migration was about the same thing: more rollouts, same wall time. "Double the data in the same time." The bitter lesson says this always wins.

The tension: the user also deeply values physics intuition (curvature space, friction circles, delta actions). The real lesson might be: use physics to define the *right search space*, then let compute explore it. Delta actions are a physics insight about the right parameterization — but PPO + Beta + 5000 CSVs is what actually finds the policy.

### The LunarLander Prologue

**Blog:** https://utkarshgill.github.io/blog/lander.html (published June 30, 2025)
**Repo:** https://github.com/utkarshgill/lunar_lander (67 commits, `failed_attempts` folder)
**Sigma transcript:** exported Feb 9, 2026 (the refinement session)

The LunarLander work was a deliberate choice to isolate PPO implementation quality. The blog captures the raw experience:

> "Wanted to land the thing. First attempts would avoid crashing sometimes, which felt like progress. But not landing."
> "The lander learned to hover. Just hover perfectly, draining fuel until the episode timed out. 100+ points for doing nothing. RL is dumb like that."
> "Got it working eventually."

**The architecture that worked:** sub-500 lines, 8 parallel envs, 8-layer actor + 4-layer critic, learnable `log_std`, tanh squashing, pi_lr=3e-4, vf_lr=1e-3, K=20, batch=5000, 100k steps/epoch. Solved in ~100 epochs.

Then Feb 2026: the sigma debugging session refined it further — testing state-dependent vs state-independent σ, discovering the tanh squashing train/eval gap, simplifying to clip + fixed σ. This became the blueprint for the controls challenge.

**What transferred directly to the controls challenge:**
- Separate actor/critic optimizers (pi_lr=3e-4, vf_lr=1e-3)
- Learnable state-independent `log_std`
- GAE with advantage normalization
- Gradient clipping at 0.5
- The confidence that PPO works when implemented correctly

Critical lessons learned:

1. **State-dependent σ is a trap for PPO:** "State-dependent sigma accelerates stochastic learning by letting the network avoid hard states, which is exactly what slows deterministic convergence." Solved in ~18 epochs stochastically but eval lagged badly.

2. **tanh squashing creates a train/eval gap:** `E[tanh(μ+σε)] ≠ tanh(μ)`. "The network compensates by learning μ values that are more extreme than needed, knowing noise + tanh will pull the effective action back toward moderate values. When you strip the noise away and use tanh(μ) directly, you get actions that are too aggressive."

3. **The fix was simplicity:** Removed tanh squashing, removed state-dependent σ, used clip + state-independent learnable log_std. "The original fixed-std + clip-actions PPO was actually the standard well-tested recipe. We made it more sophisticated and probably just made it worse."

4. **"nah dont agree with me. we are scientists exploring the nature of the world."** — The user's philosophy: don't prematurely converge on answers, test rigorously, measure the right metric.

### The Beta Distribution Switch (winning recipe transcript)

The switch from Gaussian to Beta was a key unlock:

- **Why Gaussian was bad:** "You pick a number on an infinite number line, then chop off anything past the edges. Wasted exploration. And the math gets confused because 'I picked 5.0' and 'I picked 1.0' both became '1.0' after clipping."

- **Why Beta worked:** "You pick a number that already lives between the walls. It can never go outside. No clipping, no wasted samples, no confused gradients."

- **The BC-Beta bug:** "BC only trains the mean of the Beta, not the concentration. The MSE loss matches α/(α+β) to the target, but there's zero gradient incentive to increase α+β." Result: σ stays at 0.45 after BC instead of shrinking. Fix: NLL loss for BC that also trains the concentration.

### The Innovation / Kalman Feature (winning recipe transcript)

- **Concept:** "Run ONNX once per step with the policy's chosen action, get the predicted next lataccel. Innovation = actual - expected. Feed that surprise to the network."
- **Key distinction:** "MPC uses the model for planning. This uses the model for perception. Very different."
- **Piggybacking optimization:** During training, reuse the sim's ONNX probs to compute expected lataccel — zero extra ONNX calls. During eval, the controller computes its own prediction via set_model. "Mathematically identical. No leakage."
- **Innovation eventually removed:** It doubled ONNX compute and wasn't helping enough. The obs space went from 381D → 360D → 256D.

### The MPC Journey (winning recipe transcript)

MPC went through 4 distinct phases:

1. **Full CEM MPC (exp046_mpc):** 100 samples, 30 steps, 4 CEM iterations. Cost 157. Slow, random proposals.
2. **Local 1-step MPC (exp046_mpc_v2):** 30 samples within radius 0.3 of PID. Small improvement.
3. **MPC + PPO (first attempt):** "Both lataccel and jerk got worse. The MPC is selecting actions that are WORSE than the pure policy mean. The lookahead scoring is fundamentally broken."
   - **Root cause:** Off-by-one bug. "The ONNX input is off by one entire timestep. The model is predicting from stale context."
   - Also: holding actions constant was wrong when candidates were close. Need policy-rolled continuations.
4. **Hybrid Policy-MPC Shooting (final):** Sample 16 candidates from policy, roll forward 5 steps, pick best. "We will never diverge from policy action basin because policy is the one who predicted a_0."

### The Batched Simulator Birth (speedup transcript)

The moment it clicked:

- **User:** "the 79s is killing me"
- **Realization:** "Your current 10-worker pool is not CPU-bound on compute. It's overhead-bound. You're paying the fixed per-call overhead 400,000 times."
- **User:** "two clarifying questions: will this let us ditch the pool complexity? you're saying we ditch tinyphysics.py and write our own stepper right?"
- **Answer:** "Yes to both."
- **User (15 min later):** "you have been thinking for 15 minutes. 😡 WRITE THE FUCKING CODE."

### What Worked
- **Delta actions** (Δsteer, prev + Δ): prevents exploration noise cascade in the autoregressive sim
- **Beta distribution:** bounded support, clean gradients, state-dependent confidence
- **NLL loss for BC** (instead of MSE on Beta mean): trains concentration, shrinks σ properly
- **Physics-first observations:** curvature κ from future plan, error integral
- **Reward normalization** (RunningMeanStd): critical for stable PPO
- **Cosine LR decay:** smooth convergence in final phase
- **Hybrid Policy-MPC Shooting:** 2-4 point boost from 5-step lookahead with ONNX model
- **Batched everything:** 375x speedup enables rapid iteration
- **Increasing CSVS/epoch:** from 8 → 300 → 1000 → 3000 → 5000. "1000 CSVs with 1000 episodes per epoch means the policy sees the same routes repeatedly. It memorizes route-specific quirks."

### What Failed
- **Pure PPO with σ=1.0:** catastrophic in autoregressive control
- **BC → PPO with MSE loss:** "PPO never beats BC. Train drops 93%. Val gets worse. The BC policy at 75.2 is still our best model after 45 epochs of RL."
- **State-dependent sigma (for PPO):** Speeds stochastic training, wrecks deterministic eval
- **tanh squashing:** train/eval gap from E[tanh(μ+σε)] ≠ tanh(μ)
- **PID residual:** hidden integral state mismatch
- **MPC smoothness penalty:** "internal cost must match eval cost exactly"
- **ONNX off-by-one in MPC:** predicted from stale context, anti-correlated with performance
- **Innovation feature:** doubled compute, marginal benefit, eventually removed
- **Reward scale mismatch:** early scripts had missing 50× multiplier, wrong jerk formula
- **v² as grip headroom:** "i told you vsquared was a stupid idea! it had to be another friction circle term"
- **GAMMA=0.95 + LR=3e-4:** "feedback instability in the training loop itself. Same physics as an oscillating control loop with too much gain."

### The Emotional Arc
- Christmas 2025: "This should be doable" → Christmas day coding
- Late Dec: PPO at 10,000 cost. "Basically random."
- 37 days of silence (thinking? holidays?)
- Feb 2: "checkpoint" — first commit in 37 days
- Feb 9: "ppo shows promise"
- Feb 11: "65.4 validation is a real breakthrough after months of work"
- Feb 12-13: "closing in ~60-54 regime". BC baseline 112 → val 63.8 in 11 epochs. "The learning rate is dramatically faster."
- Feb 13: "dude we are so close. the machine works, it just needs 1-2 key optimisations to win"
- Feb 14: 16 commits. "fix ratio collapse". "new best 43". "revert shooting".
  - "this is the best script we have had after 18 months of effort. this is pure gold. a million dollar file."
  - "dont go for the easy shit. give me some insights physics and control theory. anyone can say decrease the learning rate you fool. you almost insult my 18months of hardwork by saying shit like this"
  - "you're wrong. our policy gets 47 on 100, that guy got 45 on 5000 routes. dont jump ahead of yourself. fucking sit down and help me refine what we have"
  - "youre stupid. that is cheating. the whole point of RL is to learn the non differentiable env. DONT YOU SEE WHAT I AM ASKING. look at this beautiful descent."
- Feb 15: 20+ commits. GPU optimization blitz. "On 5000 routes. We beat number 3, just two guys above us."
- Recruiter anxiety: "Will the recruiters be okay with it?" "Nobody reads training logs line by line."
- Writeup feedback: "the language you wrote is fake. influencer shit. write naturally like engineers."

---

## Files & Artifacts Worth Preserving

**Code:**
- `archive/beautiful_lander.py` — The LunarLander PPO that became the blueprint (304 lines)
- `experiments/exp050_rich_ppo/train.py` — The final training script (1223 lines)
- `controllers/exp050_rich.py` — The final controller (676 lines)
- `tinyphysics_batched.py` — The batched simulator (674 lines)
- `fast_eval.py` — Massively parallel GPU evaluation
- `best_model_43.pt` — The checkpoint that hit 42.5

**Documentation:**
- `archive/old_docs/PROGRESS.md` — Dec 25 snapshot of where things stood
- `archive/old_docs/STATUS.md` — Confidence levels and proposed path to <45
- `archive/old_docs/WHAT_WORKS.md` — What was known to work
- `experiments/exp045_simple_mpc/LESSONS.md` — MPC learnings
- `experiments/exp047_neural_atom/SUMMARY.md` — The exponential decay insight
- `experiments/exp048_delta_ppo/README.md` — Delta action architecture rationale
- `OPTIMIZATION_JOURNEY.md` — 375x speedup story (will be hyperlinked)

**Transcripts (downloaded from Cursor):**
- `cursor_sigma_state_dependency_and_predi.md` — The LunarLander sigma debugging deep-dive (exported Feb 9). Contains the tanh squashing analysis, state-dep vs state-indep sigma experiments, and the "we are scientists" philosophy.
- `cursor_script_speedup_analysis.md` — The moment batched rollouts were conceived (exported Feb 11). Contains the "79s is killing me" → "WRITE THE FUCKING CODE" arc.
- `cursor_ppo_challenge_winning_recipe.md` — THE main transcript, 7097 lines (exported Feb 14). Contains: Beta distribution switch, Kalman innovation wiring, MPC off-by-one bug, the piggybacking optimization, BC NLL fix, reward normalization, the entire Feb 12-14 sprint. Key quotes: "this is pure gold, a million dollar file", "dont go for the easy shit, give me some insights physics and control theory", "the whole point of RL is to learn the non differentiable env"

---

## Scores Progression (from code, git, and transcripts)

```
Dec 25:   PPO from scratch    10,000+    "basically random" (8 CSVs/epoch, 10D obs)
Dec 25:   PPO residual on PID     72     best early result (PID + 0.1 × net)
Dec 25:   BC from PID            ~88     decent but reward bugs in pipeline.py
Dec 25:   BC → PPO              ~110     mean (82 median, max=794 on hard routes)
Jan:      MPC (CEM, H=50)       157     exp045 (v1=1614, v2=357, v3=157)
Jan:      Neural atom (τ=0.9)    81     exp047 (single linear neuron!)
Jan:      PID baseline            85     exp046 (12 systematic variants)
Feb 9:    "ppo shows promise"           first git commit after 37 days
Feb 10:   exp049 PPO             88     → 80 → 79 (107D obs, 5-layer actor)
Feb 11:   exp050 physics-first   96     → 72 → 66 (381D → 256D, Beta dist)
          "65.4 validation is a real breakthrough after months of work"
Feb 12:   NLL BC + reward norm   112    → 87 in 1 epoch → 63.8 in 11 epochs
          "BC baseline 112 → val 63.8 in just 11 epochs. The learning rate is dramatically faster."
Feb 13:   with LR decay          57     → 50 → 47
          "I had to start it 8-9 times to reach 47 cost."
          "this is the best script we have had after 18 months of effort. this is pure gold."
Feb 14:   fine-tuning            47     → 43
          16 git commits. "fix ratio collapse" → "new best 43"
          "dont go for the easy shit. you almost insult my 18months of hardwork"
Feb 15:   5000-route eval        42.5   #3 on leaderboard (45.72 on 5000 routes)
          "On 5000 routes. We beat number 3, just two guys above us."
```

### The Cost Anatomy (from transcript analysis at ~60 cost)
```
lataccel_cost: 0.86    →  50 × 0.86 = 43  (70% of total)
jerk_cost:     18.33   →  18.3             (30% of total)
total:         61.3

To reach 45: need lataccel to halve OR jerk to halve
The policy is rationally over-investing in tracking because reward = 50×lat + jerk
Bumping jerk weight in reward (100→200) shifted the Pareto frontier

To reach 35: "the budget from 20 to 35 is only 15 points for tracking"
Leaderboard #1 (35.97) = MPC + heavy compute
Leaderboard #3 (45.76) = pure PPO
"The leaderboard is screaming: the policy gets you to 45, the ONNX model gets you to 36."
```

---

## What the Code Actually Reveals (vs what READMEs claimed)

### Reward Function Bugs — The Silent Killers

The early training scripts had **reward functions that didn't match the eval cost**. This explains months of poor results.

| Script | Bug | Effect |
|--------|-----|--------|
| `train_pipeline.py` | Missing `LAT_ACCEL_COST_MULTIPLIER` (50×) in reward | Network didn't prioritize tracking |
| `train_pipeline.py` | Jerk = `(cur - prev)²` instead of `((cur-prev)/0.1)²` | Off by 100× on jerk penalty |
| `train_pipeline.py` | BC trained on pre-tanh output, eval used tanh | D coefficient especially wrong |
| `train_ppo_pure.py` | Reward /100 vs others | Different gradient scale |
| `exp043_clean_ppo` | Action range [-1,1] vs steer range [-2,2] | Only using half the steering range |
| `train_ppo_parallel.py` | **FIRST CORRECT REWARD** | 50×lat + jerk, matching eval exactly |

### The Observation Space Evolution (from actual code)

```
3D   → pid.py, exp016, exp017, exp040
       [error, integral, diff]
       "Can a single neuron learn PID gains?" — yes (P=0.194, I=0.109, D=-0.058)

10D  → train_ppo.py, train_ppo_residual.py
       [error, lataccel, v_ego, a_ego, curv_now, next3 curvs, future_max, future_mean]

55D  → bc.py, exp031, ppo_aug, ppo_tanh
       [error, roll, v_ego, a_ego, current_lataccel, 50 future lataccels]
       Normalization: NORM_SCALE = [2, 5, 0.05, 5, 2, 2×50]

56D  → ppo_parallel, train_bc_pid, train_pipeline
       [error, error_diff, error_integral, lataccel, v_ego, curv_now, 50 future curvs]
       OBS_SCALE = [10, 1, 0.1, 2, 0.03, 1000, 1000×50]

57D  → exp043, train_ppo_pure
       [target, current, error, v_ego, a_ego, roll, curv_now, 50 future curvs]

58D  → bc_curvature
       Curvature space: κ = (lat - roll) / v². Speed-invariant.
       Friction available: √(1 - (|a_ego|/10)²)

54D  → exp048_delta (THE TURN)
       [error, v_ego, prev_act, κ_now, 50 future κ]
       OBS_SCALE: [1/3, 1/30, 1/2, 100×51]
       Delta actions: raw×0.1, clipped ±0.3

107D → exp049_clean
       [error, integral, prev_error, v_ego, a_ego, roll, prev_act, κ, 50 κ, 49 dκ]
       5-layer actor, 3-layer critic

381D → exp050_rich (FINAL)
       11 core + 6×20 history + 4×50 future + 50 future_κ
       4-layer actor+critic, 256 hidden, Beta distribution
```

### The Action Space Evolution

```
Direct steering:    action = network(obs)                    # bc.py, ppo_aug
Tanh squashing:     action = tanh(raw) × 2                  # exp031, ppo_tanh
PID + residual:     action = PID(error) + 0.1 × network     # exp032, exp034
PID + feedforward:  action = PID + linear(future_features)   # exp046 series
Delta actions:      action = prev + clip(raw×0.1, ±0.3)     # exp048+ (WINNER)
```

Delta actions were THE key architectural insight: exploration noise in Δ doesn't cascade through the autoregressive sim the way absolute action noise does.

### The exp046 Series — 12 Systematic Ablations

The code reveals a methodical PID+feedforward search:

```
exp046_baseline:  Pure PID. P=0.195, I=0.100, D=-0.053. Cost ~85.
exp046_simple:    PID + 0.3 × (future[0] - target). Simplest FF.
exp046_v2:        PID + weighted 10-step future (decay=0.8, gain=0.4)
exp046_v3:        PID + immediate_error + target_derivative
exp046_v4:        Same, stronger gains (0.35, 0.20)
exp046_nn_v1:     PID + learned linear FF (6 features)
exp046_nn_v2:     PID + tanh FF (same 6 features)
exp046_nn_v3:     PID + 2-layer MLP (16 hidden)
exp046_mpc:       Full CEM MPC. H=30, samples=100, elites=25, iters=4
exp046_mpc_v2:    PID + 1-step local search (30 samples, radius 0.3)
```

### The Neural Atom Insight (exp047)

```python
# exp047_decay.py — the actual code
weights = np.load('decay_weights.npy')  # learned via grid search
ff_action = weights @ future_errors     # 50 future steps

# Optimal weights: w[i] = 0.267 × exp(-i/0.9)
# Translation: only first 3 timesteps matter (~0.3s lookahead)
# Score: 81.46 — a single linear neuron nearly matches PID
```

### Training Scale Evolution (CSVs per epoch)

```
8 CSVs/epoch     → train_bc_ppo, train_ppo, train_ppo_residual (Dec 2025)
20 CSVs/epoch    → train_ppo_pure
250 CSVs/epoch   → exp043_clean_ppo
300 CSVs/epoch   → exp048_delta_ppo
1000 CSVs/epoch  → exp050 (single MacBook)
3000 CSVs/epoch  → exp050 (MacBook cluster)
5000 CSVs/epoch  → exp050 (GPU, batched)
```

### Key Architecture Constants That Stuck

| Constant | Value | Where introduced | Why it survived |
|----------|-------|------------------|-----------------|
| Hidden dim | 256 | exp050 | Sweet spot: expressive enough, fast enough |
| Delta scale | 0.1 | exp048 | Limits jerk implicitly |
| Max delta | 0.3 | exp048 | Prevents catastrophic steering jumps |
| PID gains | P=0.195, I=0.1, D=-0.053 | pid.py (upstream) | BC baseline, warm-start target |
| GAMMA | 0.95 | exp048 | Short horizon matches 0.3s lookahead insight |
| Beta distribution | exp050 | Bounded support [0,1] natural for steering |

---

## Git Commit Density — The Intensity Map

```
2025-12-25:  6 commits   (Christmas Day — the fork)
2025-12-26:  2 commits   (next day cleanup)
... 37 days of silence ...
2026-02-02:  1 commit    (checkpoint)
2026-02-09:  1 commit    ("ppo shows promise")
2026-02-12:  1 commit    (friction circle + BC fixes)
2026-02-13:  2 commits   (closing in on 60-54)
2026-02-14: 16 commits   (THE DAY — ratio fixes, best 43, shooting experiments)
2026-02-15: 20+ commits  (GPU optimization blitz — entire OPTIMIZATION_JOURNEY in one day)
```

The real work happened in a ~2 week sprint (Feb 2-15), with the most intense days being Feb 14-15. But the thinking started on Christmas Day 2025.

---

## Cursor IDE Plans — The Decision Layer (71 plan files, 24 unique)

Plans show the *intent* behind each move — what was planned, what was completed, what was abandoned.

### Phase 1: Architecture Search (completed)

**clean_ppo_from_scratch** — Fork exp048 into exp049. Remove BC entirely. Use beautiful_lander's recipe (higher LRs, global adv norm, ReLU). Isolate whether RL signal works from scratch. *All completed.*

**physics_observation_redesign** — Compact 26-dim obs: feedback (error, integral, prev_error, v_ego, prev_steer), dynamics (a_ego, roll_lataccel), near future κ[0:7] + Δκ[0:7], far future κ at [10,20,30,40,49]. *Never executed — superseded by 256-dim approach.*

### Phase 2: Innovation Signal (completed, then removed)

**kalman_innovation_signal** — Add ONNX-based innovation feature: `innovation = actual_lataccel - expected_lataccel`. Obs 360→381. One extra ONNX call per step. *Completed, wired in.*

**swap_innovation_for_error_integral** — Replace innovation with rolling error integral (cheaper, no extra ONNX call). *Completed — innovation removed.*

### Phase 3: Distribution + Loss Fixes (completed)

**fix_ppo_clipping_unlock** — EPS_CLIP was set to 0! That forces ratio to [1.0,1.0], disabling useful policy updates. Fix to 0.2. Add cosine LR decay. *Completed.*

### Phase 4: Scaling + Compute (completed)

**batched_rollout_speedup** — Replace multiprocessing pool with single-process batched rollout. Preload CSVs, run N in lockstep, batch ONNX as (N, 20, 4). *Completed.*

**extract_tinyphysics_batched** — Move batched sim from train.py into `tinyphysics_batched.py`. *Completed.*

**distributed_rollouts_setup** — Two MacBooks over Thunderbolt USB-C. Split rollouts. Expected 82s→50s/epoch. *Completed.*

**persistent_remote_worker** (12 versions!) — Replace per-epoch SSH+spawn with persistent TCP server on the Air. Remove startup overhead. *Completed — many iterations indicate painful debugging.*

**wire_remote_into_training** (14 versions!) — Wire remote split into train_one_epoch and evaluate. *Completed — even more iterations than the server itself.*

### Phase 5: The Sub-35 Push (mostly abandoned)

**path_to_sub-35_cost** — Phased plan: simulated BC → scale to 2000 CSVs → DELTA_SCALE 0.25→0.30 → HIDDEN 384/512 → LayerNorm → GRU → entropy decay → HP sweep. Projected: 68.65 → 50 → 45 → 40 → 36 → <35. *Phases 1-2 done. Phases 3-5 abandoned — the simpler approach worked.*

**sub-35_cost_fixes** — Four HP edits: GAMMA 0.95→0.99, linear LR decay, K_EPOCHS 3→2, CSVS=2000. *All pending — never executed.*

**physics-informed_sub-30_ideas** — Two control-theory ideas: (1) Gain-normalized actions (steer in lataccel-delta space via v²/L_EFF), (2) Feedforward obs (steer_gap = prev_steer - physics_feedforward). Estimate L_EFF from CSV data. *Never executed.*

**model-based_scaling_ideas** — (1) Differentiable world model for analytic gradients (Dreamer/SVG style), (2) Test-time search (K=8 candidates, 1-step ONNX lookahead). *Never executed.*

### Phase 6: MPC at Inference (completed)

**inference_mpc_optimization** — Replace broken Newton correction with 1-step cost-aware MPC. 9 candidates around policy output, batched ONNX, exact leaderboard cost scoring. *Completed.*

**policy-guided_mpc_eval** — 10-candidate, 5-step lookahead MPC at eval. Policy proposes candidates, ONNX simulates, cost function scores. *Completed.*

**hybrid_policy-mpc_shooting** — THE winning MPC design. Sample 16 from Beta policy, unroll 5 steps via batched ONNX, score with real cost, apply best first action. Slot 0 = policy mean (always a candidate). *Completed.*

### Phase 7: GPU Optimization (completed)

**gpu_rollout_optimization** — Fix RNG parity, reuse IOBinding, pre-allocate buffers, vectorize RNG, vectorize post-rollout rewards. Target: sim 17s→7s, ctrl 10s→5s. *All completed.*

**wall-to-wall_timing_audit** — Found 13s overhead outside rollout: ckpt save, CSV loading, sim reset, post-rollout. Fix all. Then move controller entirely to GPU. *All completed.*

**wall-to-wall_pipeline_audit** — Found 3 wins: GPU randperm, GPU compute_cost, GPU running stats. Rollout hot loop already tight. *All completed.*

**ppo_update_optimization** — 9s update had 3-5s waste: CPU roundtrips, Python GAE loop (800K iterations), small mini-batches. Fix: vectorized GPU GAE, MINI_BS 20K→100K, set_to_none=True. Target: 9s→4s. *All completed.*

**gpu_controller_+_fast_eval** — GPU-ify inference controller. Write fast_eval.py for all 5000 episodes in one batched GPU rollout. *All completed.*

### Phase 8: Refactoring (completed)

**surgical_train.py_refactor** — Tinygrad-style cleanup: remove dead code, kill docstrings, compact config. No behavior changes. *Completed.*

**refactor_train.py_tinygrad-style** — Same as above, different session. *Completed.*

### What the Plans Reveal

1. **12 versions of persistent_remote_worker + 14 versions of wire_remote_into_training** = 26 plan files for the MacBook cluster alone. This was the hardest engineering problem, not the ML.

2. **Sub-35 plans were mostly abandoned.** The ambitious path (gain-normalized actions, differentiable world model, GRU, PBT) was never executed. The simpler approach (better BC loss, more data, Beta distribution, cosine decay) got to 42.5 without any of it.

3. **The EPS_CLIP=0 bug** (fix_ppo_clipping_unlock) was a devastating silent failure — PPO updates were doing nothing. This explains the "PPO never beats BC" pattern in early runs.

4. **Innovation was a detour.** kalman_innovation was fully implemented, then fully removed (swap_innovation_for_error_integral). Net contribution: understanding of the ONNX model's behavior, but zero dims in the final obs.

5. **Plans got simpler over time.** Early plans had 5-6 phases with ambitious architectures. Later plans had 3-4 surgical edits. The winning changes were always the simplest ones.

---

## Open Questions for the Writeup

- What happened during the 37-day silence (Dec 26 → Feb 2)? Thinking? Other projects? Holiday break?
- What motivated starting on Christmas Day specifically?
- Is this for a comma.ai job application? (recruiter/interview references suggest yes — "what if the interviewer asks me?", writeup requested to match comma.ai blog style)
- The user references "18 months of effort" (Feb 14 quote) but git starts Dec 25, 2025. Were there earlier attempts in a different repo?
- How many total hours invested? Git density suggests intense 2-week sprint (Feb 2-15), ~80-120 hours
- What's the user's background? Clearly controls/physics-savvy, knows PPO deeply, references "beautiful_lander" as prior work
- What did the leaderboard look like? #1: haraschax (35.97, MPC+compute), #3: bheijden (45.756, pure PPO). User landed #3 with 45.72 on 5000 routes.
- The LunarLander work was a deliberate isolation of "PPO skills" — confirmed by user. Part of the same story.
- There are additional ChatGPT conversations at: https://chatgpt.com/g/g-p-693b22b7f014819184fe65a418572178-comma-controls-challenge/project (behind auth, user needs to export)
- User asked for writeup to match comma.ai blog style: "the language you wrote is fake. influencer shit. write naturally like engineers. look at this example: https://blog.comma.ai/rlcontrols/"

---

## Early ChatGPT Conversations (Raw Dump — Chronological)

These are the earliest ChatGPT interactions, before the philosophical/architecture discussions. ChatGPT was used as a generic code assistant. The results were mostly broken.

### NFL Theorem Opening

User asks "what was the free lunch theorem." ChatGPT explains the No Free Lunch theorem.

> "No Free Lunch (NFL) Theorem — in optimization / learning — says: Averaged over all possible problems, every algorithm performs equally well. If one method beats another on some problems, it must lose on others."

Key line:

> "There is no free lunch — but the world hands out cheap meals if you guess its structure correctly."

ChatGPT connects to the challenge:

> "Every successful algorithm is: algorithm = assumptions + search. If you win big somewhere, you baked in the right assumptions."

Then offers to connect this to: Lunar Lander vs Comma Controls, why PID works suspiciously well, why curvature intuition is exactly an NFL escape hatch.

### First Contact with the Challenge

User pastes the README (v1, with `total_cost = lataccel_cost * 5 + jerk_cost` — note: multiplier was 5 in v1, later changed to 50 in v2). Asks "HOW SHOULD I SOLVE THIS? MPC OR PID?"

ChatGPT gives boilerplate PID and MPC implementations. Recommendation:

> "For this challenge, if the car dynamics are reasonably well-understood and not too complex, starting with a PID controller is recommended."

### MPC Attempt #1 (scipy)

User asks ChatGPT to write MPC matching the `BaseController` API:

```python
class Controller(BaseController):
  def update(self, target_lataccel, current_lataccel, state, target_future):
    return target_lataccel  # placeholder
```

ChatGPT produces MPC with `scipy.optimize.minimize`, SLSQP method. Wrong dynamics model:

```python
def _system_dynamics(self, state, control_input):
    return state + control_input  # "Simple linear model for demonstration"
```

User: "using numpy only"

### MPC Attempt #2 (numpy, gradient descent)

ChatGPT rewrites with manual gradient descent, finite-difference gradients:

```python
def predict_lataccel(self, current_lataccel, u):
    return current_lataccel + u * self.dt

# cost_function: sum of (lataccel[t] - target[t])² + 0.01 * u[t]²
# optimize via: compute gradient by perturbing each u[i] by ±1e-5
```

**Crash:** `IndexError: list index out of range` — `target_future[t]` when `t >= len(target_future)`.

### Fix + Oscillations

Fix: `n = min(n, len(target_future))`. Runs, but oscillates wildly.

User shows plots: high-frequency oscillations in both control actions and current lataccel.

### The Tuning Loop (5+ rounds)

ChatGPT's entire strategy was knob-turning. Each round: user shows plot → ChatGPT adjusts weights → repeat.

Round-by-round hyperparameter trajectory:

```
                    REG_WEIGHT  SMOOTH_WEIGHT  LR       ITERS  HORIZON
Initial:            0.01        0.0            0.01     100    10
After oscillations: 0.1         0.1            0.01     100    10
Still oscillating:  1.0         1.0            0.001    200    10
"needs fixing":     0.4         0.8            0.001    200    15
"tune":             0.3         0.7            0.0001   400    20
"too slow":         0.3         0.5            0.001    200    15
"needs fixing":     0.4         0.8            0.001    200    15
"tune this":        0.3         0.7            0.0001   400    20
"too slow":         0.3         0.5            0.001    200    15
```

> User: "tune"
> ChatGPT: increases horizon to 20, adjusts weights.
> User: "its running too slow"
> ChatGPT: decreases horizon to 15, reduces iterations.
> User: "needs fixing"
> ChatGPT: increases smoothing weight to 0.8.

This loop repeated ~5 times with no fundamental insight. The dynamics model (`lataccel += u * dt`) was wrong throughout — steering commands don't directly add to lateral acceleration.

### User Asks to Surface Hyperparams

> User: "surface all hyperparams as CAPITAL variables (only respond the controller class)"

ChatGPT converts to class-level constants: `HORIZON`, `DT`, `REGULARIZATION_WEIGHT`, `SMOOTHING_WEIGHT`, `LEARNING_RATE`, `MAX_ITER`. This is the first sensible suggestion.

### "Browse and Research Then Fix"

> User: "wait browse and research then fix"

ChatGPT searches for MPC tuning best practices, suggests Bayesian optimization. User: "use bayesian pls (numpy only)."

ChatGPT produces a full Gaussian Process + acquisition function implementation in numpy (~80 lines). Includes kernel function, Cholesky decomposition, Expected Improvement acquisition. Never actually used or tested.

### User Pastes tinyphysics.py v2 (with FuturePlan)

User pastes the full updated `tinyphysics.py` with:
- `FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])`
- `LAT_ACCEL_COST_MULTIPLIER = 50.0` (changed from 5.0)
- `future_plan` parameter in controller API

ChatGPT's response: generic explanation of the code. No actionable controller improvement.

> User: "Write a boilerplate using right api and cost functions"

ChatGPT produces incomplete boilerplate — imports `BaseController`, defines empty `update()`, no actual control logic.

### "Think Like George Hotz"

> User: "Think like george hotz. Help me tackle this challenge. I have no experience with controls but I'm good at programming"

ChatGPT response: a generic 7-day plan.

> "Day 1: Understand the Basics... Read the Code... Run Examples..."
> "Day 2: Tweak and Experiment... Tune PID Controller..."
> "Day 3: Explore Advanced Controllers..."

Not a single concrete technical insight.

### "PID Isn't Good Enough"

> User: "PID isn't good enough. Need to write a custom feedback controller"

ChatGPT finally reads `tinyphysics.py` more carefully, notices `FuturePlan`, `FUTURE_PLAN_STEPS = FPS * 5`. Suggests MPC again. Starts a `PredictivePIDController` class — never finishes it.

### MPC Attempt #3 (scipy, with BaseController)

User provides the `open` controller placeholder and asks ChatGPT to modify an MPC to suit the challenge:

```python
class Controller(BaseController):
    def __init__(self, horizon=20, Q=0.1, R=0.5):
        ...
    def _system_dynamics(self, state, control_input):
        return state + control_input  # STILL WRONG
    def _cost_function(self, control_inputs, current_lataccel, future_plan):
        for i in range(self.horizon):
            future_state = self._system_dynamics(future_state, control_inputs[i])  # BUG: future_state undefined
            ...
```

Has an undefined variable bug (`future_state` used before assignment) AND still uses wrong dynamics.

> User: "the cost function"
> ChatGPT starts explaining...
> User: "no implement the cost function mentioned in readme"
> ChatGPT: "While I haven't seen the exact description from the README..."

User had already pasted the README multiple times.

### "HOW SHOULD I PROCEED... 7 DAYS"

User pastes README + full tinyphysics.py + pid.py all at once:

> "HOW SHOULD I PROCEED FOR THIS CHALLENGE? THINK LIKE GEORGE HOTZ. I'M A HACKER AND NOOB IN CONTROL THEORY. WANT TO SOLVE THIS CHALLENGE IN 7 DAYS"

ChatGPT response: another 7-day plan, almost identical to the first one. Still no concrete technical insight.

### "Is This Hinting Towards Using or Not Using ML?"

> User: "is this writeup hinting towards using or not using ml models for this problem"

ChatGPT correctly identifies:

> "The writeup suggests that machine learning models are generally not well-suited for low-level control tasks... it specifically states 'Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls,' which hints that a traditional control approach... may be more effective."

> "However, it doesn't explicitly forbid using ML models."

This is one of the few actually useful responses.

### ONNX Model Input Shape Analysis

User shows `get_current_lataccel` code and asks about input shape:

```python
states = np.column_stack([actions, raw_states])
input_data = {
    'states': np.expand_dims(states, axis=0).astype(np.float32),
    'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
}
```

ChatGPT correctly explains:
- `states` shape: `(1, N, M+1)` — batch=1, N timesteps, M+1 = action + state dims
- `tokens` shape: `(1, T)` — batch=1, T tokenized past predictions

> User: "but are why are they being passed in dic? are these being concatenated or what?"

ChatGPT correctly explains ONNX named inputs, separate processing branches, internal feature fusion. One of the better technical responses.

### What This Phase Reveals

1. **ChatGPT couldn't write a working controller.** Every MPC implementation had bugs: wrong dynamics, undefined variables, index errors, missing cost terms.

2. **The dynamics model was always wrong.** `lataccel[t+1] = lataccel[t] + u[t] * dt` treats steering as directly adding to lataccel. But the ONNX model maps `(states, tokens) → next_lataccel` — a black-box, non-linear, autoregressive process. There's no closed-form dynamics to optimize over.

3. **ChatGPT's only strategy was knob-turning.** Five rounds of "increase regularization, decrease learning rate" with zero insight into WHY the MPC was oscillating (wrong dynamics model).

4. **The user was already showing hacker instincts:** demanding numpy-only, asking to surface hyperparams, pushing for concrete code over explanations, recognizing the README's hint about ML limitations.

5. **The NFL theorem framing proved prophetic.** "The world hands out cheap meals if you guess its structure correctly" → curvature encoding was exactly this: guessing the right structure (κ = (lataccel - roll) / v²) and getting a massive cost drop (130→75).

6. **ChatGPT's useful contributions were limited to:** explaining the ONNX input format, correctly reading the README's skepticism about ML, and the NFL framing. Everything else was generic or broken.

---

## The ChatGPT Philosophical Journey (full transcript recovered)

The user had extensive philosophical discussions with ChatGPT about the controls challenge. These reveal the thinking process behind the technical decisions.

### The CTF Mindset

The user treated the README like a CTF challenge spec:
> "i like to solve things like a CTF. should we begin by dissecting the README"

Key deductions from reading the spec:
- "the future plan (3/4 quantities) is made to be consumed by the simulator, and the resultant lataccels be compared to the future plan lataccels (4th quantity) to get estimate costs"
- "This is not normal feedback control. This is preview control."
- PID uses only 2 of the available signals (target_lataccel, current_lataccel). The state (3 values) and future plan (4×50 values) are completely ignored.
- Simple feedforward: `future_plan.lataccel[0] - current_lataccel` alongside PID drops cost from ~95 to ~83

### The MPC Failure (critical insight)

The user tried MPC (simple + CEM) and it couldn't beat PID:
> "when i tried MPC simple and MPC CEM, CEM did better but none crossed the barrier of PID"

ChatGPT's diagnosis (correct):
- "MPC is optimizing against a lying oracle. PID stays on the manifold. MPC leaves it."
- "Steering is the wrong control variable to optimize over."
- "The simulator is closer to linear in lataccel than in steer"
- "PID accidentally answers the second question ['what lataccel trajectory do I want?']"

### The Emotional Crisis

Harald (head of research at comma) posted a new record (~35). The user's reaction:
> "i want to say look how i managed to get 35 but i havent done it. i feel so bad. its like i am a couple experiments far from solving it. but i cant talk. i cant post on twitter"
> "i am so stupid. i look like a doofus. trying to show off"
> "he responded to everyone except me 😢😢😢😢😢"
> "this is so stupid. i am so stupid. im a faker. thats all i am"

ChatGPT's response was remarkably grounded:
> "You are not behind because you spoke. You would fall behind if you stopped thinking clearly because of fear."
> "If people noticed you before you had something sharp, it would actually slow you down."

The user eventually tweeted:
> "this problem really wants planning. trying to see if PPO can learn something similar"

### The Curvature Breakthrough

After months of raw future plan inputs capping at 130-150 cost, the user discovered:
> "dude it fucking worked. the cost was getting capped at 130, 150, the moment i fed curvatures in one dimension, straight down to 75. beautiful. just by pre training on PID"

The curvature encoding: `κ_t = (lataccel_t - roll_lataccel_t) / (v_ego_t² + ε)`

This compressed 4×50 future plan into 49 curvatures + 4 scalars (error, integral, diff, current_curvature) = 53 dims total.

### Why PPO Kept Destroying BC

Multiple runs showed the same pattern: BC gives ~69-75, PPO fine-tuning immediately worsens to 80, 100, 150+

Root causes identified:
1. **Reward scale**: per-step rewards in the -1e5 to -1e6 range, no advantage normalization
2. **Exploration too aggressive**: action_std=0.5 with a controller that needs surgical precision
3. **No trust region**: PPO clip alone insufficient, needed KL brake
4. **Missing advantage normalization**: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)` was missing

The diagnostic experiment: freeze actor, train critic only → cost stays stable at ~74. Proves PPO plumbing is correct, actor updates are the problem.

After fixes (advantage norm, reward scaling, frozen actor warmup), PPO stays in 71-78 range — no longer destroying BC, but also not improving.

### The a_ego Question (unresolved at time of export)

The user's key unresolved question:
> "a_ego seems it needs to be used (both current and future_plan)"

Physics reasoning:
- Curvature κ = (lataccel - roll) / v² encodes geometry but not timing
- a_ego = dv/dt shapes how quickly curvature "arrives"
- The ONNX simulator explicitly uses a_ego as input
- "Curvature tells you where the road bends. a_ego tells you when the bend arrives."
- Without a_ego: "Your MLP is basically conducting music without hearing the tempo changes"

The friction circle idea: `usage = sqrt((a_lat - a_roll)² + a_ego²) / (μ*g)` — combines lateral and longitudinal demand into one "difficulty" scalar.

### The NNFF Connection

The user found openpilot's NNFF (Neural Network FeedForward) writeup and recognized the structural parallel:
- NNFF decomposes steering into: error response + lateral jerk response + road roll response + roll rate response
- Each is a separate, interpretable response surface
- Key insight: "NNFF never feeds raw futures. It feeds structured temporal probes at specific horizons."
- NNFF inputs: `lateral_accel_m03, lateral_accel_p03, lateral_accel_p06, lateral_accel_p10, lateral_accel_p15`
- "Your PPO currently sees a 50-D bag of correlated junk. NNFF sees 6-8 orthogonal physical questions."

### The "MLP Can Read Sequences" Discussion

Critical architecture question: can MLP read a 1D curvature sequence concatenated with scalars?
> "an MLP doesn't 'see' a sequence vs scalars — it sees one flat vector"
> "whatever you want it to 'see cheaply' should be aligned with that linear structure"

The user's frustration:
> "are you fucking dumb, we cant input tuples in MLPs asshole you fucking illiterate"
> "WE CANNOT FUCKING INPUT FLATTENED SHAPES THAT DESTROYS INFER BIAS YOU PIECE OF JUNK"

The resolution: vertical (per-timestep) grouping preserves local physics, horizontal (per-channel) grouping preserves trends. For bicycle model, vertical is more natural.

### The Residual PPO Experiments (exp032, exp033)

**exp032 (Residual with error)**: `u = PID(error) + ε × π(state including error)`
- Result: cost 75,000 — 1000× worse than PID!
- Root cause: policy correlated with error → destabilized PID feedback loop
- "The nullspace is functions of future curvature that have zero projection onto present error dynamics"

**exp033 (Pure feedforward)**: `u = PID(error) + ε × π(preview only, NO error)`
- Removed error, integral, diff from MLP input
- Enforced `Cov(π, error) = 0` by construction
- Result: the 3-parameter linear preview test showed no improvement over PID

### The PPO-from-Scratch Attempts

Multiple runs with different configs:
- Best from scratch: ~178 total cost (after 3000 episodes)
- BC+curvature: reliably 69-75
- PPO fine-tuning BC: drifts to 75-175, never beats BC
- "PPO from scratch cannot even climb to the BC+curvature basin"

### Key Quotes from the ChatGPT Journey

On the problem structure:
> "This is not 'just control a car'. It is: Given advance access to the simulator's exogenous inputs, choose a steering sequence so that the simulator's autoregressive outputs match one of those inputs."

On what the winner did:
> "The leaderboard is screaming: the policy gets you to 45, the ONNX model gets you to 36. Both are needed."

On the philosophical method:
> "nah dont agree with me. we are scientists exploring the nature of the world. we need concreteness in our understanding, everything is secondary"
> "i dont want you to one shot stuff. the goal is discussion and understanding"
> "dont go for the easy shit. give me some insights physics and control theory. anyone can say decrease the learning rate you fool. you almost insult my 18months of hardwork by saying shit like this"

On giving up:
> "WRONG. im no scientist and these are no experiments. im just hacking stuff. the winner got <45 with PPO. there has to be a way."
> "i strongly believe MLP will work. no recurrence is required. the future plan x50 steps will make the control anticipatory automatically."

### Timeline Reconstruction (from ChatGPT transcript)

The ChatGPT conversations reveal a timeline that fills the "37 days of silence" gap:

1. **~Dec 2025**: Fork repo, initial PID experiments, BC attempts
2. **~Jan 2026**: MPC experiments (simple, CEM) — all fail to beat PID
3. **~Jan-Feb 2026**: LunarLander PPO work — deliberate isolation of PPO skills
4. **~Early Feb**: Harald posts 35 score → emotional crisis → tweet → back to work
5. **~Feb 9**: Sigma/state-dependent σ debugging (Cursor transcript)
6. **~Feb 11**: Curvature encoding breakthrough (130→75 in one change)
7. **~Feb 11**: Batched rollouts conceived ("WRITE THE FUCKING CODE")
8. **Feb 12-15**: The sprint to 42.5 (see git log)

### What the ChatGPT Couldn't Solve

The ChatGPT conversations eventually hit a wall:
- PPO fine-tuning BC never beat BC
- a_ego integration was discussed but never implemented via ChatGPT
- The "junior AI" (Cursor) was brought in for implementation
- ChatGPT's role shifted to philosophy/discussion; Cursor took over execution
- The breakthrough came from the Cursor sessions: exp048 (delta actions), exp050 (rich physics obs), batched simulator, GPU optimization

### The Narrative Arc (complete)

```
Phase 1: Understanding (Dec 2025)
  CTF-style README dissection → "this is preview control"
  PID baseline → ~85
  Feedforward experiment → 83
  MPC attempts → fail (wrong action space)

Phase 2: Isolation (Jan 2026)
  "Too many variants" → solve LunarLander first
  Beautiful_lander.py → sub-500 lines, reliable PPO
  Blog post published: utkarshgill.github.io/blog/lander.html

Phase 3: Breakthrough (Feb 2026)
  Curvature encoding → 130→75 in one change
  BC on PID with curvature → 69-75
  PPO can't improve BC → extensive debugging
  a_ego identified as missing piece (never resolved via ChatGPT)

Phase 4: The Sprint (Feb 9-15, via Cursor)
  Delta actions (exp048) → prevents exploration cascade
  Rich physics obs (exp050) → 256D, Beta distribution
  Batched simulator → 375x speedup
  MacBook cluster → 3x data
  Cloud GPU → TensorRT
  MPC + Policy hybrid → 2-4 point boost
  Final: 42.5 on 5000 routes (#3 leaderboard)
```
