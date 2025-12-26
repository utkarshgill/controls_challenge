# Project Status

**Goal**: Control cost < 45  
**PID baseline**: 101.74 ± (median 80.76) on 100 files
**Current best**: ~100 (BC)  
**Gap**: 2.2×

---

## Clean Structure ✅

```
Root (minimal!)
├── tinyphysics.py (simulator)
├── controllers/ (reusable)
├── data/ (shared)
└── experiments/ (all work here)

experiments/exp003_ppo_bc_init/
├── train.py           ← Full training code
├── run.sh             ← One command
├── README.md          ← Hypothesis & results
└── results/

Each experiment = self-contained snapshot
No shared training code to break things!
```

---

## Experiments Complete

### exp000: Baseline
- PID: 100
- BC: 100  
- PPO (broken): ~100

### exp001: BC with a_ego
- Failed (wrong approach - BC can't exceed PID)

### exp002: PPO with fixed hyperparameters
- Fixed reward, lr, exploration
- Result: 497 (5× worse than PID!)
- Unstable (exploded after epoch 10)
- **Lesson**: Random init is too chaotic

---

## Next: exp003

**Hypothesis**: BC initialization fixes instability

**Run**:
```bash
bash experiments/exp003_ppo_bc_init/run.sh
```

**Expected**:
- Start at ~100 (not 15k)
- Stable training
- Improve beyond BC
- Target: < 45

---

## Key Learnings

1. ✅ Fixed critical bugs (reward, lr, exploration)
2. ✅ PPO CAN learn (15k→500 proves it)
3. ❌ Random init unstable
4. 💡 BC proves network capacity
5. 💡 Need stable baseline to improve from

---

## If exp003 Fails

Try in order:
1. Add a_ego (physics-critical)
2. Reduce lr (1e-3 → 1e-4)
3. Simplify state (50 → 10 future steps)
4. Different architecture (LSTM/CNN)

---

## Repo Hygiene ✅

- Root is clean (library only)
- Each experiment self-contained
- One command to reproduce
- Results stay in experiment folder
- Pattern documented in `experiments/HOW_TO_RUN_EXPERIMENTS.md`
