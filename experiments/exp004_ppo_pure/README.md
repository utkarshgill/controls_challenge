# Experiment 004: Pure PPO (No BC)

**Status**: 🏃 Ready to run  
**Date**: 2024-12-26

---

## Hypothesis

Skip BC entirely. Train PPO from scratch using battle-tested architecture from `beautiful_lander.py`.

**Why this should work:**
- Winner achieved < 45 with pure PPO
- BC is fundamentally limited by PID (ceiling ~85)
- PPO can explore beyond expert demonstrations

## Architecture

```python
State: 55D normalized
├── Scalars (5): error, roll, v_ego, a_ego, current
└── Future (50): upcoming lataccel trajectory

Network: ActorCritic (from beautiful_lander.py)
├── Shared trunk: state → hidden
├── Actor head: trunk → action_mean, log_std
└── Critic head: trunk → value

Action: tanh(sample(mean, std)) × 2.0
Log_prob: Gaussian - log(1 - tanh²)  # Change of variables
```

## Key Features

1. **Proper PPO**:
   - GAE for advantage estimation
   - Clipped surrogate objective
   - Entropy bonus for exploration
   - Gradient clipping

2. **Parallel training**: 8 AsyncVectorEnv

3. **Normalization**: Same as exp003 (v_ego scaled down)

4. **No BC dependency**: Train from random init

## Hyperparameters

```
lr = 1e-3
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
entropy_coef = 0.001
K_epochs = 10
batch_size = 10,000
num_envs = 8
steps_per_epoch = 10,000
```

All matched to `beautiful_lander.py` (proven to work).

## Run

```bash
cd experiments/exp004_ppo_pure
source ../../.venv/bin/activate
python train.py
```

Training: ~30-60 min for 100 epochs

## Evaluation

```bash
cd /Users/engelbart/Desktop/stuff/controls_challenge
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller ppo
```

## Expected Results

```
Target:    45.0
PID:       84.85
exp002:   497 (broken PPO, random init)
exp003:  4782 (broken BC)
exp004:    ?? (clean PPO)
```

**Goal**: < 85 (beat PID), ideally < 45 (win)

## Results

*TBD after training*

---

## Notes

- Skipped BC entirely (kept failing)
- Used proven architecture (beautiful_lander.py)
- Clean implementation, no hacks
- If this fails, problem is elsewhere (state design, reward, etc.)

