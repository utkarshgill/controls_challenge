# Submission: exp050_rich — Physics-Aligned PPO Controller

## Files

- `exp050_rich.py` — Controller (drop into `controllers/`)
- `best_model.pt` — Trained weights (~6MB)
- `report.html` — Generated after running eval (see below)

## Setup

```bash
# Clone the official repo
git clone https://github.com/commaai/controls_challenge
cd controls_challenge
pip install -r requirements.txt

# Copy controller + weights
cp submission/exp050_rich.py controllers/
cp submission/best_model.pt controllers/
# OR: cp submission/best_model.pt experiments/exp050_rich_ppo/best_model.pt

# Run official eval (5000 segments)
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller exp050_rich --baseline_controller pid
```

## Architecture

- 256-dim observation: 16 core signals + 20-step action history + 20-step lataccel history + flat 4x50-step future plan
- 4-layer 256-wide actor (Beta distribution) + 4-layer critic
- Delta actions: `action = prev_action + clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)`
- Trained with PPO (gamma=0.95, lambda=0.9, clip=0.2)

## MPC Shooting (optional)

The controller includes an N-step MPC shooting mode for test time that samples candidate trajectories
from the policy, rolls them forward through the ONNX physics model, and picks the
lowest-cost action. This shaves ~1-2 points off total cost but makes eval ~50x slower
(~3s/seg vs ~0.1s/seg).

The policy alone is strong enough to be competitive. MPC acts as a
refinement layer. It uses the same policy as a proposal distribution but validates
candidates against the actual physics model before committing. The rollouts take time. On GPU (`CUDA=1`),
the MPC path is batched and runs significantly faster.

```bash
# total_cost = 45.552 (report_pure_ppo.html) with pure policy (fast, ~2 min for 5000 segs) 
ugill@ppo-worker:~/controls_challenge$ MODEL=best_model_43.pt .venv/bin/python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller exp050_rich --baseline_controller pid
Running rollouts for visualizations...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.41it/s]
Running batch rollouts => baseline controller: pid
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4995/4995 [01:22<00:00, 60.26it/s]
Running batch rollouts => test controller: exp050_rich
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4995/4995 [01:52<00:00, 44.51it/s]
Report saved to: './report.html'


# total_cost = 44.834 (report_ppo_with_mpc.html) with MPC shooting (~56 min on a 48-core C4D instance for 5000 segs, but ~1 point better)
ugill@ppo-worker:~/controls_challenge$ MODEL=best_model_43.pt MPC=1 MPC_H=5 MPC_ROLL=1 MPC_N=16 .venv/bin/python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller exp0
50_rich --baseline_controller pid
Running rollouts for visualizations...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:54<00:00, 10.85s/it]
Running batch rollouts => baseline controller: pid
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4995/4995 [01:22<00:00, 60.23it/s]
Running batch rollouts => test controller: exp050_rich
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4995/4995 [56:21<00:00,  1.48it/s]
Report saved to: './report.html'

```

## Physics Go Brrrrr: `tinyphysics_batched.py`

I built `tinyphysics_batched.py` from scratch -- a batched simulator that runs N
episodes in lockstep with one ONNX call per timestep. Batched inference, IOBinding,
GPU-resident histories, TensorRT FP16, CSV caching, all-GPU controller loop.
Took 5000-episode rollouts from **50 minutes to 8 seconds** (~375x). The full
optimization story is in `BRRRRR.md`.


## Attempt at Fast Evaluation: `fast_eval.py`) - [not used due to score mismatch]

I also tried speeding up eval using `fast_eval.py` to run the full 5000-segment eval on GPU. 
The idea was to reimplement the exact controller + simulator logic as batched
tensor ops, prove bit-for-bit parity with `eval.py`, include it as a fast verifier.
I got close but never nailed perfect parity. The physics model amplifies tiny
floating-point differences across 580 steps, and matching every RNG consumption
path, float32/float64 cast, and tokenizer edge case between sequential and batched
execution turned into a rabbit hole. Scores are close (within ~0.1) but not exact, so I ended up not using it.

## Notes

The training was not clean. I kept stopping runs, changing stuff, resuming from
stable checkpoints. Should have locked down the config and done one long 
uninterrupted run. Instead I kept tweaking knobs the whole time.

The action smoothness penalty (`ACT_SMOOTH`) was only added during the final
fine-tuning phase, after the policy hit a floor around 42. The original reward
matched the official cost exactly (lataccel error + jerk). But jerk penalizes
changes in *actual lataccel* (simulator output), not changes in the *steering
action* (policy output). Adding `|d_action|^2 * coef` to the reward gave the
policy a more direct gradient for smoothness during the last stretch.

