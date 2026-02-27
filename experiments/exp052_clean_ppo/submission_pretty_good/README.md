exp052 is exp050 with the bugs fixed and the cruft removed.

same architecture (256-dim obs, 4-layer beta actor, delta actions, PPO) but cleaned up.
controller is 150 loc and train.py is 652 lines now (down from 1233 in exp050).

scores:
- total_cost = 43.729 (5000 segs, official eval)
- total_cost = 41.37 (100 segs, batch metrics)

purged:
- mpc shooting (N-step, single-step correction, gpu path, cpu path)
- colored noise (AR(1) logit perturbation)
- low-pass filter, rate limiter, speed normalization
- value function clipping, huber loss
- multi-remote execution stack (REMOTE_HOSTS/PORTS, FRAC splits, tcp workers)
- backend branching (CUDA/BATCHED/unbatched simulator paths)

fixed:
- reward temporal alignment: rewards now use post-step simulator histories,
  so the policy gets credit for the action that caused the outcome
- jerk/action-rate boundary: torch.diff(prepend=) keeps finite differences
  strictly inside the official cost window

what stayed:
- 256-dim obs = 16 core + 20 h_act + 20 h_lat + 4x50 future
- beta distribution, delta actions (action = prev + clip(raw * delta_scale))
- bc pretrain then ppo fine-tune on tinyphysics_batched.py
- sigma floor penalty, cosine lr decay, reward rms norm, gae

usage:
  unzip archive and copy both controller and weights to controllers/

  then run:
  ```bash
  MODEL=controllers/weights.pt python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller exp052_clean
  ```
note:
training script still has some experimental stuff. to be cleaned up for a single unbroken run with a fixed schedule next.
