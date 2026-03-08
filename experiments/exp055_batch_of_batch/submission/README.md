exp055 is exp050 with the bugs fixed and the cruft removed.

same architecture (256-dim obs, 4-layer beta actor, delta actions, PPO) but less lines.

scores:
- total_cost = 42.206 (5000 segs, official eval)
- total_cost = 40.69 (100 segs, batch metrics)

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
- variance reduction: `samples_per_route` rolls out the same route multiple
  times per epoch and centers advantages within-route before PPO updates

what stayed:
- 256-dim obs = 16 core + 20 h_act + 20 h_lat + 4x50 future
- beta distribution, delta actions (action = prev + clip(raw * delta_scale))
- bc pretrain then ppo fine-tune on tinyphysics_batched.py
- sigma floor penalty, cosine lr decay, reward rms norm, gae

usage:
  unzip archive and copy both controller and weights to controllers/

  then run:
  ```bash
  MODEL=controllers/weights.pt python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller exp055_controller
  ```

trained with: `TRT=1 CUDA=1 CSVS=5000 EPOCHS=5000 python train.py` on a single 5090 for ~10.41 hours (7-8s per epoch) @ $0.301/hr = ~$3.14

included files:
- `exp055_controller.py`
- `README.md`
- `report.html`
- `train.py`
- `weights.pt`
