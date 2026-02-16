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

- 256-dim observation: target/current lataccel, curvature, vehicle state, 20-step action/lataccel history, 50-step future plan
- 4-layer 256-wide actor (Beta distribution) + 4-layer critic
- Delta actions: `action = prev_action + clip(raw * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)`
- Trained with PPO (gamma=0.95, lambda=0.9, clip=0.2)

## MPC Shooting (optional)

The controller includes an N-step MPC shooting mode that samples candidate trajectories
from the policy, rolls them forward through the ONNX physics model, and picks the
lowest-cost action. This shaves ~2-3 points off total cost but makes eval ~50x slower
(~3s/seg vs ~0.1s/seg).

```bash
# Default: pure policy (fast, ~6 min for 5000 segs)
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller exp050_rich --baseline_controller pid

# With MPC shooting (~1 hour on a 48-core C4D instance for 5000 segs, but ~2 points better)
MPC=1 MPC_H=5 MPC_N=16 MPC_ROLL=1 python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 5000 --test_controller exp050_rich --baseline_controller pid
```

The tradeoff: the policy alone is strong enough to be competitive. MPC acts as a
refinement layer — it uses the same policy as a proposal distribution but validates
candidates against the actual physics model before committing. On GPU (`CUDA=1`),
the MPC path is batched and runs significantly faster.
