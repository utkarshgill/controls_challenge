# Experiment 039: PID + Learned Feedforward

## Hypothesis

PID controller is reactive - always behind due to rate limits (0.5 m/s² per 0.1s).
Winner got <45 cost by adding anticipatory control using future plan preview.

**Approach:** Keep PID frozen (proven stable), learn only feedforward term via PPO.

## Architecture

```
Total Action = PID(error) + FF(future_plan)
              [fixed]      [learned via PPO]
```

### Feedforward Network
- Input: 4-channel 1D signal over 50 timesteps
  - `lataccel[50]` - target lateral acceleration
  - `v_ego[50]` - velocity (affects responsiveness)
  - `a_ego[50]` - forward acceleration (weight transfer)
  - `roll_lataccel[50]` - road banking
- Architecture: 3-layer 1D CNN → MLP
  - Conv learns temporal patterns in future trajectory
  - MLP maps to single FF action
- Output: FF steering ∈ [-1, 1] (tanh bounded)

### PID Controller (Fixed)
```python
Kp = 0.195
Ki = 0.100  
Kd = -0.053
action = Kp*error + Ki*∫error + Kd*derror/dt
```

## Training

**PPO setup (from beautiful_lander.py):**
- Dual models: CPU for rollout, device for update
- 16 parallel environments
- 100k steps per epoch
- Batch size: 5000, K_epochs: 20
- γ=0.99, λ=0.95, ε=0.2

**Initialization:**
- FF network: small random weights → outputs ≈ 0
- Starts at PID performance (~75 cost)
- PPO explores improvements from there

**Reward:**
- Negative step cost (shaped by GAE)
- Final cost computed at episode end
- Optimization directly matches cost function

## Why This Should Work

1. **Safe exploration:** FF=0 → pure PID (stable baseline)
2. **Clear signal:** FF helps → cost drops → positive reward
3. **Smaller problem:** Only learning FF (not full control)
4. **Physics-informed:** Future plan contains bicycle model trajectory
5. **Battle-tested:** PPO structure proven on LunarLander

## Expected Results

- **Baseline (PID):** ~75 cost
- **Target:** <60 cost (20% improvement)
- **Winner achieved:** <45 cost

If this works, FF network should learn:
- Preview ~1.0s ahead (rate limit = 0.5 m/s² per 0.1s)
- Speed-adaptive gains (lighter at high speed)
- Smooth ramping (avoid fighting PID)

## Running

```bash
cd experiments/exp039_pid_plus_ff
python train.py

# With GPU/MPS
METAL=1 python train.py

# Adjust parallelism
NUM_ENVS=24 python train.py
```

## Files

- `train.py` - Main training script
- `best_model.pth` - Best model checkpoint (created during training)
- `README.md` - This file

## Next Steps

If this beats PID:
1. Analyze learned FF: which horizons matter most?
2. Try bicycle model initialization for FF
3. Fine-tune on test set
4. Submit to leaderboard

If this doesn't beat PID:
1. Check if FF is being used (monitor FF action magnitude)
2. Try different reward shaping
3. Consider BC initialization for FF
4. Revisit feature engineering

