# Experiment 014 V2: PPO with Official Simulator

## What Changed

**V1 (train_ppo.py)**: Custom environment wrapper
- ❌ Didn't match official evaluation (cost 1036 vs 90)
- ❌ Hours of debugging mismatches

**V2 (train_ppo_v2.py)**: Uses official `TinyPhysicsSimulator` directly
- ✅ Perfect match with evaluation  
- ✅ BC stochastic: ~168, BC deterministic: ~113, Official eval: ~90
- ✅ Clean architecture

## Architecture

```
PPOController (wraps policy)
    ↓
TinyPhysicsSimulator.rollout()  ← Official simulator!
    ↓
Collect trajectories
    ↓
PPO update
```

### Key Innovation

Instead of building a custom gym environment, we:
1. Wrap our policy as a `Controller` class
2. Use official `sim.rollout()` for data collection
3. Extract trajectories from the rollout
4. Do PPO updates on collected data

**Result**: Environment matches evaluation EXACTLY!

## How It Works

### 1. PPOController Class

```python
class PPOController:
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build state vector (same as BC)
        # Get action from policy
        # Track trajectory data
        return action
```

Implements the standard Controller interface, so it works with `TinyPhysicsSimulator`.

### 2. Collect Rollouts

```python
controller = PPOController(actor_critic, state_mean, state_std, collect_data=True)
sim = TinyPhysicsSimulator(model, data_file, controller=controller)
cost_dict = sim.rollout()  # Official rollout!
trajectory = controller.get_trajectory()  # Extract data for PPO
```

### 3. PPO Update

```python
ppo.update(trajectories, batch_size, num_epochs)
```

Standard PPO on collected trajectories.

## Expected Performance

### BC Baseline
- Deterministic: ~113 (route 00000), ~90 (100 routes average)
- Stochastic (PPO training): ~168 (route 00000)

### PPO Training
- Start: ~168 (stochastic BC)
- Target: < 113 (beat deterministic BC)
- Stretch: < 90 (beat BC average)

## Running

```bash
cd experiments/exp014_ppo_finetune
source ../../.venv/bin/activate

# Train
python train_ppo_v2.py

# Expected output:
# Iter 0 | Cost: ~168 (stochastic BC baseline)
# Iter 10 | Cost: ~150 (learning)
# Iter 50 | Cost: ~120 (improving)
# Iter 200 | Cost: < 110 (target)
```

## Files

- `train_ppo_v2.py` - Main training script (uses official simulator)
- `train_ppo.py` - V1 (custom env, has bugs)
- `ppo_best_v2.pth` - Best checkpoint
- `ppo_final_v2.pth` - Final checkpoint

## Evaluation

After training, create controller file:

```python
# controllers/ppo_exp014_v2.py
class Controller:
    def __init__(self):
        # Load ppo_best_v2.pth
        # Initialize actor-critic
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get action (deterministic)
        return action
```

Then evaluate:

```bash
cd ../..
python eval.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data --num_segs 100 \
  --test_controller ppo_exp014_v2 --baseline_controller pid
```

## Why This Works

**Shannon's Principle**: "Don't fight the system, work with it"

Instead of trying to recreate the simulator's behavior:
- ✅ Use the actual simulator
- ✅ Wrap our policy to work with it
- ✅ Guaranteed perfect match

**Benefits**:
1. No environment bugs
2. Costs match evaluation exactly
3. Cleaner code
4. Faster to implement than debugging custom env

## Lessons Learned

1. **Use official tools when possible** - Don't recreate what exists
2. **Validate early** - Test BC in env before building PPO
3. **Keep it simple** - Controller wrapper is simpler than gym env
4. **Trust the process** - When debugging takes too long, rethink approach

This is the Shannon way: elegant, simple, and it works! 🎯



