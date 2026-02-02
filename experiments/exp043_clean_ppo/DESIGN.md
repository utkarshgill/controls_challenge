# Clean PPO Design - Scientific Approach

## Environment Comparison

### LunarLander (beautiful_lander.py)
- **State**: 8D continuous (position, velocity, angle, etc.)
- **Action**: 2D continuous (main engine, side engine)
- **Episode**: Variable length, ends on crash/landing
- **Parallelization**: 24 envs running simultaneously
- **Reward**: Dense, provided by environment every step
- **Success**: Mean reward > 250

### Controls Challenge
- **State**: 55D continuous (lataccel, errors, future plan, etc.)
- **Action**: 1D continuous (steer angle)
- **Episode**: Fixed length (~1000 steps per CSV file)
- **Parallelization**: None - one CSV at a time
- **Reward**: Must derive from cost (lataccel penalty + jerk penalty)
- **Success**: Mean cost < 45 (PID baseline: 126)

## Key Components from beautiful_lander.py

1. **ActorCritic Network**
   - Actor: state → action_mean, log_std (learnable)
   - Critic: state → value
   - Both use ReLU, multiple layers

2. **PPO Update**
   - GAE for advantage estimation
   - Clipped surrogate objective
   - Separate optimizers (pi_lr=3e-4, vf_lr=1e-3)
   - Gradient clipping (max_norm=0.5)
   - Multiple epochs per batch (K_epochs=20)

3. **Action Distribution**
   - Gaussian with tanh squashing
   - Log prob correction for tanh

4. **Training Loop**
   - Collect batch of experience
   - Update policy K times
   - Repeat

## Adaptation Strategy

### Phase 1: Verify BC Baseline
- [ ] Clean BC implementation
- [ ] Verify it gets ~240 cost
- [ ] This is our "warm start"

### Phase 2: Minimal PPO (No Warm Start)
- [ ] ActorCritic matching beautiful_lander structure
- [ ] PPO class with exact same hyperparameters
- [ ] Single CSV rollout collection
- [ ] Verify mechanics work (loss decreases)

### Phase 3: PPO from BC Warm Start
- [ ] Load BC weights into actor
- [ ] Train critic from scratch OR pre-train
- [ ] Compare to BC baseline

### Phase 4: Hyperparameter Search
- [ ] Adjust based on results
- [ ] Focus on: std initialization, learning rates, K_epochs

## Critical Questions to Answer

1. **Reward Shaping**: How to convert cost → reward?
   - Option A: reward = -cost (simple)
   - Option B: Dense per-step rewards
   - Option C: Shaped reward (bonus for staying close)

2. **Parallelization**: Can we run multiple CSVs in parallel?
   - Probably not initially (TinyPhysicsSimulator limitation)
   - Start with sequential, optimize later

3. **Value Normalization**: Do we need it?
   - beautiful_lander.py normalizes advantages but not values
   - Start without, add if needed

4. **Episode Length**: All CSVs same length?
   - No - varies ~500-1500 steps
   - GAE should handle this naturally

## Success Metrics

- **Epoch 0**: Untrained PPO should be terrible (~10,000+ cost)
- **Epoch 10**: Should see improvement over random
- **Epoch 50**: Should approach BC performance (240)
- **Epoch 100+**: Should beat BC, target <126 (PID)
- **Final Goal**: <45 cost (winner's performance)
