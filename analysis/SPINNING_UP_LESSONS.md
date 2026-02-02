# Lessons from OpenAI Spinning Up

## Our PPO Problems vs Spinning Up Solutions

### Problem 1: Noisy Reward Signal
**What we did wrong:**
```python
# Collect with exploration
action_noisy = mean + std * noise
execute(action_noisy)
reward = -cost_from_noisy_execution  # Includes failure from noise!

# This tells policy: "Your mean action leads to bad outcomes"
# But it's the NOISE that caused the bad outcome!
```

**Spinning Up solution:**
- Collect trajectories with stochastic policy (for exploration)
- But rewards ARE from those stochastic actions
- The KEY: Use GAE to properly weight rewards by causality
- The advantage function accounts for "what would happen anyway" (baseline)

### Problem 2: Wrong Advantage Calculation
**What we did wrong:**
```python
# We used episode-end total cost as reward
# Or raw per-step costs without proper GAE
```

**Spinning Up solution:**
```python
# Reward-to-go (future rewards only)
rtg[t] = reward[t] + gamma * rtg[t+1]

# Then subtract baseline (value function)
advantage[t] = rtg[t] - V(state[t])

# Or use GAE (better):
delta[t] = reward[t] + gamma * V(state[t+1]) - V(state[t])
advantage[t] = sum of exponentially weighted deltas
```

### Problem 3: Critic Learning
**What we did wrong:**
```python
# Critic was untrained initially
# Or learned wrong scale of returns
# Returns were -125, critic predicted 0.1
```

**Spinning Up solution:**
```python
# Critic minimizes MSE to ACTUAL returns
critic_loss = (V(state) - reward_to_go)**2

# Train critic CONCURRENTLY with policy
# So it always tracks current policy's value function
```

### Problem 4: Dense vs Sparse Rewards
**What we did wrong:**
```python
# Started with sparse (episode-end only)
# Created huge variance in gradients
```

**Spinning Up solution:**
- Use DENSE rewards (every step)
- Let GAE handle temporal credit assignment
- This is why LunarLander works: reward every step

## The Correct PPO Structure

Based on Spinning Up + beautiful_lander.py:

```python
def train_one_epoch():
    # 1. COLLECT DATA with stochastic policy
    for step in range(steps_per_epoch):
        action_mean, action_std, value = policy(state)
        action = sample_from_normal(action_mean, action_std)  # Exploration
        
        next_state, reward, done = env.step(action)  # ACCEPT the stochastic outcome
        
        store(state, action, reward, value, log_prob)
    
    # 2. COMPUTE ADVANTAGES with GAE (on the stochastic data)
    advantages = compute_gae(rewards, values, dones)  # Reward-to-go with baseline
    returns = advantages + values  # What we actually got
    
    # 3. NORMALIZE advantages (reduce variance)
    advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
    
    # 4. MULTIPLE UPDATE EPOCHS on collected data
    for epoch in range(K_epochs):
        # Recompute log_probs and values with CURRENT policy
        new_log_probs, new_values = policy.evaluate(states, actions)
        
        # PPO clipped loss
        ratio = exp(new_log_probs - old_log_probs)
        clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
        policy_loss = -min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Critic loss (learn to predict returns)
        critic_loss = (new_values - returns)**2.mean()
        
        # Total loss
        loss = policy_loss + 0.5 * critic_loss - entropy_coef * entropy
        
        optimize(loss)
```

## Key Insights for Our Problem

1. **Accept stochastic outcomes**: Don't try to "fix" noisy actions
2. **Use GAE properly**: It handles reward attribution correctly
3. **Dense rewards**: Per-step cost (we have this)
4. **Train critic concurrently**: So it tracks current policy
5. **Multiple epochs**: Reuse collected data efficiently

## Why This Should Work for Controls

LunarLander:
- State: 8D vehicle properties
- Action: 2D continuous (thrust)
- Reward: Dense, per-step
- PPO: Works great (~250 return)

Our problem:
- State: 6D vehicle properties (current_lat, target_lat, v, a, roll, prev_action)
- Action: 1D continuous (steering)
- Reward: Dense, per-step (negative cost)
- PPO: Should work if we follow Spinning Up structure!

The winner got <45 (vs PID ~107) = 58% improvement.
This is HUGE but achievable if PPO is done right.

## Next Steps

1. Implement PPO exactly like beautiful_lander.py structure
2. Use vehicle-centric state (exp030)
3. Dense per-step rewards (negative cost)
4. Proper GAE (like beautiful_lander)
5. Train critic concurrently
6. Accept stochastic exploration outcomes (don't "correct" with mean action)

The key realization: Spinning Up's structure handles exploration correctly by:
- Using GAE to attribute rewards properly
- Training critic to provide good baselines
- Multiple epochs to learn efficiently from data

