# Experiment 040: BC + PPO PID Tuning

## Hypothesis

Can PPO fine-tune PID gains to beat hand-tuned values?

## Approach

**Phase 1: Behavioral Cloning**
- Single neuron: 3 weights, no bias
- Input: `[error, error_integral, error_diff]`
- Output: `steering action`
- Train on PID demonstrations to recover: `P=0.195, I=0.100, D=-0.053`

**Phase 2: PPO Fine-Tuning**
- Start from BC weights (≈ PID)
- Use PPO to optimize directly on cost function
- Let gradient descent discover better gains

## Why This Might Work

1. **PID is hand-tuned** - not optimized for this specific cost function
2. **Simple search space** - only 3 parameters to tune
3. **Good initialization** - start from working controller
4. **Direct optimization** - PPO sees actual cost, not demos

## Expected Results

- **BC baseline:** ~75 cost (matching PID)
- **PPO improvement:** Could reach 60-70 if gains can be improved
- **Learned weights:** Should deviate from PID if better gains exist

## What We Learn

If PPO **beats PID:**
- Hand-tuned gains aren't optimal
- Proves RL can improve classical control

If PPO **equals PID:**
- Either PID is already optimal for this cost
- Or search space too small (need more features)

If PPO **degrades:**
- Training instability
- Need different approach (feedforward, not just gain tuning)

## Running

```bash
cd experiments/exp040_pid_tuning
python train.py
```

## Key Difference from exp039

**exp039:** Learn feedforward from future plan (big network)
**exp040:** Just tune 3 PID gains (simplest possible)

This tests: "Is the problem in PID structure or just the gains?"

