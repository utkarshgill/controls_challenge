# exp021 PPO Findings

## Problem: Standard PPO Fails

### Observations
1. **BC baseline**: 74.88 cost (deterministic)
2. **With exploration noise (σ=0.0067)**: 87.61 cost (+17% degradation)
3. **After PPO updates**: 76.4+ cost (policy degrading, not improving)

### Why PPO Fails
- Control task is extremely sensitive to perturbations
- Even tiny exploration noise (σ=0.0067) causes 17% performance drop
- Exploration trajectories have costs 100-144 (vs 75 target)
- These failed trajectories provide **harmful gradients** that degrade the policy
- BC regularization either:
  - Too strong (λ=0.1): Policy frozen at BC, no learning
  - Too weak (λ=0.001): Policy degrading from bad exploration

### Key Insight
**On-policy RL with exploration doesn't work when:**
1. The task has tight tolerances
2. Exploration breaks the system completely  
3. Failed trajectories dominate the training signal

##Alternative Approaches to Consider

1. **Offline RL / Conservative Q-Learning**
   - Train only on BC trajectories (no new exploration)
   - Use value functions to extrapolate improvements
   
2. **Model-Based RL**
   - Learn dynamics model first
   - Plan in model space without real exploration
   
3. **Evolutionary Strategies / CMA-ES**
   - No gradients, direct parameter search
   - Evaluate full episodes, select best
   
4. **Behavior Cloning with Better Teacher**
   - Current teacher (PID + rich state) → 75 cost
   - Need better teacher that actually uses curvature info
   
5. **Supervised Fine-tuning on Optimal Trajectories**
   - Generate optimal trajectories offline (MPC/iLQR)
   - Clone those instead of PID

## Winner's Recipe (<45 cost)
The winner claimed to use PPO. Possible explanations:
- Different state representation that's less sensitive
- Curriculum learning (gradually increase difficulty)  
- Offline RL (train on demonstrations, not live rollouts)
- Heavy reward shaping
- **OR**: They didn't actually use PPO, just listed it as method

## Current Status
- BC achieves 74.88 cost
- PPO degradesit to 76.4+
- Need fundamentally different approach to break through BC ceiling



