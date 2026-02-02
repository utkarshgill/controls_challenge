# Experiment 017: Baseline - Proof of Concept

## Concept
**The simplest possible neural controller that works:**
- 1 neuron (3 weights, no bias)
- Input: [error, error_integral, error_diff]
- Trained on 100 PID demonstrations
- Recovers exact PID coefficients

## Results
Learned weights match PID perfectly:
- P = 0.195 (PID: 0.195) ✓
- I = 0.100 (PID: 0.100) ✓
- D = -0.053 (PID: -0.053) ✓

## Why This Matters
**Proof that neural networks can learn control from demonstrations.**

This is the foundation. From here we grow.



