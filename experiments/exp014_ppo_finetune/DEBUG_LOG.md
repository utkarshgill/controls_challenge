# Debug Log - PPOController vs BC Mismatch

## Problem
PPOController (deterministic=True) gets **149.03**  
BC official gets **100.72**  
Gap: **48.31 points**

## What We've Verified

### ✅ Networks Are Identical
- BCNetwork and ActorCritic produce **identical outputs** for same input
- Weights load correctly (verified by comparing specific weight values)
- Forward pass matches exactly

### ✅ Simulator Is Deterministic
- Running BC controller 3x produces exact same cost (100.716330)
- ONNX model is deterministic

### ✅ Actions Match Initially  
- Step 0-81: Actions match **exactly** (0 difference)
- Step 82: Actions suddenly diverge

### ❌ State Diverges At Step 82
- Official `current_lataccel` at step 82: `-0.034213`
- My `current_lataccel` at step 82: `0.102639` (when testing separately)
- BUT when running in same script, they match!

## Current Status
- Fixed: Added `deterministic` parameter to PPOController
- Fixed: Use `deterministic=True` in evaluation
- NOT Fixed: Still getting 149.03 instead of 100.72

## Next Steps
1. Create a standalone controller that uses ActorCritic with EXACT same code as bc_exp013.py
2. If that works, diff it line-by-line with PPOController.update()
3. Find the subtle difference causing 48-point gap

## Hypothesis
There's a subtle difference in PPOController.update() that we haven't spotted yet, possibly:
- Array dtype handling?
- Tensor device mismatch?
- Some state not being reset properly?



