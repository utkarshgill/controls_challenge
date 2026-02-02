# V3 FIXES - Aggressive Attack on <45 Cost

## What We Changed

### 1. REWARD SCALING ⚡
**Before:** `reward = -step_cost / 10000.0` → tiny rewards like -0.05  
**After:** `reward = -step_cost / 100.0` → reasonable rewards like -5.0 to -50.0  
**Why:** PPO can't learn from microscopic signals

### 2. REWARD RANGE 🎯
**Before:** Computed rewards for ALL steps (0-580)  
**After:** Only steps in [100:500] (official cost range)  
**Why:** Match exactly what we're evaluated on

### 3. LEARNING RATE 📉
**Before:** 3e-4 (too aggressive)  
**After:** 1e-4 (gentle fine-tuning)  
**Why:** Starting from GOOD policy (BC=100), don't break it

### 4. PPO EPOCHS ⏱️
**Before:** 10 epochs per batch (overfitting risk)  
**After:** 4 epochs  
**Why:** Prevent overfitting to individual batches

### 5. ENTROPY BONUS 🎲
**Before:** 0.01 (encourages exploration)  
**After:** 0.0 (NO exploration)  
**Why:** We WANT to stay near BC, just fine-tune

### 6. LOGGING 📊
**Before:** Minimal output  
**After:** Show min/max/mean costs per iteration  
**Why:** See what's actually happening

## Expected Outcome

**Iteration 0:** ~100-120 (should match BC baseline)  
**Iteration 10:** <90 (modest improvement)  
**Iteration 50:** <70 (good progress)  
**Iteration 100:** <50 (getting close!)  
**Iteration 200:** **<45 (TARGET!)** 🎯

## If This Still Doesn't Work

Next attempts:
1. Even lower LR (5e-5)
2. Smaller clip epsilon (0.1 instead of 0.2)
3. Different reward function (exponential penalty)
4. Add curriculum learning (easy routes first)
5. Try TRPO instead of PPO (more stable)

## Training Started
Check `train_v3.log` for real-time progress

**WE'RE NOT STOPPING UNTIL <45!**



