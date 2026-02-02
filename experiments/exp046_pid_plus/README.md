# Experiment 046: PID Plus

## Strategy

**Start from what works, grow incrementally.**

### Phase 1: Baseline (CURRENT)
- Pure PID from `controllers/pid.py`
- Expected cost: ~103
- Goal: Verify it works

### Phase 2: Diagnosis
Analyze WHERE PID loses points:
- Which sections of trajectory have high error?
- When does jerk spike?
- Can we identify patterns (sharp turns, oscillations)?

### Phase 3: Targeted Improvements
Add features ONE AT A TIME:
1. **Feedforward:** Use `future_plan.lataccel[0]` to anticipate
2. **Gain scheduling:** Adapt P/I/D based on conditions
3. **Jerk penalty:** Smooth out action changes
4. **1-step lookahead:** Check if next action improves (simple MPC)
5. **Multi-step if needed:** Extend horizon only if 1-step works

### Rules
1. **Each change must be justified by diagnosis**
2. **Test after every change** (no blind modifications)
3. **If cost increases, revert immediately**
4. **Document what works and what doesn't**

## Progress Log

### v0: Pure PID Baseline
- **Single trajectory (00000.csv):** 102.87
- **Batch average (100 routes):** 84.85
- **CORRECT baseline:** 84.85 ← This is what matters!
- Status: PID is ALREADY competitive (<90)
- Challenge: Beat 84.85 to improve on PID
