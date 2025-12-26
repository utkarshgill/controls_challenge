# Experiment 001: BC with a_ego

## Hypothesis
Adding current `a_ego` (longitudinal acceleration) fixes catastrophic failures on high-braking files by enabling friction circle awareness.

## Changes
1. State: 56D → 57D (added `state.a_ego` at position 5)
2. OBS_SCALE: added `20.0` for a_ego normalization
3. Network: input_dim = 57

## Physics Reasoning
```
Friction circle: √(a_lat² + a_long²) ≤ 9.8 m/s²

When braking at -3.75 m/s²:
→ Only ~9.0 m/s² left for lateral
→ Same steering has different effect
→ BC needs to know a_ego to predict correctly
```

## Expected Results
- Mean: 92.4 → 85 (optimistic) or 88 (realistic)
- File 00069: 1401 → 500 (optimistic) or 700 (realistic)

## Run
```bash
cd experiments/exp001_bc_with_a_ego
python train.py      # ~15 min (1000 files, 30 epochs)
python evaluate.py   # ~2 min (100 files)
```

## Status
⏳ Ready to run

