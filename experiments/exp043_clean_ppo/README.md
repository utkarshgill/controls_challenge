# Clean PPO for Controls Challenge

## Two-Stage Training

### Stage 1: Behavioral Cloning (5-10 minutes)
```bash
python train_bc.py
```

This will:
- Collect PID demonstrations from 1000 CSVs
- Train actor network to imitate PID (10 epochs, intentionally undertrained)
- Save `bc_init.pt` checkpoint (~250 cost expected)

### Stage 2: PPO Fine-Tuning (45 minutes)
```bash
python train.py
```

This will:
- Auto-detect and load `bc_init.pt` if it exists
- Fine-tune with PPO from BC starting point
- Target: <100 cost

## Architecture

- **Input**: 57 dims (7 current + 50 future curvatures)
- **Network**: Simple MLP (57 → 128 → 128 → 1)
- **No Conv1D**: Removed for speed and simplicity

## Key Features

- ✅ Physics-based curvature features
- ✅ Minibatched PPO updates (~1s per epoch)
- ✅ Parallel rollout collection (~25s per epoch)
- ✅ Reward scaling for stability
- ✅ Warm-start from BC (optional)

## Performance

- **From scratch**: ~1200 cost after 15 epochs (plateaus)
- **From BC (expected)**: ~100-250 cost after 50 epochs

## Why BC → PPO?

Starting from random: 4000 → 100 (40x improvement, very hard!)
Starting from BC: 250 → 100 (2.5x improvement, feasible!)
