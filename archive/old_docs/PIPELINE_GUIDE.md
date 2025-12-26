# Unified Training Pipeline Guide

## 📋 Overview

`train_pipeline.py` is the **SINGLE SOURCE OF TRUTH** for our training process.

It runs in two stages:
1. **BC (Behavioral Cloning)** - Learn to imitate PID controller
2. **PPO (Reinforcement Learning)** - Improve beyond BC using RL

## 🚀 Usage

```bash
# Run the full pipeline (BC → PPO)
python train_pipeline.py
```

This will:
1. Collect expert data from PID controller (1000 files)
2. Train BC for 50 epochs
3. Evaluate BC on 100 validation files
4. Save `bc_checkpoint.pth`
5. Initialize PPO with BC weights
6. Train PPO for 100 epochs with 8 parallel environments
7. Evaluate PPO on 100 validation files
8. Save `ppo_best.pth`

## 🔧 Configuration

All hyperparameters are at the top of `train_pipeline.py`:

### BC Settings
```python
BC_N_FILES = 1000          # Expert data files
BC_N_EPOCHS = 50           # Training epochs
BC_BATCH_SIZE = 512
BC_LR = 1e-3
```

### PPO Settings
```python
PPO_N_EPOCHS = 100         # Training epochs
PPO_NUM_ENVS = 8           # Parallel environments
PPO_STEPS_PER_EPOCH = 10000
PPO_LR = 1e-5              # TODO: Increase to 3e-4
PPO_LOG_STD_INIT = log(0.05)  # TODO: Increase to 0.1
```

### Architecture (Shared)
```python
STATE_DIM = 56             # Input state dimension
ACTION_DIM = 1             # Steering output
HIDDEN_DIM = 128           # Network width
TRUNK_LAYERS = 1           # Shared trunk depth
HEAD_LAYERS = 3            # Actor/Critic head depth
```

## 📊 Expected Results

### BC Stage
- Training time: ~5-10 minutes
- Expected cost: ~84 (matches PID baseline)

### PPO Stage
- Training time: ~30 minutes (100 epochs)
- Expected cost: ~110 mean / ~82 median
- Cost distribution:
  - 10% < 38 (beating target!)
  - 25% < 52 (close to target)
  - 50% < 82 (BC-level)
  - High variance on hard routes

## 🐛 Debugging from Here

The unified pipeline makes debugging easier:

### 1. Check BC Stage
If BC fails or gets poor cost (>100):
- Check expert data collection (inspect states/actions)
- Check state normalization (OBS_SCALE)
- Check network forward pass
- Increase BC_N_FILES or BC_N_EPOCHS

### 2. Check PPO Stage
If PPO explodes (cost >1000):
- Check reward calculation in TinyPhysicsGymEnv.step()
- Check episode reset logic
- Verify state building matches BC

If PPO doesn't improve (cost stays ~110):
- Increase exploration: PPO_LOG_STD_INIT = log(0.1)
- Increase learning rate: PPO_LR = 3e-4
- Train longer: PPO_N_EPOCHS = 500

### 3. Check State/Action Flow
Add debugging prints:
```python
# In build_state():
print(f"State: error={error:.3f}, v_ego={state.v_ego:.1f}, curv={curv_now:.6f}")

# In TinyPhysicsGymEnv.step():
print(f"Action: {action_value:.3f}, Reward: {reward:.3f}, Cost: {self.episode_cost:.1f}")
```

### 4. Check Episode Costs
Monitor PPO training output:
```
Epoch   0 | Cost: 301.54 | Best: 301.54 | Episodes: 16
Epoch   5 | Cost: 115.00 | Best:  56.31 | Episodes: 16
```

If "Episodes: 0":
- Environments not completing episodes
- Check data file loading
- Check episode termination condition

If costs are constant:
- Policy not updating
- Check PPO update logic
- Check gradient flow

## 🎯 Next Improvements

### Quick Wins (in train_pipeline.py)
1. **Increase exploration**: `PPO_LOG_STD_INIT = np.log(0.1)`
2. **Increase learning rate**: `PPO_LR = 3e-4`
3. **Train longer**: `PPO_N_EPOCHS = 500`
4. **Add anti-windup**: Clamp error_integral in build_state()

Expected gain: 110 → 70 cost

### Medium Changes (requires refactor)
1. **Compress state**: 56D → 10D (bin future curvatures)
2. **Dense rewards**: Immediate feedback instead of sparse
3. **Curriculum**: Train on easy routes first

Expected gain: 70 → 45 cost

## 📁 Output Files

```
bc_checkpoint.pth         # BC weights + evaluation
ppo_best.pth              # PPO weights (best during training)
```

To use the trained controller:
```bash
# Evaluate on validation set
python eval_ppo_simple.py

# Official evaluation (5000 files)
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                      --data_path ./data \
                      --num_segs 5000 \
                      --controller ppo_parallel
```

## 🔍 Key Functions

| Function | Purpose |
|----------|---------|
| `build_state()` | Constructs 56D state vector |
| `ActorCritic` | Shared network for BC and PPO |
| `collect_expert_data_single_file()` | PID data collection |
| `train_bc()` | BC training loop |
| `TinyPhysicsGymEnv` | Gymnasium wrapper |
| `PPO` | PPO algorithm implementation |
| `train_ppo()` | PPO training loop |
| `evaluate_controller()` | Validation evaluation |

## 🎓 Understanding the Flow

```
1. Load dataset → split train/val
          ↓
2. Create ActorCritic network (random init)
          ↓
3. BC Stage:
   - Collect PID expert data (1000 files)
   - Train network to clone PID actions
   - Evaluate → save bc_checkpoint.pth
          ↓
4. PPO Stage:
   - Initialize network with BC weights
   - Create 8 parallel Gym environments
   - For 100 epochs:
     * Collect 10k steps of experience
     * Update policy with PPO
     * Track episode costs
   - Evaluate → save ppo_best.pth
          ↓
5. Final summary: BC cost vs PPO cost
```

## ⚠️ Common Issues

### "No episodes completed"
- Episodes taking too long
- Environment stuck
- Check simulator step logic

### "Cost exploding (>1000)"
- Policy diverging
- Learning rate too high
- Reward scale wrong
- Stop training and reduce PPO_LR

### "Cost not improving"
- Stuck in local optimum
- Exploration too low
- Increase PPO_LOG_STD_INIT
- Increase PPO_LR
- Train longer

### "BC cost high (>100)"
- Expert data quality issue
- State normalization wrong
- Check OBS_SCALE values
- Increase BC_N_FILES

## 🚀 Quick Start Checklist

- [ ] Data in `./data/*.csv`
- [ ] Model at `./models/tinyphysics.onnx`
- [ ] Virtual env activated
- [ ] Run `python train_pipeline.py`
- [ ] Monitor BC stage (~10 min)
- [ ] Monitor PPO stage (~30 min)
- [ ] Check `bc_checkpoint.pth` and `ppo_best.pth` created
- [ ] Evaluate with `eval_ppo_simple.py`

## 📖 See Also

- `STATUS.md` - Current results and roadmap
- `WHAT_WORKS.md` - Quick reference
- `beautiful_lander.py` - Reference PPO implementation

