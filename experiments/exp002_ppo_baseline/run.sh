#!/bin/bash
# Experiment 002: PPO Baseline
# Run from project root: bash experiments/exp002_ppo_baseline/run.sh

set -e

echo "=========================================="
echo "Experiment 002: PPO Baseline"
echo "=========================================="

# Train PPO (saves to ppo_parallel_best.pth in root)
echo "[1/3] Training PPO..."
python scripts/train_ppo_parallel.py

# Move checkpoint to experiment folder
echo "[2/3] Moving checkpoint..."
mv ppo_parallel_best.pth experiments/exp002_ppo_baseline/results/checkpoints/

# Evaluate
echo "[3/3] Evaluating..."
cd experiments/exp002_ppo_baseline
python eval_final_ppo.py

echo "=========================================="
echo "✅ Experiment complete!"
echo "Results in: experiments/exp002_ppo_baseline/results/"
echo "=========================================="

