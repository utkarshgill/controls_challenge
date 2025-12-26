#!/bin/bash
# Experiment 003: PPO with BC Initialization
# Run from project root: bash experiments/exp003_ppo_bc_init/run.sh

set -e
cd "$(dirname "$0")"  # cd to experiment folder

echo "=========================================="
echo "Experiment 003: PPO with BC Init"
echo "=========================================="

# Train PPO with BC initialization
echo "[1/2] Training PPO (BC initialized)..."
python train.py \
    --init-from ../baseline/results/checkpoints/bc_pid_best.pth \
    2>&1 | tee results/logs/training.log

# Move checkpoint
echo "[2/2] Saving results..."
mv ../../ppo_parallel_best.pth results/checkpoints/

echo "=========================================="
echo "✅ Experiment complete!"
echo "Best checkpoint: results/checkpoints/ppo_parallel_best.pth"
echo "=========================================="

