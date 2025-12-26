#!/bin/bash
# Experiment 004: Pure PPO
# Run from project root: bash experiments/exp004_ppo_pure/run.sh

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "Experiment 004: Pure PPO"
echo "=========================================="

echo "[1/2] Training PPO..."
python train.py 2>&1 | tee results/logs/training.log

echo ""
echo "[2/2] Evaluating PPO..."
cd ../..
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller ppo

echo ""
echo "=========================================="
echo "✅ Experiment complete!"
echo "=========================================="

