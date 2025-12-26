#!/bin/bash
# Experiment 003: Clean BC from PID
# Run from project root: bash experiments/exp003_ppo_bc_init/run.sh

set -e
cd "$(dirname "$0")"  # cd to experiment folder

echo "=========================================="
echo "Experiment 003: Clean BC from PID"
echo "=========================================="

# Train BC
echo "[1/2] Training BC..."
python train_bc.py 2>&1 | tee results/training.log

# Evaluate
echo ""
echo "[2/2] Evaluating BC..."
cd ../..
source .venv/bin/activate
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller bc

echo ""
echo "=========================================="
echo "✅ Experiment complete!"
echo "Checkpoint: experiments/exp003_ppo_bc_init/results/checkpoints/bc_best.pth"
echo "Controller: controllers/bc.py"
echo "=========================================="

