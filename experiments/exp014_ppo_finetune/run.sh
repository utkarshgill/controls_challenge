#!/bin/bash
# Experiment 014: PPO Fine-Tuning from BC

set -e

echo "============================================"
echo "Experiment 014: PPO Fine-Tuning"
echo "============================================"
echo ""

# Check if BC checkpoint exists
BC_CHECKPOINT="../exp013_bc_from_pid/bc_best.pth"
if [ ! -f "$BC_CHECKPOINT" ]; then
    echo "❌ BC checkpoint not found: $BC_CHECKPOINT"
    echo "   Please run exp013 first to train BC baseline."
    exit 1
fi

echo "✓ BC checkpoint found"
echo ""

# Train PPO
echo "Starting PPO training..."
echo "This will take ~2-3 hours on MPS (M-series Mac)"
echo "Press Ctrl+C to stop"
echo ""

python train_ppo.py

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo ""

# Evaluate
if [ -f "ppo_best.pth" ]; then
    echo "Running evaluation..."
    python evaluate.py
else
    echo "❌ No checkpoint found. Training may have failed."
fi



