#!/bin/bash
# Complete PPO workflow with critic pre-training
# Slow is smooth. Smooth is fast.

set -e

echo "========================================================================"
echo "PPO Training with Pre-trained Critic"
echo "========================================================================"
echo ""

# Step 1: Pre-train critic
echo "Step 1: Pre-training critic on BC rollouts..."
echo "------------------------------------------------------------------------"
python pretrain_critic.py \
    --num_rollouts 100 \
    --epochs 50 \
    --lr 1e-3

echo ""
echo "✓ Critic pre-trained"
echo ""

# Step 2: Train PPO with pre-trained critic
echo "Step 2: Training PPO with pre-trained critic..."
echo "------------------------------------------------------------------------"
python train_ppo.py \
    --critic_checkpoint ./outputs/pretrained_critic.pt \
    --epochs 50 \
    --episodes_per_epoch 10 \
    --eval_interval 10 \
    --K_epochs 10

echo ""
echo "✓ PPO training complete"
echo ""

# Step 3: Evaluate
echo "Step 3: Evaluating trained PPO model..."
echo "------------------------------------------------------------------------"
python eval_ppo.py

echo ""
echo "========================================================================"
echo "Complete workflow finished!"
echo "========================================================================"
