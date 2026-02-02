#!/bin/bash
# Experiment 039: PID + Learned Feedforward

set -e

echo "================================"
echo "Experiment 039: PID + FF"
echo "================================"
echo ""

# Test architecture first
echo "Step 1: Testing architecture..."
python test_arch.py

echo ""
echo "✓ Architecture tests passed!"
echo ""
echo "Step 2: Starting training..."
echo "  - This will take a while (~1-2 hours)"
echo "  - Best model saved to best_model.pth"
echo "  - Monitor eval_cost: should drop below 75 if working"
echo ""

# Start training
python train.py

echo ""
echo "================================"
echo "Training complete!"
echo "================================"
echo ""
echo "Step 3: Evaluating trained model..."
python evaluate.py

echo ""
echo "Done! Check best_model.pth for saved weights."

