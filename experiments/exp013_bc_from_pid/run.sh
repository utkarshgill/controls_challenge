#!/bin/bash
set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "============================================"
echo "Experiment 013: BC from PID (v3 - Fixed)"
echo "============================================"

# Step 1: Collect clean PID data
echo ""
echo "Step 1: Collecting PID trajectories..."
echo "  - Only from step >= 100 (CONTROL_START_IDX)"
echo "  - Avoids logged steerCommand contamination"
python collect_data.py 2>&1 | tee collect_v3.log

# Step 2: Train BC network
echo ""
echo "Step 2: Training BC network..."
echo "  - Network: 3 layers × 128 hidden units"
echo "  - Output: Tanh → Scale to [-2, 2]"
python train_bc.py 2>&1 | tee train_v3.log

# Step 3: Evaluate
echo ""
echo "Step 3: Evaluating BC vs PID..."
python evaluate.py 2>&1 | tee eval_v3.log

echo ""
echo "============================================"
echo "Experiment 013 complete!"
echo "============================================"
