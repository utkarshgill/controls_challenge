#!/bin/bash

# Experiment 042: Behavioral Cloning Baseline
# Train and evaluate BC model

set -e  # Exit on error

echo "========================================================================"
echo "Experiment 042: Behavioral Cloning"
echo "========================================================================"

# Configuration
MODEL_PATH="../../models/tinyphysics.onnx"
DATA_DIR="../../data"
OUTPUT_DIR="./outputs"
NUM_TRAIN=800
NUM_VAL=100
NUM_TEST=100
EPOCHS=50
BATCH_SIZE=256
LR=0.001

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Train/Val/Test: $NUM_TRAIN/$NUM_VAL/$NUM_TEST files"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""

# Step 1: Train BC model
echo "========================================================================"
echo "Step 1: Training BC model"
echo "========================================================================"
python train_bc.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train "$NUM_TRAIN" \
    --num_val "$NUM_VAL" \
    --num_test "$NUM_TEST" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR"

# Step 2: Evaluate BC model
echo ""
echo "========================================================================"
echo "Step 2: Evaluating BC model"
echo "========================================================================"
python eval_bc.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --bc_checkpoint "$OUTPUT_DIR/best_model.pt" \
    --num_files 100

echo ""
echo "========================================================================"
echo "Experiment 042: Complete"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "  - best_model.pt"
echo "  - final_model.pt"
echo "  - training_history.pt"
