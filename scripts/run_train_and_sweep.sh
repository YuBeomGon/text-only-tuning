#!/bin/bash
# Train B for 1000 steps, then sweep eval across all checkpoints × alphas.
#
# Usage:
#   bash scripts/run_train_and_sweep.sh [train_config]
#
# Example:
#   bash scripts/run_train_and_sweep.sh configs/train_v2.yaml
#   bash scripts/run_train_and_sweep.sh  # defaults to configs/train_v2.yaml

set -e

CONFIG="${1:-configs/train_v2.yaml}"
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
SWEEP_OUTPUT="experiments/outputs/sweep_$(basename $CKPT_DIR)"

echo "========================================"
echo " Config:     $CONFIG"
echo " Ckpt dir:   $CKPT_DIR"
echo " Sweep out:  $SWEEP_OUTPUT"
echo "========================================"

# Step 1: Train
echo ""
echo "[1/2] Training..."
python -m src.train.train_text_only --config "$CONFIG"

# Step 2: Sweep eval
echo ""
echo "[2/2] Sweep evaluation..."
python -m scripts.sweep_eval \
    --ckpt_dir "$CKPT_DIR" \
    --alphas 0.9 0.7 0.5 \
    --output_dir "$SWEEP_OUTPUT"

echo ""
echo "Done. Summary at: $SWEEP_OUTPUT/sweep_summary.json"
