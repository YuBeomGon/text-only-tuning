#!/bin/bash
# Train LoRA adapter, then sweep eval across all checkpoints.
#
# Usage:
#   bash scripts/run_train_and_sweep_lora.sh [train_config]
#
# Example:
#   bash scripts/run_train_and_sweep_lora.sh configs/train_lora_v1.yaml
#   bash scripts/run_train_and_sweep_lora.sh  # defaults to configs/train_lora_v1.yaml

set -e

CONFIG="${1:-configs/train_lora_v1.yaml}"
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
SWEEP_OUTPUT="experiments/outputs/sweep_lora_$(basename $CKPT_DIR)"

echo "========================================"
echo " Config:     $CONFIG"
echo " Ckpt dir:   $CKPT_DIR"
echo " Sweep out:  $SWEEP_OUTPUT"
echo "========================================"

# Step 1: Train LoRA
echo ""
echo "[1/2] Training LoRA adapter..."
python -m src.train.train_lora --config "$CONFIG"

# Step 2: Sweep eval across LoRA checkpoints
echo ""
echo "[2/2] Sweep evaluation..."
python -m scripts.sweep_eval_lora \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$SWEEP_OUTPUT"

echo ""
echo "Done. Summary at: $SWEEP_OUTPUT/sweep_summary.json"
