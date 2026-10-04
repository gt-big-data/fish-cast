#!/bin/bash
#
# Script to run the FishCast v4 sweep sequentially.

# --- Experiment Parameters ---
OUTPUT_DIR_BASE="artifacts/FishCast_Test_v4"
HIDDEN_DIMS=(64 128 256)
NUM_INDUCING_POINTS=500
CLASSIFICATION_WEIGHT=5.0
EPOCHS=50
BATCH_SIZE=128
SCHEDULER_MONITOR="val_auc"

echo "Starting FishCast sweep over hidden dimensions: ${HIDDEN_DIMS[*]}"

for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
  echo ""
  echo "----------------------------------------------------"
  echo "--- Running with hidden_dim: $HIDDEN_DIM"
  echo "----------------------------------------------------"
  
  RUN_DIR="$OUTPUT_DIR_BASE/hidden_dim_${HIDDEN_DIM}"
  
  python scripts/train_fishcast.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-inducing-points $NUM_INDUCING_POINTS \
    --classification-weight $CLASSIFICATION_WEIGHT \
    --hidden-dim $HIDDEN_DIM \
    --scheduler-monitor $SCHEDULER_MONITOR \
    --scheduler-patience 5 \
    --scheduler-factor 0.5 \
    --threshold-steps 41 \
    --output-dir "$RUN_DIR" \
    --device "cuda"
done

echo ""
echo "--- Sweep finished! Results are in $OUTPUT_DIR_BASE ---"