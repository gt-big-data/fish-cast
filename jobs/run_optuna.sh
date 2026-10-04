#!/bin/bash
set -euo pipefail
# Script to run the automated FishCast Optuna sweep.

echo "Starting Optuna Hyperparameter Sweep (100 Trials) for FishCast..."

python scripts/tune_fishcast.py \
  --data data/krillcast_merged.csv \
  --output-dir artifacts/FishCast_Optuna_Sweep \
  --n-trials 100 \
  --epochs 50

echo "--- Sweep finished! ---"
