#!/bin/bash
#SBATCH --account=paceship-fishcast
#SBATCH --job-name=grid_search
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_GT_USERNAME@gatech.edu

set -euo pipefail

# --- Load Environment ---
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# --- Load System Modules ---
module load cuda/12.6.1
module load python/3.10

# 1. Navigate to your project directory
source .venv/bin/activate

# 2. Execute your orchestrator shell script
bash run_optuna.sh