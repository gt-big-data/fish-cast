#!/bin/bash
#SBATCH --job-name=fishcast_v4
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1             # Request 1 V100
#SBATCH --mem=64G                     # Request system memory
#SBATCH --time=16:00:00               # Max walltime
#SBATCH --output=logs/slurm-%j.out    # Standard output log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_GT_USERNAME@gatech.edu

# --- Load Environment ---
cd $SLURM_SUBMIT_DIR

# --- Load System Modules ---
module load cuda/12.6.1
module load python/3.10

# 1. Navigate to your project directory and activate the environment
source .venv/bin/activate

# 2. Execute your orchestrator shell script
bash run_v4_experiment.sh