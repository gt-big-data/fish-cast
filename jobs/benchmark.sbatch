#!/bin/bash
# FishCast benchmark on PACE: train the GP models with several seeds, then compare
# them with the baselines on the held-out test years.
#
# Platform week 1 starter. Submit from the repository root:
#     sbatch jobs/benchmark.sbatch
# Results land in artifacts/benchmark_<date>_<jobid>/benchmark_table.csv
#
# Adjust the partition, time and GPU lines for your allocation. GP training is light;
# a CPU node is fine (about 1-2 minutes per epoch on CPU for the full dataset).
#SBATCH --account=paceship-fishcast
#SBATCH --job-name=fishcast-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/benchmark_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.10
source .venv/bin/activate

MODELS="${MODELS:-climate_conditioned_gp autoregressive_gp latent_state_augmented_gp}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-30}"
DATA="${DATA:-data/krillcast_merged.csv}"
OUT="artifacts/benchmark_$(date +%Y%m%d)_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee "$OUT/RUN_INFO.txt"
echo "models: $MODELS | seeds: $SEEDS | epochs: $EPOCHS | data: $DATA" | tee -a "$OUT/RUN_INFO.txt"

RUNS=()
for model in $MODELS; do
  for seed in $SEEDS; do
    run="$OUT/runs/${model}_s${seed}"
    echo "== $model seed $seed"
    # TODO (Analysis week 1 task): once train_fishcast.py has --seed, pass it here.
    python scripts/train_fishcast.py --model-type "$model" --data "$DATA" \
      --output-dir "$run" --epochs "$EPOCHS" --device cpu
    RUNS+=("$run")
  done
done

python scripts/benchmark_skill.py --data "$DATA" --gp-runs "${RUNS[@]}" --output "$OUT"
python viz/figures/leaderboard.py "$OUT/benchmark_table.csv" --out "$OUT/leaderboard" --hide-seeds || true
echo "done: $OUT/benchmark_table.csv"
