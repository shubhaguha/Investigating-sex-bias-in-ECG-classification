#!/bin/bash
# ResNet with Attention on 3 training sex ratios x 3 random seeds, on one test fold (9 array tasks).
# Ratios are male_female: F0 (100_0), F50 (50_50) and F100 (0_100). The seed changes the training
# randomness (weight init, batch order, crops); the split in dataset_division.json stays the same.
# All other settings are train_pipeline.py's defaults, as in train_best.sh.
#
#   sbatch slurm/train_grid.sh                 (from the repository root; needs prepare_data.sh first)
#   sbatch --array=4 slurm/train_grid.sh       (re-run one task: ratio = RATIOS[i / 3], seed = SEEDS[i % 3])
#   FOLD=2 sbatch slurm/train_grid.sh          (use another test fold, 0-4)
#
# Each run writes results/<experiment>/: test_metrics.json, resources.json (time per phase, peak
# GPU and host memory) and gpu_usage.csv (nvidia-smi every 30 s). Summarize all runs with
# `bash slurm/report.sh <job id>` or `python scripts/summarize_runs.py`.

#SBATCH --job-name=ecg-grid
#SBATCH --array=0-8
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --partition=gpu_a100
#SBATCH --account=gusr58922

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/ecg_data}"
VENV="${VENV-$HOME/venvs/ecg-sexbias}"    # set VENV= to skip activation (e.g. when using conda)
# module load CUDA/12.1                     # cluster-specific environment setup goes here

MODEL=resnet_attention
RATIOS=(100_0 50_50 0_100)
SEEDS=(42 43 44)
FOLD="${FOLD:-0}"
TASK="${SLURM_ARRAY_TASK_ID:?run with sbatch (or set SLURM_ARRAY_TASK_ID to 0-8)}"
RATIO="${RATIOS[TASK / ${#SEEDS[@]}]}"
SEED="${SEEDS[TASK % ${#SEEDS[@]}]}"
EXPERIMENT="${MODEL}_${RATIO}_fold${FOLD}_seed${SEED}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f train_pipeline.py ]] || { echo "Submit from the repository root" >&2; exit 1; }
[[ -f "$DATA_ROOT/dataset_division.json" ]] || { echo "$DATA_ROOT/dataset_division.json not found; run slurm/prepare_data.sh first" >&2; exit 1; }
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# A run that already finished (test results written) is not trained again: a resubmitted array, or
# a grid task that train_best.sh already ran (50_50, fold 0, seed 42), exits here.
if [[ -f "results/$EXPERIMENT/test_metrics.json" ]]; then
    echo "$EXPERIMENT already finished (results/$EXPERIMENT/test_metrics.json exists); skipping. Delete it to retrain."
    exit 0
fi

echo "$EXPERIMENT on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU visible"

# Sample GPU utilization and memory while training runs
mkdir -p "results/$EXPERIMENT"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,nounits \
        -l 30 > "results/$EXPERIMENT/gpu_usage.csv" &
    monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' EXIT
fi

python train_pipeline.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --sex_ratio "$RATIO" \
    --fold "$FOLD" \
    --seed "$SEED" \
    --experiment_id "$EXPERIMENT"
# no --gpu: "--gpu 0" parses as the list [0], which makes train_pipeline.py wrap the model in
# DataParallel, and utils/train.py then no longer recognises ResnetAttention; the default is GPU 0

echo "Finished $(date)"
