#!/bin/bash
# 5-fold cross-validation for the two single-sex training sets, F0 (sex_ratio 100_0, all male) and
# F100 (0_100, all female), with seed 42, to match train_best.sh's F50 folds. Same settings as
# train_best.sh otherwise. One array task per ratio and test fold:
#   tasks 0-4: 100_0, folds 0-4      tasks 5-9: 0_100, folds 0-4
#
#   sbatch slurm/train_folds.sh                (from the repository root; needs prepare_data.sh first)
#   sbatch --array=7 slurm/train_folds.sh      (re-run one task: ratio = RATIOS[i / 5], fold = i % 5)
#
# Runs are named like the grid's (resnet_attention_<ratio>_fold<f>_seed42), so fold 0, which
# train_grid.sh already ran with seed 42, is skipped. Outputs as in train_best.sh; summarize them
# with scripts/summarize_runs.py or slurm/report.sh.

#SBATCH --job-name=ecg-folds
#SBATCH --array=0-9
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
RATIOS=(100_0 0_100)                        # male_female: F0, F100
SEED=42
TASK="${SLURM_ARRAY_TASK_ID:?run with sbatch (or set SLURM_ARRAY_TASK_ID to 0-9)}"
RATIO="${RATIOS[TASK / 5]}"
FOLD=$((TASK % 5))
EXPERIMENT="${MODEL}_${RATIO}_fold${FOLD}_seed${SEED}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f train_pipeline.py ]] || { echo "Submit from the repository root" >&2; exit 1; }
[[ -f "$DATA_ROOT/dataset_division.json" ]] || { echo "$DATA_ROOT/dataset_division.json not found; run slurm/prepare_data.sh first" >&2; exit 1; }
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# A run that already finished (test results written) is not trained again: a resubmitted array, or
# fold 0, which train_grid.sh already ran with seed 42, exits here.
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

# -u: unbuffered, so the log shows each epoch as it ends and survives a time-limit kill
python -u train_pipeline.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --sex_ratio "$RATIO" \
    --fold "$FOLD" \
    --seed "$SEED" \
    --experiment_id "$EXPERIMENT"
# no --gpu: "--gpu 0" parses as the list [0], which makes train_pipeline.py wrap the model in
# DataParallel, and utils/train.py then no longer recognises ResnetAttention; the default is GPU 0

echo "Finished $(date)"
