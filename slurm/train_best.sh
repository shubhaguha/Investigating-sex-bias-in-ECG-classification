#!/bin/bash
# Best configuration from Galanty et al. (MELBA 2025): ResNet with Attention trained on a
# sex-balanced training set (F50, 50% female / 50% male), evaluated with 5-fold cross-validation.
# One array task per test fold; all other settings are train_pipeline.py's defaults, which match
# the paper (focal loss alpha 0.75 / gamma 2, Adam lr 1e-3, batch size 128, up to 100 epochs).
#
#   sbatch slurm/train_best.sh                 (from the repository root; needs prepare_data.sh first)
#   sbatch --array=3 slurm/train_best.sh       (re-run a single fold)
#
# Test results (ROC AUC, pROC AUC, PR AUC for AF, SR, MI; female test set first, then male) are
# printed at the end of each log in slurm/logs/. Weights go to results/model_weights/. Each run also
# writes results/<experiment>/: test_metrics.json, resources.json (time per phase, peak GPU and host
# memory) and gpu_usage.csv (nvidia-smi every 30 s); see slurm/report.sh.

#SBATCH --job-name=ecg-best
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_fold%a.out
#SBATCH --partition=gpu_a100
#SBATCH --account=gusr58922

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/ecg_data}"
VENV="${VENV-$HOME/venvs/ecg-sexbias}"    # set VENV= to skip activation (e.g. when using conda)
# module load CUDA/12.1                     # cluster-specific environment setup goes here

MODEL=resnet_attention
RATIO=50_50
FOLD="${SLURM_ARRAY_TASK_ID:?run with sbatch (or set SLURM_ARRAY_TASK_ID to 0-4)}"
EXPERIMENT="${MODEL}_${RATIO}_fold${FOLD}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f train_pipeline.py ]] || { echo "Submit from the repository root" >&2; exit 1; }
[[ -f "$DATA_ROOT/dataset_division.json" ]] || { echo "$DATA_ROOT/dataset_division.json not found; run slurm/prepare_data.sh first" >&2; exit 1; }
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
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
    --experiment_id "$EXPERIMENT" \
    --gpu 0

echo "Finished $(date)"
