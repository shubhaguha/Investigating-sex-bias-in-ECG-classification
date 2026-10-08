#!/bin/bash
# Full experiment grid as a SLURM job array: 3 models x 5 sex ratios x 5 folds = 75 tasks.
#
#   Task IDs   Model
#   0-24       cnn
#   25-49      resnet_attention
#   50-74      xresnet101
#   Within a model, ID % 5 is the fold and (ID / 5) % 5 picks the ratio
#   (100_0, 75_25, 50_50, 25_75, 0_100, male_female).
#
# Submit from the repository root:
#   sbatch slurm/train_array.sh                    # everything
#   sbatch --array=0-24 slurm/train_array.sh       # only the CNN
#   sbatch --array=0-74%10 slurm/train_array.sh    # at most 10 tasks at a time
#
# Tasks whose results/<experiment>/test_results.pickle already exists exit immediately,
# so the same command can be resubmitted to fill in failed or timed-out tasks.
#
# Adjust the #SBATCH lines and the settings below for your cluster.

#SBATCH --job-name=ecg-sexbias
#SBATCH --array=0-74
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
##SBATCH --partition=gpu
##SBATCH --account=my_account

set -euo pipefail

# ---- settings -------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$REPO/data/PhysioNet2021_preprocessed}"
VENV="${VENV:-$REPO/.venv}"            # set VENV= to skip activation (e.g. when using conda)
EXTRA_ARGS="${EXTRA_ARGS:-}"           # extra train_pipeline.py arguments, e.g. "--epochs 50"
# module load CUDA/12.1                # cluster-specific environment setup goes here
# ---------------------------------------------------------------------------

MODELS=(cnn resnet_attention xresnet101)
RATIOS=(100_0 75_25 50_50 25_75 0_100)

TASK_ID="${SLURM_ARRAY_TASK_ID:?run this with sbatch (or set SLURM_ARRAY_TASK_ID)}"
if (( TASK_ID < 0 || TASK_ID >= ${#MODELS[@]} * ${#RATIOS[@]} * 5 )); then
    echo "Task ID $TASK_ID is outside 0-74" >&2
    exit 1
fi
MODEL="${MODELS[TASK_ID / 25]}"
RATIO="${RATIOS[(TASK_ID / 5) % 5]}"
FOLD=$(( TASK_ID % 5 ))
EXPERIMENT="${MODEL}_${RATIO}_fold${FOLD}"

cd "$REPO"
if [[ ! -f train_pipeline.py ]]; then
    echo "Submit this job from the repository root ($REPO has no train_pipeline.py)" >&2
    exit 1
fi
if [[ ! -f "$DATA_DIR/dataset_division.json" ]]; then
    echo "$DATA_DIR/dataset_division.json not found; run scripts/make_dataset_division.py first" >&2
    exit 1
fi
if [[ -f "results/$EXPERIMENT/test_results.pickle" ]]; then
    echo "$EXPERIMENT already finished; skipping"
    exit 0
fi
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "Task $TASK_ID: model=$MODEL ratio=$RATIO fold=$FOLD on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU visible"

# shellcheck disable=SC2086
python train_pipeline.py \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" \
    --sex_ratio "$RATIO" \
    --fold "$FOLD" \
    --experiment_id "$EXPERIMENT" \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --gpu 0 \
    $EXTRA_ARGS

echo "Finished $(date)"
