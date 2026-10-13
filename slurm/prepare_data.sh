#!/bin/bash
# One-off CPU job: preprocess the PhysioNet 2021 download and create dataset_division.json.
#
#   sbatch slurm/prepare_data.sh        (from the repository root; see slurm/submit_best.sh)
#
# Writes $DATA_ROOT/PhysioNet2021_preprocessed/ and $DATA_ROOT/dataset_division.json.
# Re-running skips recordings that are already preprocessed and keeps an existing split file.

#SBATCH --job-name=ecg-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

CINC_DIR="${CINC_DIR:-$HOME/physionet.org}"
DATA_ROOT="${DATA_ROOT:-$HOME/ecg_data}"
VENV="${VENV:-$HOME/venvs/ecg-sexbias}"    # set VENV= to skip activation (e.g. when using conda)
# module load Python/3.11                   # cluster-specific environment setup goes here

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f train_pipeline.py ]] || { echo "Submit from the repository root" >&2; exit 1; }
[[ -d "$CINC_DIR" ]] || { echo "$CINC_DIR not found" >&2; exit 1; }
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "Preprocessing $CINC_DIR -> $DATA_ROOT/PhysioNet2021_preprocessed ($(date))"
python scripts/preprocess_data.py --cinc_dir "$CINC_DIR" --data_root "$DATA_ROOT" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"

if [[ -f "$DATA_ROOT/dataset_division.json" ]]; then
    echo "$DATA_ROOT/dataset_division.json exists; keeping it (delete it to regenerate)"
else
    python scripts/make_dataset_division.py --data_root "$DATA_ROOT"
fi
echo "Done ($(date))"
