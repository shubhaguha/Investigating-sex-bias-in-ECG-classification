#!/bin/bash
# Submit the whole run, each step starting only after the previous one succeeded:
#   bash slurm/submit_best.sh              data preparation, then training folds 0-4
#   bash slurm/submit_best.sh --download   download first (see slurm/download_data.sh)
# Run from the repository root.
set -euo pipefail
dependency=()
if [[ "${1:-}" == "--download" ]]; then
    download=$(sbatch --parsable slurm/download_data.sh)
    echo "Submitted download ($download)"
    dependency=(--dependency=afterok:"$download")
fi
prep=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} slurm/prepare_data.sh)
train=$(sbatch --parsable --dependency=afterok:"$prep" slurm/train_best.sh)
echo "Submitted data preparation ($prep) and training folds 0-4 ($train)"
echo "Logs: slurm/logs/"
