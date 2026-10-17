#!/bin/bash
# Submit the whole run, each step starting only after the previous one succeeded:
#   bash slurm/submit_best.sh              data preparation, training folds 0-4, resource report
#   bash slurm/submit_best.sh --download   download first (see slurm/download_data.sh)
#   bash slurm/submit_best.sh --no-prep    the data is already prepared: training and report only
# Run from the repository root.
set -euo pipefail
dependency=()
if [[ "${1:-}" == "--download" ]]; then
    download=$(sbatch --parsable slurm/download_data.sh)
    echo "Submitted download ($download)"
    dependency=(--dependency=afterok:"$download")
fi
if [[ "${1:-}" != "--no-prep" ]]; then
    prep=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} slurm/prepare_data.sh)
    echo "Submitted data preparation ($prep)"
    dependency=(--dependency=afterok:"$prep")
fi
train=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} slurm/train_best.sh)
report=$(sbatch --parsable --dependency=afterany:"$train" slurm/report.sh "$train")
echo "Submitted training folds 0-4 ($train) and the resource report ($report)"
echo "Logs: slurm/logs/"
