#!/bin/bash
# Submit the 3 ratios x 3 seeds run (slurm/train_grid.sh), each step starting only after the previous
# one succeeded, followed by the runtime and memory report (slurm/report.sh):
#   bash slurm/submit_grid.sh              data preparation, training, report
#   bash slurm/submit_grid.sh --download   download first (see slurm/download_data.sh)
#   bash slurm/submit_grid.sh --no-prep    the data is already prepared: training and report only
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
train=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} slurm/train_grid.sh)
report=$(sbatch --parsable --dependency=afterany:"$train" slurm/report.sh "$train")
echo "Submitted training, 3 ratios x 3 seeds ($train), and the resource report ($report)"
echo "Logs: slurm/logs/"
