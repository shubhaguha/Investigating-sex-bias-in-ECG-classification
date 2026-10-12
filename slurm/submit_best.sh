#!/bin/bash
# Submit data preparation, then the 5 training folds once it has succeeded.
#   bash slurm/submit_best.sh        (from the repository root)
set -euo pipefail
prep=$(sbatch --parsable slurm/prepare_data.sh)
train=$(sbatch --parsable --dependency=afterok:"$prep" slurm/train_best.sh)
echo "Submitted data preparation ($prep) and training folds 0-4 ($train)"
echo "Logs: slurm/logs/"
