#!/bin/bash
# Runtime and memory of finished jobs, as recorded by SLURM, plus a summary of every run in results/.
#
#   bash slurm/report.sh <job id> [<job id> ...]     (from the repository root)
#
# submit_best.sh and submit_grid.sh queue this to run after training, writing slurm/logs/report_<id>.out.

#SBATCH --job-name=ecg-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
VENV="${VENV:-$HOME/venvs/ecg-sexbias}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
for job in "$@"; do
    # Elapsed: wall time; TotalCPU: CPU time used; MaxRSS: peak host memory of the job step
    sacct -j "$job" --units=G \
        --format=JobID%-18,JobName%-14,State%-11,Elapsed,TotalCPU,MaxRSS,ReqMem,AllocTRES%-40,NodeList
    echo
done
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python scripts/summarize_runs.py
