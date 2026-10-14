#!/bin/bash
# Runtime, memory and disk I/O of finished jobs, as recorded by SLURM; disk usage of the data and the
# outputs; and a summary of every run in results/.
#
#   bash slurm/report.sh <job id> [<job id> ...]     (from the repository root)
#
# submit_best.sh and submit_grid.sh queue this to run after training, writing slurm/logs/report_<id>.out.

#SBATCH --job-name=ecg-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
# The gusr58922 budget has no CPU partition (rome is refused), so this CPU job runs on a GPU node.
# With a CPU budget, use that partition instead and drop --gpus.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --account=gusr58922

set -uo pipefail
CINC_DIR="${CINC_DIR:-$HOME/physionet.org}"
DATA_ROOT="${DATA_ROOT:-$HOME/ecg_data}"
VENV="${VENV-$HOME/venvs/ecg-sexbias}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
for job in "$@"; do
    # Elapsed: wall time; TotalCPU: CPU time used; MaxRSS: peak host memory of the job step;
    # MaxDiskRead/MaxDiskWrite: most data read/written by one task of the step
    sacct -j "$job" --units=G \
        --format=JobID%-18,JobName%-14,State%-11,Elapsed,TotalCPU,MaxRSS,ReqMem,MaxDiskRead,MaxDiskWrite,AllocTRES%-40,NodeList
    echo
done

# Size and file count (many clusters cap both in their quotas) of the data and the outputs
echo "Disk usage"
printf '%-60s %10s %10s\n' path size files
for path in "$CINC_DIR" "$DATA_ROOT/PhysioNet2021_preprocessed" "$DATA_ROOT/dataset_division.json" \
            results results/model_weights runs slurm/logs "$VENV"; do
    [[ -e "$path" ]] || continue
    printf '%-60s %10s %10s\n' "$path" "$(du -sh "$path" | cut -f1)" "$(du -s --inodes "$path" 2>/dev/null | cut -f1)"
done
echo
echo "Free space"
df -h "$HOME" "$DATA_ROOT" . 2>/dev/null | awk 'NR == 1 || !seen[$0]++'
command -v quota >/dev/null && { echo; quota -s 2>/dev/null; }
echo
if [[ -n "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python scripts/summarize_runs.py
