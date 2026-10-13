#!/bin/bash
# Download the six PhysioNet/CinC 2021 databases the paper uses, in parallel.
#
#   sbatch slurm/download_data.sh       (from the repository root; see slurm/submit_best.sh)
#   bash slurm/download_data.sh         (also works without SLURM, e.g. in tmux on a login node)
#
# A plain `wget -r` fetches ~176,000 small files one after the other, so it is slow because of the
# per-request round trips, not the bandwidth. This job runs one wget per g* subfolder, PARALLEL at a
# time, and skips the two databases the paper does not use (PTB, St Petersburg INCART).
#
# Files land in the same tree as `wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/`
# run from $DOWNLOAD_DIR, i.e. $DOWNLOAD_DIR/physionet.org/files/challenge-2021/1.0.3/training/...,
# so files from an earlier or interrupted download are reused. Stop any wget still writing there
# first. Resubmitting resumes; the job fails until every recording has its .hea and .mat file.
#
# Compute nodes on many clusters have no internet access: use a partition that does (often called
# "transfer" or "io"), or run it with bash on a login node.

#SBATCH --job-name=ecg-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
##SBATCH --partition=transfer

set -euo pipefail

DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME}"
PARALLEL="${PARALLEL:-8}"              # concurrent wget processes; keep it modest towards physionet.org
BASE_URL="${BASE_URL:-https://physionet.org/files/challenge-2021/1.0.3}"

# Database folder -> number of recordings in v1.0.3 (87,663 in total, as in the paper's Figure 2)
declare -A EXPECTED=(
    [ptb-xl]=21837 [cpsc_2018]=6877 [cpsc_2018_extra]=3453
    [georgia]=10344 [chapman_shaoxing]=10247 [ningbo]=34905
)
DATABASES=(ptb-xl cpsc_2018 cpsc_2018_extra georgia chapman_shaoxing ningbo)

export WGET_OPTS="-r -N -c -np --no-verbose --tries=20 --waitretry=10 --timeout=60"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"
echo "Downloading into $DOWNLOAD_DIR with $PARALLEL parallel downloads ($(date))"

# One work item per g* subfolder (one per database if its listing cannot be read)
folders=()
for db in "${DATABASES[@]}"; do
    subdirs=$(wget -q -O - "$BASE_URL/training/$db/" | grep -oE 'href="g[0-9]+/"' | grep -oE 'g[0-9]+' | sort -u -V || true)
    if [[ -z "$subdirs" ]]; then
        echo "  $db: could not list subfolders, downloading it as one unit"
        folders+=("$BASE_URL/training/$db/")
    else
        echo "  $db: $(wc -w <<< "$subdirs") subfolders"
        for g in $subdirs; do
            folders+=("$BASE_URL/training/$db/$g/")
        done
    fi
done

# wget exits non-zero for a few failed files; the check below decides whether the download is complete
# shellcheck disable=SC2016
printf '%s\n' "${folders[@]}" | xargs -P "$PARALLEL" -n 1 sh -c 'wget $WGET_OPTS "$0" || echo "wget reported errors for $0"'

# Check that every recording has both files
root="$DOWNLOAD_DIR/${BASE_URL#*://}/training"
missing=0
echo "Recordings downloaded (.hea with matching .mat / expected):"
for db in "${DATABASES[@]}"; do
    complete=0
    if [[ -d "$root/$db" ]]; then
        complete=$(find "$root/$db" -name '*.hea' | while read -r h; do [[ -f "${h%.hea}.mat" ]] && echo; done | wc -l)
    fi
    echo "  $db: $complete / ${EXPECTED[$db]}"
    (( complete >= EXPECTED[$db] )) || missing=$((missing + EXPECTED[$db] - complete))
done
if (( missing > 0 )); then
    echo "$missing recordings still missing; resubmit to resume ($(date))" >&2
    exit 1
fi
echo "Download complete ($(date))"
