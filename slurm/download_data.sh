#!/bin/bash
# Download the six PhysioNet/CinC 2021 databases the paper uses, in parallel.
#
#   sbatch slurm/download_data.sh       (from the repository root; see slurm/submit_best.sh)
#   bash slurm/download_data.sh         (also works without SLURM, e.g. in tmux on a login node)
#   bash slurm/download_data.sh --missing
#                                       (after a partial download: fetch only the missing files)
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
# Resuming re-checks every file already downloaded (one request each), which takes hours.
# --missing instead reads each g* folder's listing (one request per folder), compares it with the
# files on disk, and downloads only the .hea/.mat files that are not there yet, plus any .mat that an
# interrupted download left damaged (wrong size for its header; see scripts/check_download.py).
#
# Compute nodes on many clusters have no internet access: use a partition meant for data transfer
# (staging on Snellius, often "transfer" or "io" elsewhere), or run it with bash on a login node.

#SBATCH --job-name=ecg-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# staging: Snellius's partition for data transfers
#SBATCH --partition=staging
#SBATCH --account=gusr58922

set -euo pipefail

MISSING_ONLY=false
if [[ "${1:-}" == "--missing" ]]; then
    MISSING_ONLY=true
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--missing]" >&2
    exit 2
fi

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
CHECK="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/check_download.py"
[[ -f "$CHECK" ]] || { echo "Run from the repository root ($CHECK not found)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found (needed by $CHECK)" >&2; exit 1; }
root="$DOWNLOAD_DIR/${BASE_URL#*://}/training"   # where wget -r puts the databases
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

if $MISSING_ONLY; then
    # File URLs listed on the server whose local copy (same tree as wget -r) does not exist
    missing_list="$DOWNLOAD_DIR/.challenge-2021-missing.txt"
    : > "$missing_list"
    for folder in "${folders[@]}"; do
        if [[ ! "$folder" =~ /g[0-9]+/$ ]]; then
            echo "  ${folder}: no subfolder listing, run without --missing for this database" >&2
            continue
        fi
        listing=$(wget -q --tries=5 --timeout=60 -O - "$folder") || { echo "  could not list $folder" >&2; continue; }
        local_dir="$DOWNLOAD_DIR/${folder#*://}"
        { grep -oE 'href="[^"]*\.(hea|mat)"' <<< "$listing" || true; } | sed -E 's/^href="(.*\/)?//; s/"$//' | sort -u |
            while read -r name; do
                [[ -f "$local_dir$name" ]] || echo "$folder$name"
            done >> "$missing_list"
    done
    # .mat files that exist but do not match their header: delete them and fetch them again
    damaged=0
    while read -r mat_file; do
        rm -f "$mat_file"
        echo "$BASE_URL/training/${mat_file#"$root"/}"
        damaged=$((damaged + 1))
    done < <(python3 -I "$CHECK" --list-bad "$root" "${DATABASES[@]}") >> "$missing_list"
    echo "$(wc -l < "$missing_list") files to download, $damaged of them replacing damaged .mat files (list: $missing_list)"
    # -x recreates the host/path folders, so files land where wget -r puts them
    xargs -P "$PARALLEL" -n 50 wget -x -c --no-verbose --tries=20 --waitretry=10 --timeout=60 \
        < "$missing_list" || echo "wget reported errors; the check below counts what is still missing"
else
    # wget exits non-zero for a few failed files; the check below decides whether the download is complete
    # shellcheck disable=SC2016
    printf '%s\n' "${folders[@]}" | xargs -P "$PARALLEL" -n 1 sh -c 'wget $WGET_OPTS "$0" || echo "wget reported errors for $0"'
fi

# Check that every recording has its .hea and a .mat of the size the header describes
counts=$(python3 -I "$CHECK" "$root" "${DATABASES[@]}")
missing=0
echo "Recordings downloaded (.hea with intact .mat / expected):"
while read -r db complete; do
    echo "  $db: $complete / ${EXPECTED[$db]}"
    (( complete >= EXPECTED[$db] )) || missing=$((missing + EXPECTED[$db] - complete))
done <<< "$counts"
if (( missing > 0 )); then
    echo "$missing recordings missing or damaged; run again (with --missing to fetch only those) ($(date))" >&2
    exit 1
fi
echo "Download complete ($(date))"
