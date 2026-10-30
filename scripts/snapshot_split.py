"""One-off snapshot of everything needed to reproduce the train/val/test split of finished runs.

dataset_division.json stores positions in the recording list that train_pipeline.py builds from
os.listdir order, so on its own it only means something next to that exact data folder. This script
saves, in one folder:

  dataset_division.json   an unchanged copy of the split file
  records.tsv             the recording list as train_pipeline.py sees it now: index, WFDB folder/file
  split_by_name.json      the same split with every index replaced by its recording name, so it can be
                          rebuilt even if the folder is listed in another order or moved
  data_manifest.tsv       size and SHA-256 of every .hea/.mat file the list uses (skip with --no-hash)
  environment.txt         date, host, Python, git commit and status of the repository, pip freeze
  README.txt              what was checked and what each file is

It also checks that the recording list stored in the split file ('meta' -> 'records', written by
scripts/make_dataset_division.py) is identical to the folder's current listing. If the data folder
has not been touched since training, that holds and the snapshot describes exactly what the runs used.

Only the standard library is needed. Run it on the cluster from the repository root:

    python snapshot_split.py --data_root ~/ecg_data
    python snapshot_split.py --data_root ~/ecg_data --out ~/ecg_split_snapshot --no-hash
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# The folders and their order as in train_pipeline.py
FOLDERS = ['WFDB_PTBXL', 'WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_ChapmanShaoxing', 'WFDB_Ningbo']


def list_records(data_dir):
    """Same files and order as data_loader.find_challenge_files, relative to data_dir (e.g. 'WFDB_Ga/E1.hea')."""
    records = []
    for folder in FOLDERS:
        directory = os.path.join(data_dir, folder)
        for f in os.listdir(directory):
            root, extension = os.path.splitext(f)
            if not root.startswith('.') and extension == '.hea':
                if os.path.isfile(os.path.join(directory, root + '.hea')) and \
                        os.path.isfile(os.path.join(directory, root + '.mat')):
                    records.append(f'{folder}/{root}.hea')
    return records


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f'({" ".join(cmd)} failed: {e})'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', required=True,
                        help='Directory containing PhysioNet2021_preprocessed/ and dataset_division.json')
    parser.add_argument('--out', default=None,
                        help='output folder (default: <data_root>/split_snapshot_<date>)')
    parser.add_argument('--repo', default='.', help='repository the runs were trained from (default: current folder)')
    parser.add_argument('--no-hash', action='store_true', help='skip the SHA-256 manifest of the data files')
    parser.add_argument('--workers', type=int, default=8, help='threads for hashing')
    args = parser.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    data_dir = os.path.join(data_root, 'PhysioNet2021_preprocessed')
    division_path = os.path.join(data_root, 'dataset_division.json')
    out = os.path.abspath(os.path.expanduser(
        args.out or os.path.join(data_root, f'split_snapshot_{datetime.date.today().isoformat()}')))
    os.makedirs(out, exist_ok=False)

    with open(division_path) as f:
        division = json.load(f)
    shutil.copy2(division_path, os.path.join(out, 'dataset_division.json'))

    records = list_records(data_dir)
    with open(os.path.join(out, 'records.tsv'), 'w') as f:
        f.write('index\trecord\n')
        for i, r in enumerate(records):
            f.write(f'{i}\t{r}\n')

    stored = division.get('meta', {}).get('records')
    if stored is None:
        check = 'NOT CHECKED: the split file has no stored recording list; names below come from the current listing.'
        names = records
    elif stored == records:
        check = 'OK: the split file\'s recording list is identical to the current listing (same files, same order).'
        names = stored
    else:
        only_stored = len(set(stored) - set(records))
        only_now = len(set(records) - set(stored))
        check = (f'MISMATCH: the split file lists {len(stored)} recordings, the folder now {len(records)} '
                 f'({only_stored} only in the split file, {only_now} only in the folder'
                 f'{", same files in another order" if not only_stored and not only_now else ""}). '
                 'Names below come from the split file\'s own list, i.e. the order the split was made for; '
                 'check whether training ran before or after the folder changed.')
        names = stored

    by_name = {}
    used = set()
    for fold, entry in division.items():
        if fold == 'meta':
            continue
        by_name[fold] = {}
        for key, idx in entry.items():
            by_name[fold][key] = [names[i] for i in idx]
            used.update(idx)
    by_name['meta'] = {k: v for k, v in division.get('meta', {}).items() if k != 'records'}
    with open(os.path.join(out, 'split_by_name.json'), 'w') as f:
        json.dump(by_name, f, indent=1)

    if not args.no_hash:
        files = []
        for r in sorted({names[i] for i in used}):
            files += [r, r[:-len('.hea')] + '.mat']
        paths = [os.path.join(data_dir, f) for f in files]
        print(f'Hashing {len(paths)} files...')
        with ThreadPoolExecutor(args.workers) as pool:
            digests = list(pool.map(lambda p: sha256(p) if os.path.isfile(p) else 'MISSING', paths))
        with open(os.path.join(out, 'data_manifest.tsv'), 'w') as f:
            f.write('file\tbytes\tsha256\n')
            for name, p, d in zip(files, paths, digests):
                size = os.path.getsize(p) if os.path.isfile(p) else -1
                f.write(f'{name}\t{size}\t{d}\n')

    repo = os.path.abspath(args.repo)
    with open(os.path.join(out, 'environment.txt'), 'w') as f:
        f.write(f'date: {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'host: {socket.gethostname()}\n')
        f.write(f'python: {sys.version.split()[0]} ({platform.platform()})\n')
        f.write(f'data_root: {data_root}\n')
        f.write(f'repository: {repo}\n')
        f.write(f'git branch: {run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)}\n')
        f.write(f'git commit: {run(["git", "rev-parse", "HEAD"], repo)}\n')
        f.write(f'git status:\n{run(["git", "status", "--short"], repo) or "(clean)"}\n')
        f.write(f'\npip freeze:\n{run([sys.executable, "-m", "pip", "freeze"])}\n')

    with open(os.path.join(out, 'README.txt'), 'w') as f:
        f.write(f'Split snapshot of {data_root}, made {datetime.date.today().isoformat()}.\n\n{check}\n\n')
        f.write(f'{len(records)} recordings listed now; {len(used)} of them are used by some split.\n')
        f.write('Files: dataset_division.json (copy), records.tsv (index -> recording now), split_by_name.json '
                '(split with recording names), ' + ('' if args.no_hash else 'data_manifest.tsv (size and SHA-256 '
                'of the used .hea/.mat files), ') + 'environment.txt (software and git state).\n')

    print(check)
    print(f'Wrote {out}')
    if stored is not None and stored != records:
        sys.exit(1)


if __name__ == '__main__':
    main()
