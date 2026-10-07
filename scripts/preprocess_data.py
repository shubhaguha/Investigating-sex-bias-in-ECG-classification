"""Preprocess the PhysioNet/CinC 2021 data into the layout train_pipeline.py reads.

For each of the six databases used in the paper, every recording is
  - resampled to 500 Hz,
  - bandpass filtered 1-47 Hz (3rd-order Butterworth, zero-phase),
and written to <out_dir>/WFDB_<database>/<record>.hea/.mat (the g*/ subfolders are flattened).
Segmenting to 4096 samples and z-score normalisation happen at load time in data/data_loader.py.

Signals stay in the original ADC units (int16), so the output takes about as much space as the
input databases. Records that already exist in <out_dir> are skipped, so an interrupted run can
be resumed.

Usage:
    python scripts/preprocess_data.py --cinc_dir ~/data/CinC-2021 --out_dir data/PhysioNet2021_preprocessed
    python train_pipeline.py --data_dir data/PhysioNet2021_preprocessed ...
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.data_loader import load_header, get_frequency  # noqa: E402
from data.preprocessing import find_source_dirs, preprocess_recording  # noqa: E402


def rewrite_header(header, fs, data):
    """Update sampling frequency, sample count, initial values and checksums for the new signal."""
    lines = header.rstrip('\n').split('\n')
    first = lines[0].split(' ')
    num_leads = int(first[1])
    first[2] = str(int(fs))
    first[3] = str(data.shape[1])
    lines[0] = ' '.join(first)
    for i in range(num_leads):
        fields = lines[1 + i].strip().split(' ')
        if len(fields) >= 9:
            fields[5] = str(int(data[i, 0]))
            fields[6] = str(int(np.int16(np.sum(data[i].astype(np.int64)) % 65536)))  # WFDB checksum
        lines[1 + i] = ' '.join(fields)
    return '\n'.join(lines) + '\n'


def process(job):
    header_file, out_dir = job
    name = os.path.basename(header_file)[:-4]
    out_hea = os.path.join(out_dir, name + '.hea')
    out_mat = os.path.join(out_dir, name + '.mat')
    if os.path.isfile(out_hea) and os.path.isfile(out_mat):
        return 'skipped'
    mat_file = header_file[:-4] + '.mat'
    if not os.path.isfile(mat_file):
        return 'no .mat'
    try:
        header = load_header(header_file)
        data, fs = preprocess_recording(loadmat(mat_file)['val'], get_frequency(header))
        data = np.clip(np.round(data), -32768, 32767).astype(np.int16)
        savemat(out_mat, {'val': data}, do_compression=False)
        # Write the header last so a record only counts as done once both files exist
        with open(out_hea, 'w') as f:
            f.write(rewrite_header(header, fs, data))
        return 'done'
    except Exception as e:  # keep going; report at the end
        for p in (out_mat, out_hea):
            if os.path.exists(p):
                os.remove(p)
        return f'error: {header_file}: {e}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cinc_dir', default=os.path.expanduser('~/data/CinC'),
                        help='Raw CinC 2021 download (searched recursively for the database folders)')
    parser.add_argument('--out_dir', default=os.path.join(REPO, 'data', 'PhysioNet2021_preprocessed'))
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    cinc_dir = os.path.expanduser(args.cinc_dir)
    if not os.path.isdir(cinc_dir):
        sys.exit(f'{cinc_dir} does not exist')

    found = find_source_dirs(cinc_dir)
    errors = []
    for target, source_dir in found.items():
        if source_dir is None:
            print(f'{target}: NOT FOUND under {cinc_dir} - skipped')
            continue
        out = os.path.join(args.out_dir, target)
        os.makedirs(out, exist_ok=True)
        headers = sorted(glob.glob(os.path.join(source_dir, '**', '*.hea'), recursive=True))
        if len({os.path.basename(h) for h in headers}) != len(headers):
            sys.exit(f'{target}: duplicate record names in {source_dir}; cannot flatten')
        jobs = [(h, out) for h in headers]
        counts = {}
        with Pool(args.workers) as pool:
            for result in tqdm(pool.imap_unordered(process, jobs, chunksize=16), total=len(jobs), desc=target):
                key = 'error' if result.startswith('error') else result
                counts[key] = counts.get(key, 0) + 1
                if key == 'error':
                    errors.append(result)
        print(f'{target}: {source_dir} -> {out}: ' + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))

    for e in errors[:20]:
        print(e)
    if errors:
        sys.exit(f'{len(errors)} recordings failed; re-run to retry them')


if __name__ == '__main__':
    main()
