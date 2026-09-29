"""Generate a tiny synthetic dataset in the PhysioNet 2021 layout for smoke-testing the pipeline.

Creates one directory per source database (WFDB .hea/.mat pairs with random signals and
random AF/SR/MI labels) plus a matching dataset_division.json with 5 folds and all sex ratios.

Usage:
    python scripts/make_dummy_data.py --out dummy_data
    python train_pipeline.py --data_dir dummy_data --model cnn --epochs 1 --batch_size 8
"""
import argparse
import json
import os

import numpy as np
from scipy.io import savemat

SOURCES = ['WFDB_PTBXL', 'WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_ChapmanShaoxing', 'WFDB_Ningbo']
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LABELS = ['164889003', '426783006', '164865005']  # AF, SR, MI
RATIOS = ['100_0', '75_25', '50_50', '25_75', '0_100']


def write_record(directory, name, fs, n_samples, sex, dx, rng):
    data = (rng.standard_normal((12, n_samples)) * 1000).astype(np.int16)
    savemat(os.path.join(directory, name + '.mat'), {'val': data})
    lines = [f'{name} 12 {fs} {n_samples}']
    lines += [f'{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}' for lead in LEADS]
    lines += [f'#Age: {rng.integers(20, 90)}', f'#Sex: {sex}', f'#Dx: {dx}', '#Rx: Unknown', '#Hx: Unknown', '#Sx: Unknown']
    with open(os.path.join(directory, name + '.hea'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='dummy_data')
    parser.add_argument('--per_source', type=int, default=20, help='records per source database')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sexes = []
    for src in SOURCES:
        directory = os.path.join(args.out, src)
        os.makedirs(directory, exist_ok=True)
        for i in range(args.per_source):
            sex = 'Male' if i % 2 == 0 else 'Female'
            fs = 1000 if src == 'WFDB_PTBXL' and i % 3 == 0 else 500
            n_samples = int(rng.integers(3000, 6000)) * fs // 500  # includes records shorter than 4096
            dx = LABELS[i % 3]
            write_record(directory, f'{src}_{i:04d}', fs, n_samples, sex, dx, rng)
            sexes.append(sex)

    # Indices follow the order produced by data.data_loader.find_challenge_files (sorted per directory).
    males = [i for i, s in enumerate(sexes) if s == 'Male']
    females = [i for i, s in enumerate(sexes) if s == 'Female']
    division = {}
    for fold in range(5):
        m = list(rng.permutation(males))
        f = list(rng.permutation(females))
        n_test = max(2, len(m) // 5)
        entry = {'male_balanced_test_idx': [int(x) for x in m[:n_test]],
                 'female_balanced_test_idx': [int(x) for x in f[:n_test]]}
        m_rest, f_rest = m[n_test:], f[n_test:]
        n = min(len(m_rest), len(f_rest))
        for ratio in RATIOS:
            n_m = n * int(ratio.split('_')[0]) // 100
            n_f = n - n_m
            pool = [int(x) for x in m_rest[:n_m] + f_rest[:n_f]]
            n_val = max(2, len(pool) // 5)
            entry[f'val_idx_{ratio}'] = pool[:n_val]
            entry[f'train_idx_{ratio}'] = pool[n_val:]
        division[str(fold)] = entry

    with open(os.path.join(args.out, 'dataset_division.json'), 'w') as fh:
        json.dump(division, fh)
    print(f'Wrote {len(sexes)} records and dataset_division.json to {args.out}')


if __name__ == '__main__':
    main()
