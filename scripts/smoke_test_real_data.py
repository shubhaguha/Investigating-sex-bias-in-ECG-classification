"""Smoke-test the training pipeline on a small sample of the real PhysioNet/CinC 2021 data.

Works directly on the raw challenge download (e.g. training/<database>/g*/*.hea) or on a
preprocessed layout with flat WFDB_* folders. It:
  1. finds the six databases used in the paper,
  2. samples a few labelled (AF / SR / MI) recordings with known sex from each,
  3. symlinks them into the flat WFDB_* layout train_pipeline.py expects,
  4. writes a matching dataset_division.json (5 folds, every sex ratio),
  5. runs train_pipeline.py for 1 epoch with each model.

Usage:
    python scripts/smoke_test_real_data.py --cinc_dir ~/data/CinC
    python scripts/smoke_test_real_data.py --cinc_dir ~/data/CinC --models cnn --per_source 100 --epochs 2

Nothing in --cinc_dir is modified; the sample lives in --work_dir (default: smoke_test/).
This checks that the code runs on real files; the numbers it prints are not meaningful.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.data_loader import load_header, get_sex, get_labels, get_frequency  # noqa: E402

# Target folder -> folder names it may have in the raw download (lower-cased) or a preprocessed copy
SOURCES = {
    'WFDB_PTBXL': ['ptb-xl', 'ptbxl', 'wfdb_ptbxl'],
    'WFDB_CPSC2018': ['cpsc_2018', 'cpsc2018', 'wfdb_cpsc2018'],
    'WFDB_CPSC2018_2': ['cpsc_2018_extra', 'cpsc2018_2', 'wfdb_cpsc2018_2'],
    'WFDB_Ga': ['georgia', 'ga', 'wfdb_ga'],
    'WFDB_ChapmanShaoxing': ['chapman_shaoxing', 'chapmanshaoxing', 'wfdb_chapmanshaoxing'],
    'WFDB_Ningbo': ['ningbo', 'wfdb_ningbo'],
}
TARGET_CODES = {'164889003',                            # AF
                '426783006', '426177001', '427084000',  # SR (+ brady/tachy)
                '57054005', '54329005', '164865005'}    # MI
RATIOS = ['100_0', '75_25', '50_50', '25_75', '0_100']


def find_source_dirs(cinc_dir):
    found = {k: None for k in SOURCES}
    for root, dirs, _ in os.walk(cinc_dir):
        for d in dirs:
            for target, aliases in SOURCES.items():
                if found[target] is None and d.lower() in aliases:
                    found[target] = os.path.join(root, d)
    return found


def eligible_headers(source_dir):
    """Headers with a .mat file, a target label and Male/Female sex, grouped by sex."""
    by_sex = {'Male': [], 'Female': []}
    for h in sorted(glob.glob(os.path.join(source_dir, '**', '*.hea'), recursive=True)):
        if not os.path.isfile(h[:-4] + '.mat'):
            continue
        hdr = load_header(h)
        sex = get_sex(hdr)
        if sex not in by_sex or get_frequency(hdr) is None:
            continue
        if TARGET_CODES & set(get_labels(hdr, [])):
            by_sex[sex].append(h)
    return by_sex


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cinc_dir', default=os.path.expanduser('~/data/CinC'))
    parser.add_argument('--work_dir', default=os.path.join(REPO, 'smoke_test'))
    parser.add_argument('--per_source', type=int, default=40, help='recordings sampled per database (half male, half female)')
    parser.add_argument('--models', nargs='+', default=['cnn', 'resnet_attention', 'xresnet101'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cinc_dir = os.path.expanduser(args.cinc_dir)
    if not os.path.isdir(cinc_dir):
        sys.exit(f'{cinc_dir} does not exist')
    rng = random.Random(args.seed)

    found = find_source_dirs(cinc_dir)
    data_dir = os.path.join(args.work_dir, 'data')
    sexes = []
    print('Sampling recordings:')
    for target, source_dir in found.items():
        out = os.path.join(data_dir, target)
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(out):  # clear links from a previous run
            os.remove(os.path.join(out, f))
        if source_dir is None:
            print(f'  {target:22s} NOT FOUND (still downloading?) - skipped')
            continue
        by_sex = eligible_headers(source_dir)
        chosen = []
        for sex in ('Male', 'Female'):
            chosen += [(h, sex) for h in rng.sample(by_sex[sex], min(args.per_source // 2, len(by_sex[sex])))]
        for h, sex in chosen:
            name = os.path.basename(h)[:-4]
            link(h, os.path.join(out, name + '.hea'))
            link(h[:-4] + '.mat', os.path.join(out, name + '.mat'))
        print(f'  {target:22s} {source_dir}: {len(by_sex["Male"])} M / {len(by_sex["Female"])} F eligible, '
              f'{len(chosen)} sampled')

    # Same ordering as data.data_loader.find_challenge_files: databases in this order, files sorted by name
    for target in SOURCES:
        out = os.path.join(data_dir, target)
        for f in sorted(os.listdir(out)):
            if f.endswith('.hea'):
                sexes.append(get_sex(load_header(os.path.join(out, f))))

    males = [i for i, s in enumerate(sexes) if s == 'Male']
    females = [i for i, s in enumerate(sexes) if s == 'Female']
    if min(len(males), len(females)) < 10:
        sys.exit(f'Too few usable recordings ({len(males)} M / {len(females)} F); check --cinc_dir')

    division = {}
    for fold in range(5):
        m, f = males[:], females[:]
        rng.shuffle(m)
        rng.shuffle(f)
        n_test = max(2, min(len(m), len(f)) // 5)
        entry = {'male_balanced_test_idx': m[:n_test], 'female_balanced_test_idx': f[:n_test]}
        m_rest, f_rest = m[n_test:], f[n_test:]
        n = min(len(m_rest), len(f_rest))
        for ratio in RATIOS:
            n_m = n * int(ratio.split('_')[0]) // 100
            pool = m_rest[:n_m] + f_rest[:n - n_m]
            rng.shuffle(pool)
            n_val = max(2, len(pool) // 5)
            entry[f'val_idx_{ratio}'] = pool[:n_val]
            entry[f'train_idx_{ratio}'] = pool[n_val:]
        division[str(fold)] = entry
    with open(os.path.join(data_dir, 'dataset_division.json'), 'w') as fh:
        json.dump(division, fh)
    print(f'Total: {len(sexes)} recordings ({len(males)} M / {len(females)} F)\n')

    failed = []
    for model in args.models:
        print(f'===== {model} =====', flush=True)
        cmd = [sys.executable, os.path.join(REPO, 'train_pipeline.py'), '--data_dir', data_dir,
               '--model', model, '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
               '--sex_ratio', '50_50', '--fold', '0', '--experiment_id', f'smoke_{model}']
        if subprocess.run(cmd, cwd=args.work_dir).returncode != 0:
            failed.append(model)

    print('\n===== Summary =====')
    for model in args.models:
        print(f'  {model:18s} {"FAILED" if model in failed else "OK"}')
    print(f'Outputs: {os.path.join(args.work_dir, "results")}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
