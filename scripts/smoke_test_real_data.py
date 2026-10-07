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
from data.preprocessing import find_source_dirs  # noqa: E402
from data.division import build_division, dataset_header_files, load_labels_and_sex, record_ids  # noqa: E402

TARGET_CODES = {'164889003',                            # AF
                '426783006', '426177001', '427084000',  # SR (+ brady/tachy)
                '57054005', '54329005', '164865005'}    # MI


def eligible_headers(source_dir):
    """Headers with a .mat file, a target label and Male/Female sex, grouped by sex.

    Also returns counts of why the other headers were rejected."""
    by_sex = {'Male': [], 'Female': []}
    skipped = {'headers': 0, 'no .mat': 0, 'sex not Male/Female': 0, 'no AF/SR/MI label': 0}
    for h in sorted(glob.glob(os.path.join(source_dir, '**', '*.hea'), recursive=True)):
        skipped['headers'] += 1
        if not os.path.isfile(h[:-4] + '.mat'):
            skipped['no .mat'] += 1
            continue
        hdr = load_header(h)
        sex = get_sex(hdr)
        if sex not in by_sex or get_frequency(hdr) is None:
            skipped['sex not Male/Female'] += 1
            continue
        if TARGET_CODES & set(get_labels(hdr, [])):
            by_sex[sex].append(h)
        else:
            skipped['no AF/SR/MI label'] += 1
    return by_sex, skipped


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
    print('Sampling recordings:')
    for target, source_dir in found.items():
        out = os.path.join(data_dir, target)
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(out):  # clear links from a previous run
            os.remove(os.path.join(out, f))
        if source_dir is None:
            print(f'  {target:22s} NOT FOUND (still downloading?) - skipped')
            continue
        by_sex, skipped = eligible_headers(source_dir)
        chosen = []
        for sex in ('Male', 'Female'):
            chosen += [(h, sex) for h in rng.sample(by_sex[sex], min(args.per_source // 2, len(by_sex[sex])))]
        for h, sex in chosen:
            name = os.path.basename(h)[:-4]
            link(h, os.path.join(out, name + '.hea'))
            link(h[:-4] + '.mat', os.path.join(out, name + '.mat'))
        print(f'  {target:22s} {source_dir}: {len(by_sex["Male"])} M / {len(by_sex["Female"])} F eligible, '
              f'{len(chosen)} sampled')
        print('      ' + ', '.join(f'{k}: {v}' for k, v in skipped.items()))

    header_files = dataset_header_files(data_dir)
    labels, sexes = load_labels_and_sex(header_files)
    n_m, n_f = sexes.count('Male'), sexes.count('Female')
    if min(n_m, n_f) < 10:
        sys.exit(f'Too few usable recordings ({n_m} M / {n_f} F); see the counts above. '
                 'If "headers" is 0, point --cinc_dir at the downloaded data; if "no .mat" matches it, '
                 'the signal files have not been downloaded yet.')
    division, _ = build_division(labels, sexes, seed=args.seed)
    division['meta'] = {'records': record_ids(header_files, data_dir)}
    with open(os.path.join(data_dir, 'dataset_division.json'), 'w') as fh:
        json.dump(division, fh)
    print(f'Total: {len(sexes)} recordings ({n_m} M / {n_f} F)\n')

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
