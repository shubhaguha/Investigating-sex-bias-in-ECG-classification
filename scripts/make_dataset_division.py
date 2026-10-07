"""Generate dataset_division.json for train_pipeline.py from a preprocessed data directory.

See data/division.py for how the folds, sex-balanced test sets and sex-ratio training sets are built.
The file also stores the record list it was built for; train_pipeline.py checks it so a split is
never applied to a data directory with different contents.

Usage:
    python scripts/make_dataset_division.py --data_dir data/PhysioNet2021_preprocessed
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.division import (RATIOS, build_division, dataset_header_files, load_labels_and_sex,  # noqa: E402
                           prevalence, record_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PhysioNet2021_preprocessed'))
    parser.add_argument('--out', default=None, help='output file (default: <data_dir>/dataset_division.json)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--balance_train_labels', action='store_true',
                        help='give male and female training pools the same AF/SR/MI label distribution')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    header_files = dataset_header_files(args.data_dir)
    if not header_files:
        sys.exit(f'No recordings found in the WFDB_* folders of {args.data_dir}')
    labels, sexes = load_labels_and_sex(header_files)
    division, summary = build_division(labels, sexes, n_folds=args.folds, val_fraction=args.val_fraction,
                                       balance_train_labels=args.balance_train_labels, seed=args.seed)
    division['meta'] = {
        'records': record_ids(header_files, args.data_dir),
        'folds': args.folds, 'val_fraction': args.val_fraction,
        'balance_train_labels': args.balance_train_labels, 'seed': args.seed,
    }
    out = args.out or os.path.join(args.data_dir, 'dataset_division.json')
    with open(out, 'w') as f:
        json.dump(division, f)

    print(f'\n{len(header_files)} recordings, '
          f'{sum(1 for s in sexes if s == "Male")} male / {sum(1 for s in sexes if s == "Female")} female')
    for fold, s in summary.items():
        d = division[str(fold)]
        print(f'\nFold {fold}: training pools {s["pool_male"]} M / {s["pool_female"]} F -> '
              f'{s["train_total"]} train+val per ratio, {s["test_per_sex"]} test per sex')
        print(f'  test   male   {prevalence(labels, d["male_balanced_test_idx"])}')
        print(f'  test   female {prevalence(labels, d["female_balanced_test_idx"])}')
        for ratio in RATIOS:
            print(f'  train  {ratio:6s} {prevalence(labels, d[f"train_idx_{ratio}"])}   '
                  f'val n={len(d[f"val_idx_{ratio}"])}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
