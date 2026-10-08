"""Generate dataset_division.json for train_pipeline.py from a preprocessed data directory.

See data/division.py for how the folds, sex-balanced test sets and sex-ratio training sets are built
(pseudo patient IDs, age groups, stratification by age group, sex, source dataset and diagnosis).
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
from data.division import (AGE_BIN_EDGES, RATIOS, TEST_OTHER_FRACTION, build_division, dataset_header_files,  # noqa: E402
                           describe, load_metadata, record_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default=os.path.join(REPO, 'data', 'PhysioNet2021_preprocessed'))
    parser.add_argument('--out', default=None, help='output file (default: <data_dir>/dataset_division.json)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--test_other_fraction', type=float, default=TEST_OTHER_FRACTION,
                        help='Other (no AF/SR/MI) test recordings per sex, as a fraction of the AF/SR/MI ones')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    header_files = dataset_header_files(args.data_dir)
    if not header_files:
        sys.exit(f'No recordings found in the WFDB_* folders of {args.data_dir}')
    labels, sexes, ages, sources = load_metadata(header_files)
    division, summary, info = build_division(labels, sexes, ages, sources, n_folds=args.folds,
                                             val_fraction=args.val_fraction,
                                             test_other_fraction=args.test_other_fraction, seed=args.seed)
    division['meta'] = {
        'records': record_ids(header_files, args.data_dir),
        'folds': args.folds, 'val_fraction': args.val_fraction, 'seed': args.seed,
        'age_bin_edges': AGE_BIN_EDGES, 'test_other_fraction': args.test_other_fraction,
    }
    out = args.out or os.path.join(args.data_dir, 'dataset_division.json')
    with open(out, 'w') as f:
        json.dump(division, f)

    print(f'\n{info["recordings"]} recordings, {info["eligible"]} eligible in {info["pseudo_ids"]} pseudo-IDs; excluded: '
          + ', '.join(f'{k} {v}' for k, v in info['excluded'].items()))
    print('(paper, Figure 2: 87663 recordings, 85025 eligible, 749 pseudo-IDs, folds of 16297-18041)')
    print(f'Training+validation size for every fold and ratio: {info["train_size"]} (paper, Table 4: ~32300)')
    print('Columns: size, AF/SR/MI/Other counts, share per age group '
          f'({", ".join(f"{a}+" for a in AGE_BIN_EDGES)}), share per source (PTB-XL/CPSC/Georgia/Chapman/Ningbo)')
    for fold, s in summary.items():
        d = division[str(fold)]
        print(f'\nFold {fold}: {s["fold_size"]} recordings, {s["test_per_sex"]} test per sex '
              f'(incl. {s["test_other_per_sex"]} Other)')
        print(f'  test   male   {describe(labels, ages, sources, d["male_balanced_test_idx"])}')
        print(f'  test   female {describe(labels, ages, sources, d["female_balanced_test_idx"])}')
        for ratio in RATIOS:
            print(f'  train  {ratio:6s} {describe(labels, ages, sources, d[f"train_idx_{ratio}"])}')
            print(f'  val    {ratio:6s} {describe(labels, ages, sources, d[f"val_idx_{ratio}"])}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
