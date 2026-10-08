"""Build the train/val/test split (dataset_division.json) used by train_pipeline.py.

Follows Section 2.2, Figure 2 and Tables 3-4 of Galanty et al. (MELBA 2025):
  1. Exclude recordings of patients younger than 18 or with missing sex or age.
  2. Give every recording a pseudo patient ID from (age, sex, source dataset); all recordings that
     share it stay in the same fold, as the challenge data has no real patient IDs.
  3. Bin age into 18-39, 40-59, 60-79 and 80+ and split into 5 folds, stratified by age group, sex,
     source dataset and diagnostic category, keeping pseudo-ID groups together.
  4. Test sets (each fold once): for every diagnostic category, the same number of women and men
     (all of the smaller sex), sampled stratified by age group and source dataset.
  5. Training sets from the other folds with 0/25/50/75/100% women. The female share holds within
     every diagnostic category, each sex is sampled stratified by age group and source dataset, and
     the size is the same for every ratio and every fold. 10% of each training set is used for
     validation, sampled so it keeps the diagnostic distribution (and sex mix).

Diagnostic categories are label combinations of AF/SR/MI; recordings with none of the three form
"Other" (all-zero targets). The paper does not say how Other was chosen for the test sets, where
Table 4 has about 200 per sex (3.5% of the AF/SR/MI recordings) out of thousands available; the
same share is used here by default (test_other_fraction).

Indices refer to positions in the dataset that train_pipeline.py builds: every recording in the
WFDB_* folders (in SOURCES order, files sorted by name), not only the eligible ones.
"""
import os
import warnings
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data.data_loader import dataset, find_challenge_files
from data.preprocessing import SOURCES

RATIOS = ['100_0', '75_25', '50_50', '25_75', '0_100']  # male_female, in percent
TARGET_CODES = ['164889003', '426783006', '164865005']  # AF, SR, MI (after merging equivalent classes)
AGE_BIN_EDGES = [18, 40, 60, 80]  # bins [18, 40), [40, 60), [60, 80), [80, inf)
MIN_AGE = 18
OTHER = '000'  # label combination of recordings with none of AF/SR/MI
TEST_OTHER_FRACTION = 0.035  # Table 4: ~205 Other vs ~5850 AF/SR/MI test recordings per sex


def dataset_header_files(data_dir):
    """Header files in exactly the order train_pipeline.py indexes them."""
    header_files, _ = find_challenge_files([os.path.join(data_dir, db) for db in SOURCES])
    return header_files


def record_ids(header_files, data_dir):
    return [os.path.relpath(h, data_dir) for h in header_files]


def load_metadata(header_files):
    """AF/SR/MI label matrix, sex, age and source dataset of every recording, parsed as during training.

    The source is the dataset label from data_loader.dataset_labels (CPSC and CPSC-Extra share one).
    """
    ds = dataset(header_files, length=4096, nr_leads=12, equivalent_cl='sinus_mi')
    cols = [dataset.classes.index(c) for c in TARGET_CODES]
    labels = np.stack(ds.files['target'].to_list())[:, cols].astype(int)
    return labels, ds.files['sex'].tolist(), ds.files['age'].tolist(), ds.files['source'].tolist()


def age_bin(age, edges=AGE_BIN_EDGES):
    return int(np.searchsorted(edges, age, side='right')) - 1


def allocate(n, sizes):
    """Split n into integer parts proportional to sizes (largest-remainder rounding)."""
    sizes = np.asarray(sizes, dtype=float)
    if n <= 0 or sizes.sum() == 0:
        return np.zeros(len(sizes), dtype=int)
    exact = n * sizes / sizes.sum()
    alloc = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - alloc), kind='stable')[:n - alloc.sum()]:
        alloc[j] += 1
    return alloc


def stratified_sample(idx, key, n, rank):
    """Pick n of idx so that every stratum key(i) keeps its share (largest-remainder rounding).

    Within a stratum, recordings are taken in the fixed random order given by rank, so asking for
    more recordings from the same pool extends the earlier selection.
    """
    idx = list(idx)
    if n >= len(idx):
        return idx
    strata = defaultdict(list)
    for i in idx:
        strata[key(i)].append(i)
    keys = sorted(strata)
    alloc = allocate(n, [len(strata[k]) for k in keys])
    chosen = []
    for k, a in zip(keys, alloc):
        chosen += sorted(strata[k], key=rank.__getitem__)[:a]
    return chosen


def build_division(labels, sexes, ages, sources, n_folds=5, val_fraction=0.1,
                   test_other_fraction=TEST_OTHER_FRACTION, seed=42):
    labels = np.asarray(labels)
    n = len(labels)
    excluded = {'sex not Male/Female': 0, 'age missing or under 18': 0}
    eligible = []
    for i in range(n):
        if sexes[i] not in ('Male', 'Female'):
            excluded['sex not Male/Female'] += 1
        elif ages[i] is None or not np.isfinite(ages[i]) or ages[i] < MIN_AGE:
            excluded['age missing or under 18'] += 1
        else:
            eligible.append(i)
    eligible = np.array(eligible, dtype=int)

    category = {int(i): ''.join(map(str, labels[i])) for i in eligible}  # e.g. '100' = AF only, '000' = Other
    agebin = {int(i): age_bin(ages[i]) for i in eligible}
    age_source = lambda i: (agebin[i], sources[i])  # noqa: E731
    category_sex = lambda i: (category[i], sexes[i])  # noqa: E731
    categories = sorted(set(category.values()))

    # Pseudo patient IDs: recordings with the same age, sex and source are kept in one fold
    pseudo_ids = {}
    groups = np.array([pseudo_ids.setdefault((ages[i], sexes[i], sources[i]), len(pseudo_ids)) for i in eligible])
    strata = np.array([f'{agebin[i]}|{sexes[i]}|{sources[i]}|{category[i]}' for i in eligible])
    fold_of = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # strata smaller than n_folds
        kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, test_pos) in enumerate(kfold.split(eligible, strata, groups)):
            for i in eligible[test_pos]:
                fold_of[int(i)] = fold

    def by_category(idx):
        out = defaultdict(list)
        for i in idx:
            out[category[i]].append(i)
        return out

    in_fold, pool = [], []
    for fold in range(n_folds):
        in_fold.append({s: by_category(int(i) for i in eligible if fold_of[int(i)] == fold and sexes[i] == s)
                        for s in ('Male', 'Female')})
        pool.append({s: by_category(int(i) for i in eligible if fold_of[int(i)] != fold and sexes[i] == s)
                     for s in ('Male', 'Female')})

    # Training size per category: at most what the smaller sex has, so 0% and 100% women are possible.
    # The total is the same in every fold (the smallest fold's), shared out over the categories.
    caps = [[min(len(pool[f]['Male'][c]), len(pool[f]['Female'][c])) for c in categories] for f in range(n_folds)]
    n_train = min(sum(c) for c in caps)

    rng = np.random.default_rng(seed)
    rank = {int(i): r for r, i in enumerate(rng.permutation(eligible))}
    division, summary = {}, {}
    for fold in range(n_folds):
        # Test: per category the same number of women and men, stratified by age group and source
        test = {'Male': [], 'Female': []}
        for c in categories:
            if c == OTHER:
                continue
            n_c = min(len(in_fold[fold]['Male'][c]), len(in_fold[fold]['Female'][c]))
            for s in test:
                test[s] += stratified_sample(in_fold[fold][s][c], age_source, n_c, rank)
        n_other = min(round(test_other_fraction * len(test['Male'])),
                      len(in_fold[fold]['Male'][OTHER]), len(in_fold[fold]['Female'][OTHER]))
        for s in test:
            test[s] += stratified_sample(in_fold[fold][s][OTHER], age_source, n_other, rank)
        entry = {'male_balanced_test_idx': sorted(test['Male']),
                 'female_balanced_test_idx': sorted(test['Female'])}

        # Training: female share within every category, each sex stratified by age group and source
        per_category = dict(zip(categories, allocate(n_train, caps[fold])))
        for ratio in RATIOS:
            female_share = int(ratio.split('_')[1]) / 100
            chosen = []
            for c, n_c in per_category.items():
                n_female = round(n_c * female_share)
                chosen += stratified_sample(pool[fold]['Male'][c], age_source, n_c - n_female, rank)
                chosen += stratified_sample(pool[fold]['Female'][c], age_source, n_female, rank)
            val = set(stratified_sample(chosen, category_sex, round(val_fraction * len(chosen)), rank))
            entry[f'train_idx_{ratio}'] = sorted(i for i in chosen if i not in val)
            entry[f'val_idx_{ratio}'] = sorted(val)
        division[str(fold)] = entry
        summary[fold] = {'fold_size': sum(len(v) for s in in_fold[fold] for v in in_fold[fold][s].values()),
                         'test_per_sex': len(test['Male']), 'test_other_per_sex': n_other}
    info = {'recordings': n, 'eligible': len(eligible), 'excluded': excluded,
            'pseudo_ids': len(pseudo_ids), 'train_size': n_train}
    return division, summary, info


def describe(labels, ages, sources, idx):
    """One-line summary of a split: size, AF/SR/MI/Other counts, age-group and source shares."""
    idx = list(idx)
    if not idx:
        return 'empty'
    lab = np.asarray(labels)[idx]
    counts = list(lab.sum(axis=0)) + [int((lab.sum(axis=1) == 0).sum())]
    bins = np.bincount([age_bin(ages[i]) for i in idx], minlength=len(AGE_BIN_EDGES)) / len(idx)
    src = np.bincount([sources[i] for i in idx], minlength=5) / len(idx)
    return (f'n={len(idx):6d}  AF {counts[0]:5d} SR {counts[1]:5d} MI {counts[2]:5d} Other {counts[3]:5d}  '
            f'age {"/".join(f"{b:.2f}" for b in bins)}  source {"/".join(f"{s:.2f}" for s in src)}')
