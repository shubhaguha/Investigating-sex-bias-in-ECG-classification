"""Build the train/val/test split (dataset_division.json) used by train_pipeline.py.

Indices refer to positions in the dataset that train_pipeline.py builds: every recording in the
WFDB_* folders (in SOURCES order, files sorted by name), not only the eligible ones.
"""
import os
from collections import defaultdict

import numpy as np

from data.data_loader import dataset, find_challenge_files
from data.preprocessing import SOURCES

RATIOS = ['100_0', '75_25', '50_50', '25_75', '0_100']  # male_female, in percent
TARGET_CODES = ['164889003', '426783006', '164865005']  # AF, SR, MI (after merging equivalent classes)


def dataset_header_files(data_dir):
    """Header files in exactly the order train_pipeline.py indexes them."""
    header_files, _ = find_challenge_files([os.path.join(data_dir, db) for db in SOURCES])
    return header_files


def record_ids(header_files, data_dir):
    return [os.path.relpath(h, data_dir) for h in header_files]


def load_labels_and_sex(header_files):
    """(n, 3) AF/SR/MI label matrix and the sex of every recording, parsed as during training."""
    ds = dataset(header_files, length=4096, nr_leads=12, equivalent_cl='sinus_mi')
    cols = [dataset.classes.index(c) for c in TARGET_CODES]
    labels = np.stack(ds.files['target'].to_list())[:, cols].astype(int)
    return labels, ds.files['sex'].tolist()


def build_division(labels, sexes, n_folds=5, val_fraction=0.1, balance_train_labels=False, seed=42):
    """Sex-stratified cross-validation split.

    - Eligible recordings: sex Male/Female and at least one of AF/SR/MI.
    - Each sex is split into n_folds folds, stratified by label combination.
    - Test sets (fold k): equal numbers of male and female recordings for every label combination,
      so both test sets have the same size and label distribution.
    - Training (other folds): one fixed total size N per fold for all ratios, so only the sex mix
      changes. N = the size of the smaller sex's pool, so that 0% and 100% female are both possible.
      Sets are nested: the males in 75_25 are a subset of those in 100_0, and so on.
    - Validation: every k-th recording of each training set (k = round(1 / val_fraction)), so it has
      the same sex ratio as the training set and recordings stay in val across ratios.
    - balance_train_labels: also give both sexes' training pools the same label distribution
      (smaller N, but sex is no longer confounded with disease prevalence).
    """
    rng = np.random.default_rng(seed)
    strata = {'Male': defaultdict(list), 'Female': defaultdict(list)}
    for i, (lab, sex) in enumerate(zip(labels, sexes)):
        if sex in strata and lab.any():
            strata[sex][tuple(int(x) for x in lab)].append(i)

    # Stratified fold assignment per sex
    fold_of = {}
    for sex in strata:
        for key in sorted(strata[sex]):
            idx = np.array(strata[sex][key])
            rng.shuffle(idx)
            offset = rng.integers(n_folds)
            for pos, i in enumerate(idx):
                fold_of[int(i)] = (pos + offset) % n_folds

    keys = sorted(set(strata['Male']) | set(strata['Female']))
    val_every = max(2, round(1 / val_fraction))
    division, summary = {}, {}
    for fold in range(n_folds):
        entry = {}
        test = {'Male': [], 'Female': []}
        pool = {'Male': {}, 'Female': {}}
        for key in keys:
            cand = {}
            for sex in strata:
                in_key = strata[sex].get(key, [])
                cand[sex] = [i for i in in_key if fold_of[i] == fold]
                pool[sex][key] = [i for i in in_key if fold_of[i] != fold]
            # same number of test recordings of this label combination for both sexes
            n = min(len(cand['Male']), len(cand['Female']))
            for sex in strata:
                test[sex] += cand[sex][:n]
            if balance_train_labels:
                n = min(len(pool['Male'][key]), len(pool['Female'][key]))
                for sex in strata:
                    pool[sex][key] = pool[sex][key][:n]
        entry['male_balanced_test_idx'] = sorted(test['Male'])
        entry['female_balanced_test_idx'] = sorted(test['Female'])

        flat = {}
        for sex in strata:
            flat[sex] = np.array([i for key in keys for i in pool[sex][key]], dtype=int)
            rng.shuffle(flat[sex])
        n_total = min(len(flat['Male']), len(flat['Female']))
        for ratio in RATIOS:
            n_m = round(n_total * int(ratio.split('_')[0]) / 100)
            chosen = {'Male': flat['Male'][:n_m], 'Female': flat['Female'][:n_total - n_m]}
            train, val = [], []
            for sex in chosen:
                for pos, i in enumerate(chosen[sex]):
                    (val if pos % val_every == 0 else train).append(int(i))
            entry[f'train_idx_{ratio}'] = sorted(train)
            entry[f'val_idx_{ratio}'] = sorted(val)
        division[str(fold)] = entry
        summary[fold] = {'test_per_sex': len(test['Male']), 'train_total': n_total,
                         'pool_male': len(flat['Male']), 'pool_female': len(flat['Female'])}
    return division, summary


def prevalence(labels, idx):
    idx = list(idx)
    if not idx:
        return 'empty'
    p = labels[idx].mean(axis=0)
    return f'n={len(idx):6d}  AF {p[0]:.3f}  SR {p[1]:.3f}  MI {p[2]:.3f}'
