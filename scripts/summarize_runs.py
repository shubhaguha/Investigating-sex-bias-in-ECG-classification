"""Summarize the runs in results/: runtime, memory, disk usage and test metrics per run, and their mean over runs.

Reads results/<experiment>/{args,resources,test_metrics}.json and gpu_usage.csv, written by
train_pipeline.py and the slurm/train_*.sh jobs, and measures what each run left on disk. Test metrics
(ROC AUC, partial ROC AUC up to FPR 0.2, PR AUC) are given per class and sex with 6 decimals. Runs are
then grouped by model and sex ratio (all folds and seeds together) with the total runtime and the mean
± SD of every metric. Writes results/summary.csv (one row per run) and results/summary_by_ratio.csv.

    python scripts/summarize_runs.py [--results_dir results]
"""
import argparse
import csv
import glob
import json
import os
import statistics

CLASSES = ['AF', 'SR', 'MI']
SEXES = ['female', 'male']
METRICS = ['roc_auc', 'p_roc_auc', 'pr_auc']
METRIC_TITLES = {'roc_auc': 'ROC AUC', 'p_roc_auc': 'partial ROC AUC (FPR <= 0.2)', 'pr_auc': 'PR AUC'}
OLD_METRIC_KEYS = {'roc_auc': 'test_roc', 'p_roc_auc': 'p_roc_auc', 'pr_auc': 'pr_auc'}


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_test_metrics(path):
    """{sex: {class: {metric: value}}}. Also reads the older layout, {sex: {'test_roc': [AF, SR, MI], ...}}."""
    metrics = load_json(path)
    if metrics is None:
        return None
    for sex in SEXES:
        if 'test_roc' in metrics[sex]:
            metrics[sex] = {name: {m: metrics[sex][OLD_METRIC_KEYS[m]][i] for m in METRICS}
                            for i, name in enumerate(CLASSES)}
    return metrics


def metric_columns():
    return [f'{m}_{name}_{sex}' for m in METRICS for sex in SEXES for name in CLASSES]


def gpu_usage(path):
    """Mean GPU utilization (%) and peak GPU memory used (MiB) from nvidia-smi samples."""
    if not os.path.isfile(path):
        return None, None
    util, memory = [], []
    with open(path) as f:
        for row in list(csv.reader(f))[1:]:
            try:
                util.append(float(row[1]))
                memory.append(float(row[2]))
            except (IndexError, ValueError):
                continue
    if not util:
        return None, None
    return round(statistics.mean(util)), round(max(memory))


def size_mb(path):
    """Disk usage of a file or directory tree, in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / 2 ** 20
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2 ** 20


def disk_usage(run_dir):
    """Size of the saved weights and of everything a run wrote (results dir, weights, history, TensorBoard logs)."""
    results_dir = os.path.dirname(run_dir)
    experiment = os.path.basename(run_dir)
    weights = os.path.join(results_dir, 'model_weights', f'{experiment}.pth')
    weights_mb = size_mb(weights) if os.path.isfile(weights) else None
    paths = [run_dir, weights, os.path.join(results_dir, 'model_weights', f'PROGRESS_{experiment}.pickle')]
    # SummaryWriter(comment='_' + experiment) logs to runs/<date>_<host>_<experiment>
    paths += glob.glob(os.path.join('runs', f'*_{experiment}'))
    total_mb = sum(size_mb(p) for p in paths if os.path.exists(p))
    return (round(weights_mb, 1) if weights_mb is not None else None), round(total_mb, 1)


def summarize(run_dir):
    args = load_json(os.path.join(run_dir, 'args.json'))
    resources = load_json(os.path.join(run_dir, 'resources.json'))
    if args is None or resources is None:
        return None
    seconds = resources['seconds']
    gpu_util, gpu_memory_used = gpu_usage(os.path.join(run_dir, 'gpu_usage.csv'))
    weights_mb, disk_mb = disk_usage(run_dir)
    row = {
        'experiment': os.path.basename(run_dir),
        'model': args['model'], 'sex_ratio': args['sex_ratio'], 'fold': args['fold'], 'seed': args['seed'],
        'gpu': resources['gpu'], 'epochs': resources['epochs'],
        'total_min': round(seconds['total'] / 60, 1),
        'load_min': round(seconds['load_dataset'] / 60, 1),
        'train_min': round(seconds['train'] / 60, 1),
        'mean_epoch_min': round(resources['mean_epoch_seconds'] / 60, 1),
        'peak_gpu_allocated_gb': round(resources['peak_gpu_allocated_mb'] / 1024, 1)
        if resources['peak_gpu_allocated_mb'] is not None else None,
        'peak_gpu_used_gb': round(gpu_memory_used / 1024, 1) if gpu_memory_used is not None else None,
        'mean_gpu_util_pct': gpu_util,
        'peak_host_memory_gb': round(resources['peak_host_memory_mb'] / 1024, 1),
        'weights_mb': weights_mb,
        'disk_mb': disk_mb,
    }
    metrics = load_test_metrics(os.path.join(run_dir, 'test_metrics.json'))
    for m in METRICS:
        for sex in SEXES:
            for name in CLASSES:
                row[f'{m}_{name}_{sex}'] = round(metrics[sex][name][m], 6) if metrics else None
    return row


def print_table(rows, columns, headers=None):
    headers = headers or columns
    cells = [[f'{r[c]:.6f}' if isinstance(r[c], float) and c in METRIC_COLUMNS else str(r[c]) for c in columns]
             for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]
    print('  '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in cells:
        print('  '.join(v.ljust(w) for v, w in zip(row, widths)))


METRIC_COLUMNS = set(metric_columns())


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return f'{values[0]:.6f}'
    return f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    rows = [r for r in (summarize(d) for d in sorted(glob.glob(os.path.join(args.results_dir, '*'))))
            if r is not None]
    if not rows:
        print(f'No finished runs in {args.results_dir}/')
        return
    # ratios are male_female; order by female share (F0, F25, ..., F100)
    rows.sort(key=lambda r: (r['model'], int(r['sex_ratio'].split('_')[1]), r['fold'], r['seed']))

    print('Runtime, memory and disk usage per run')
    print_table(rows, ['experiment', 'gpu', 'epochs', 'total_min', 'load_min', 'train_min', 'mean_epoch_min',
                       'peak_gpu_allocated_gb', 'peak_gpu_used_gb', 'mean_gpu_util_pct', 'peak_host_memory_gb',
                       'weights_mb', 'disk_mb'])
    print(f"Total disk used by these runs: {sum(r['disk_mb'] for r in rows) / 1024:.2f} GB")
    print()

    class_columns = [f'{name}_{sex}' for sex in SEXES for name in CLASSES]
    for m in METRICS:
        print(f'Test {METRIC_TITLES[m]} per run')
        print_table(rows, ['experiment'] + [f'{m}_{c}' for c in class_columns], ['experiment'] + class_columns)
        print()

    # Mean over all runs of a model and sex ratio: the folds of train_best.sh, the seeds of train_grid.sh
    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['sex_ratio']), []).append(r)
    means = []
    for (model, ratio), group in groups.items():
        mean = {'model': model, 'sex_ratio (M_F)': ratio, 'runs': len(group),
                'folds': ','.join(str(f) for f in sorted({r['fold'] for r in group})),
                'seeds': ','.join(str(s) for s in sorted({r['seed'] for r in group})),
                'total_runtime_h': f"{sum(r['total_min'] for r in group) / 60:.2f}",
                'mean_runtime_h': f"{statistics.mean(r['total_min'] for r in group) / 60:.2f}"}
        for c in metric_columns():
            mean[c] = mean_sd([r[c] for r in group])
        means.append(mean)
    group_columns = ['model', 'sex_ratio (M_F)', 'runs', 'folds', 'seeds']
    print('Runtime per model and sex ratio (sum and mean over its runs)')
    print_table(means, group_columns + ['total_runtime_h', 'mean_runtime_h'])
    print(f"Total runtime of all runs: {sum(r['total_min'] for r in rows) / 60:.2f} h")
    print()
    for m in METRICS:
        print(f'Test {METRIC_TITLES[m]}, mean ± SD over the runs of each model and sex ratio')
        print_table(means, group_columns + [f'{m}_{c}' for c in class_columns], group_columns + class_columns)
        print()

    out = os.path.join(args.results_dir, 'summary.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_means = os.path.join(args.results_dir, 'summary_by_ratio.csv')
    with open(out_means, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    print(f'Wrote {out} (one row per run) and {out_means} (mean ± SD per model and sex ratio)')


if __name__ == '__main__':
    main()
