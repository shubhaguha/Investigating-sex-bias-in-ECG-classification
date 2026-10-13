"""Summarize the runs in results/: runtime, memory and test ROC AUC per run, and the mean over seeds.

Reads results/<experiment>/{args,resources,test_metrics}.json and gpu_usage.csv, written by
train_pipeline.py and the slurm/train_*.sh jobs. Writes results/summary.csv.

    python scripts/summarize_runs.py [--results_dir results]
"""
import argparse
import csv
import glob
import json
import os
import statistics

CLASSES = ['AF', 'SR', 'MI']


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


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


def summarize(run_dir):
    args = load_json(os.path.join(run_dir, 'args.json'))
    resources = load_json(os.path.join(run_dir, 'resources.json'))
    if args is None or resources is None:
        return None
    seconds = resources['seconds']
    gpu_util, gpu_memory_used = gpu_usage(os.path.join(run_dir, 'gpu_usage.csv'))
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
    }
    metrics = load_json(os.path.join(run_dir, 'test_metrics.json'))
    for sex in ['female', 'male']:
        for i, name in enumerate(CLASSES):
            row[f'auc_{name}_{sex}'] = round(metrics[sex]['test_roc'][i], 3) if metrics else None
    return row


def print_table(rows, columns):
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in columns]
    print('  '.join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in rows:
        print('  '.join(str(r[c]).ljust(w) for c, w in zip(columns, widths)))


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

    print('Runtime and memory per run')
    print_table(rows, ['experiment', 'gpu', 'epochs', 'total_min', 'load_min', 'train_min', 'mean_epoch_min',
                       'peak_gpu_allocated_gb', 'peak_gpu_used_gb', 'mean_gpu_util_pct', 'peak_host_memory_gb'])
    print()

    auc_columns = [f'auc_{name}_{sex}' for sex in ['female', 'male'] for name in CLASSES]
    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['sex_ratio'], r['fold']), []).append(r)
    means = []
    for (model, ratio, fold), group in groups.items():
        mean = {'model': model, 'sex_ratio (M_F)': ratio, 'fold': fold, 'runs': len(group),
                'train_min': round(statistics.mean(r['train_min'] for r in group), 1)}
        for c in auc_columns:
            values = [r[c] for r in group if r[c] is not None]
            if not values:
                mean[c] = None
            elif len(values) == 1:
                mean[c] = f'{values[0]:.3f}'
            else:
                mean[c] = f'{statistics.mean(values):.3f}±{statistics.stdev(values):.3f}'
        means.append(mean)
    print('Test ROC AUC, mean ± SD over seeds')
    print_table(means, ['model', 'sex_ratio (M_F)', 'fold', 'runs', 'train_min'] + auc_columns)

    out = os.path.join(args.results_dir, 'summary.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
