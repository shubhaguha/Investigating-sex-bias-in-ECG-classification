# Investigating Sex Bias in ECG Classification

This repository contains the code for the paper **"Investigating sex bias in ECG classification for Atrial Fibrillation, Sinus Rhythm and Myocardial Infarction"** by Maria Galanty, Björn van der Ster, Alexander P. Vlaar, and Clara I. Sánchez, published in *Machine Learning for Biomedical Imaging (MELBA)*, 2025.

📄 [Paper](https://doi.org/10.59275/j.melba.2025-9fe7)

---

## Overview

We systematically evaluate sex bias in deep learning models for 12-lead ECG classification across three diagnostic categories: Sinus Rhythm (SR), Atrial Fibrillation (AF), and Myocardial Infarction (MI). Three model architectures are trained under five different male/female training ratios (0%–100% female) and evaluated on sex-stratified test sets.

---

## Models

- **CNN** — Four-block convolutional neural network
- **xResNet101** — Deep residual network with 101 layers
- **ResNet with Attention** — PhysioNet Challenge 2021 winning architecture; residual CNN with multi-head attention mechanism

---

## Dataset

Data is sourced from the [PhysioNet/Computing in Cardiology Challenge 2021](https://physionet.org/content/challenge2021/1.0.3/) and includes:

- China 12-Lead ECG Challenge Database (+ Extra)
- PTB-XL
- Georgia 12-Lead ECG Dataset
- Chapman University Dataset
- Shaoxing People's Hospital / Ningbo First Hospital Databases

All signals are resampled to 500 Hz, bandpass filtered (1–47 Hz, 3rd-order Butterworth), segmented to 4096 points, and z-score normalised.

---

## Experimental Setup

Training sets are constructed at five sex ratios:

| Config | Female | Male |
|--------|--------|------|
| F0 | 0% | 100% |
| F25 | 25% | 75% |
| F50 | 50% | 50% |
| F75 | 75% | 25% |
| F100 | 100% | 0% |

Evaluation uses 5-fold cross-validation with sex-stratified test sets and reports ROC AUC, partial ROC AUC (FPR ≤ 0.2), and PR AUC.


---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing

Download the [PhysioNet 2021 challenge data](https://physionet.org/content/challenge-2021/1.0.3/), then run:

```bash
python scripts/preprocess_data.py --cinc_dir ~/data/CinC-2021
```

This finds the six databases in the download, resamples every recording to 500 Hz, applies a 1–47 Hz 3rd-order Butterworth bandpass filter (zero-phase, via `scipy.signal.sosfiltfilt`), and writes the result to `data/PhysioNet2021_preprocessed/` (the default `--data_dir` of `train_pipeline.py`). Signals stay in int16 ADC units, so the output is about the size of the six input databases. It uses all CPU cores (`--workers` to change), and records already written are skipped, so an interrupted run can be resumed. Segmenting to 4096 samples and z-score normalisation are done at load time.

## Train/test splits

```bash
python scripts/make_dataset_division.py --data_dir data/PhysioNet2021_preprocessed
```

writes `dataset_division.json` into the data directory. The original split file is not published, so this generator rebuilds it following Section 2.2, Figure 2 and Tables 3–4 of the paper (see `data/division.py`):

- Recordings of patients younger than 18 or with missing sex or age are excluded. Recordings with none of AF/SR/MI stay in as an "Other" category (all-zero targets).
- Pseudo patient IDs are formed from age, sex and source dataset; all recordings sharing one stay in the same fold.
- Age is binned into 18–39, 40–59, 60–79 and 80+, and the data is split into 5 folds stratified by age group, sex, source dataset and diagnostic category.
- In each test fold, the male and female test sets get the same number of recordings per diagnostic category, sampled stratified by age group and source.
- Training sets (F0–F100) come from the other folds. The female share holds within every diagnostic category, each sex is sampled stratified by age group and source, and the size is the same for every ratio and fold. 10% is used for validation, keeping the diagnostic distribution.
- The paper does not say how "Other" was chosen for the test sets; Table 4 has about 3.5% as many as the AF/SR/MI recordings, which is the default (`--test_other_fraction`).

The script prints its totals next to the paper's (87,663 recordings, 85,025 eligible, 749 pseudo-IDs, about 32,300 training recordings) and the per-category, age-group and source composition of every split, for comparison with Tables 3 and 4.

The file stores the list of recordings it was built for, and `train_pipeline.py` stops if the data directory no longer matches it.

## Data layout

`--data_dir` must contain one folder per database, each with WFDB `.hea`/`.mat` pairs, plus the split file:

```
<data_dir>/
├── WFDB_PTBXL/  WFDB_CPSC2018/  WFDB_CPSC2018_2/
├── WFDB_Ga/     WFDB_ChapmanShaoxing/  WFDB_Ningbo/
└── dataset_division.json
```

`dataset_division.json` maps each fold (`"0"`–`"4"`) to `male_balanced_test_idx`, `female_balanced_test_idx`, and `train_idx_<ratio>` / `val_idx_<ratio>` for every ratio in `100_0, 75_25, 50_50, 25_75, 0_100` (male_female). Indices refer to recordings in the order the folders above are listed, with files sorted by name within each folder. Pass `--division_file` to use a split file stored elsewhere.

## Running

```bash
python train_pipeline.py --data_dir /path/to/PhysioNet2021_preprocessed \
    --model xresnet101 --sex_ratio 50_50 --fold 0 --experiment_id xres_f50_fold0
```

Models: `cnn`, `resnet_attention`, `xresnet101`. Run `python train_pipeline.py -h` for all options. Outputs go to `results/<experiment_id>/` (arguments and test results for the female and male test sets), `results/model_weights/` (weights and training progress), and `runs/` (TensorBoard logs).

### Full experiment on a SLURM cluster

`slurm/train_array.sh` runs the whole grid (3 models × 5 sex ratios × 5 folds) as a 75-task job array, one GPU per task. Edit the `#SBATCH` lines (partition, account, time, memory) and the environment setup for your cluster, then from the repository root:

```bash
sbatch slurm/train_array.sh                   # all 75 runs
sbatch --array=0-24 slurm/train_array.sh      # CNN only (25-49: resnet_attention, 50-74: xresnet101)
DATA_DIR=/scratch/$USER/PhysioNet2021_preprocessed sbatch slurm/train_array.sh
```

Each task trains `<model>_<ratio>_fold<k>`, writing logs to `slurm/logs/` and results to `results/`. Finished runs are skipped, so the same command can be resubmitted after failures or timeouts. `EXTRA_ARGS="--epochs 50"` passes extra options to `train_pipeline.py`.

### Smoke test with synthetic data

```bash
python scripts/make_dummy_data.py --out dummy_data
python train_pipeline.py --data_dir dummy_data --model cnn --epochs 1 --batch_size 8
```

### Smoke test with the real data

This works on either the raw download (`training/<database>/g*/`) or the preprocessed folder. Point it at the preprocessed folder to test exactly what training will see:

```bash
python scripts/smoke_test_real_data.py --cinc_dir data/PhysioNet2021_preprocessed
```

This samples ~40 labelled recordings with known sex from each database, links them into `smoke_test/data/` in the layout above, writes a matching split file, and trains each model for one epoch. It reports which databases it found and ends with an OK/FAILED summary per model. It leaves the original data unchanged.

---

## Citation

```bibtex
@article{galanty2025sexbias,
  title={Investigating sex bias in ECG classification for Atrial Fibrillation, Sinus Rhythm and Myocardial Infarction},
  author={Galanty, Maria and van der Ster, Björn and Vlaar, Alexander P. and Sánchez, Clara I.},
  journal={Machine Learning for Biomedical Imaging},
  volume={3},
  year={2025},
  doi={10.59275/j.melba.2025-9fe7}
}
```

---

## Acknowledgements

This work was supported by the University of Amsterdam Research Priority Area Artificial Intelligence for Health Decision-making.
