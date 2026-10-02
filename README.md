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

## Data layout

Download and preprocess the PhysioNet 2021 databases so that `--data_dir` contains one folder per database, each with WFDB `.hea`/`.mat` pairs, plus the split file:

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

### Smoke test with synthetic data

```bash
python scripts/make_dummy_data.py --out dummy_data
python train_pipeline.py --data_dir dummy_data --model cnn --epochs 1 --batch_size 8
```

### Smoke test with the real data

Once the PhysioNet 2021 data is downloaded (the raw `training/<database>/g*/` layout works as is), run:

```bash
python scripts/smoke_test_real_data.py --cinc_dir ~/data/CinC
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
