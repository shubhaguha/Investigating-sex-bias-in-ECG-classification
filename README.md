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

## Running

```bash
pip install -r requirements.txt

# 1. Resample to 500 Hz and bandpass filter (1-47 Hz) the six databases of the PhysioNet 2021 download
python scripts/preprocess_data.py --cinc_dir /path/to/physionet.org --data_root /path/to/ecg_data

# 2. Create the 5-fold split with sex-balanced test sets and the F0-F100 training sets
python scripts/make_dataset_division.py --data_root /path/to/ecg_data

# 3. Train and evaluate one configuration (model, sex ratio male_female, test fold)
python train_pipeline.py --data_root /path/to/ecg_data --model resnet_attention --sex_ratio 50_50 --fold 0 --experiment_id resnet_attention_50_50_fold0
```

Steps 1 and 2 recreate pieces that are not part of this repository: the preprocessed data folder and `dataset_division.json`. Preprocessing follows the CinC 2021 winning entry the paper builds on; the split follows Section 2.2, Figure 2 and Tables 3-4 of the paper (see `data/division.py`). Test results are printed at the end of training (ROC AUC, pROC AUC and PR AUC per class, for the female and then the male test set); weights and training history go to `results/model_weights/`.

### On a SLURM cluster: best configuration only

`slurm/` runs the paper's best configuration, ResNet with Attention trained on the sex-balanced set (F50), over the 5 cross-validation folds. Create a Python environment with `requirements.txt` (default location `~/venvs/ecg-sexbias`, set `VENV` to change it), adjust the `#SBATCH` lines for your cluster, then from the repository root:

```bash
bash slurm/submit_best.sh
```

This submits `slurm/prepare_data.sh` (CPU: preprocessing of `/home/sguha/physionet.org` into `/home/sguha/ecg_data`, then the split) and, once it succeeds, `slurm/train_best.sh` as a 5-task GPU array, one task per test fold. Set `CINC_DIR` or `DATA_ROOT` to use other paths. Each fold's test results are printed at the end of its log in `slurm/logs/`.

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
