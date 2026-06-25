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
