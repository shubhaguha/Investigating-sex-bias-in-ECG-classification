"""Offline preprocessing that produces the PhysioNet2021_preprocessed/ folders train_pipeline.py reads.

Resampling and filtering are taken from the CinC 2021 winning entry (ISIBrnoAIMT, team_code.py,
dataset.__getitem__), which the paper cites for its preprocessing: resample to 500 Hz, then a
3rd-order Butterworth bandpass of 1-47 Hz applied with zero-phase filtering (filtfilt).
Random segmentation to 4096 samples and z-score normalisation happen at load time in data_loader.py.
"""
import numpy as np
from scipy import signal

# 3rd-order Butterworth bandpass, 1-47 Hz at 500 Hz (cut-offs normalised to the 250 Hz Nyquist frequency)
B, A = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')


def preprocess_recording(data, fs):
    """Resample to 500 Hz and bandpass filter 1-47 Hz. Returns the filtered array and the new fs."""
    data = np.nan_to_num(data)

    # resample to 500hz
    if fs == float(1000):
        data = signal.resample_poly(data, up=1, down=2, axis=-1)  # to 500Hz
        fs = 500
    elif fs == float(500):
        pass
    else:
        data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
        fs = 500

    data = signal.filtfilt(B, A, data)
    return data, fs


# Target folder -> folder names it may have in the raw CinC 2021 download (lower-cased) or a preprocessed copy.
# In the order train_pipeline.py reads them.
SOURCES = {
    'WFDB_PTBXL': ['ptb-xl', 'ptbxl', 'wfdb_ptbxl'],
    'WFDB_CPSC2018': ['cpsc_2018', 'cpsc2018', 'wfdb_cpsc2018'],
    'WFDB_CPSC2018_2': ['cpsc_2018_extra', 'cpsc2018_2', 'wfdb_cpsc2018_2'],
    'WFDB_Ga': ['georgia', 'ga', 'wfdb_ga'],
    'WFDB_ChapmanShaoxing': ['chapman_shaoxing', 'chapmanshaoxing', 'wfdb_chapmanshaoxing'],
    'WFDB_Ningbo': ['ningbo', 'wfdb_ningbo'],
}


def find_source_dirs(cinc_dir):
    """Map each WFDB_* target folder to the matching database directory under cinc_dir (or None)."""
    import os
    found = {k: None for k in SOURCES}
    for root, dirs, _ in os.walk(cinc_dir):
        for d in sorted(dirs):
            for target, aliases in SOURCES.items():
                if found[target] is None and d.lower() in aliases:
                    found[target] = os.path.join(root, d)
    return found
