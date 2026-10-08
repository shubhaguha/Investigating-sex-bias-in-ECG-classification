import numpy as np
from scipy import signal

TARGET_FS = 500
BANDPASS = (1.0, 47.0)  # Hz
FILTER_ORDER = 3


def resample_to_500(data, fs):
    """Resample a (leads, samples) array to 500 Hz."""
    if fs == float(1000):
        return signal.resample_poly(data, up=1, down=2, axis=-1)
    if fs == float(TARGET_FS):
        return data
    return signal.resample(data, int(data.shape[1] * TARGET_FS / fs), axis=1)


def bandpass_filter(data, fs=TARGET_FS, low=BANDPASS[0], high=BANDPASS[1], order=FILTER_ORDER):
    """Zero-phase Butterworth bandpass filter along the last axis of a (leads, samples) array."""
    sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=-1)


def preprocess_recording(data, fs):
    """Resample to 500 Hz and bandpass filter 1-47 Hz. Returns the filtered array and the new fs."""
    data = np.nan_to_num(np.asarray(data, dtype=np.float64))
    data = resample_to_500(data, fs)
    return bandpass_filter(data), TARGET_FS


# Target folder -> folder names it may have in the raw CinC 2021 download (lower-cased) or a preprocessed copy.
# Ordered as train_pipeline.py reads them.
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
