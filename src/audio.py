"""Paths, the class list and audio loading shared by every notebook.

All audio enters the project through `load_audio`, which returns 8 kHz mono float32 in [-1, 1].
The 44.1 kHz files contain no sound above ~4.5 kHz and have identical channels (notebook 01),
so nothing is lost by the conversion, and the recording format stops being a label shortcut.
"""
import os
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.io import wavfile
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")
DATA_DIR = Path(os.environ.get("BABYCRY_DATA_DIR", REPO.parent))
CLEAN_INDEX = REPO / "metadata" / "clean_index.csv"

CLASSES = ["belly_pain", "burping", "cold_hot", "discomfort",
           "hungry", "lonely", "scared", "tired"]
SR = 8000


def load_clean_index():
    """The 479 clips used for modelling (written by notebook 01)."""
    return pd.read_csv(CLEAN_INDEX, dtype={"age_code": str})


def to_float(y):
    """PCM samples of any WAV integer type -> float in [-1, 1]."""
    if y.dtype == np.uint8:
        return (y.astype(np.float64) - 128) / 128
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.float64) / -np.iinfo(y.dtype).min
    return y.astype(np.float64)


def load_audio(path, sr=SR):
    """Read a WAV file as mono float32 at `sr` Hz.

    Relative paths (as in clean_index.csv) are resolved against DATA_DIR.
    """
    path = Path(path)
    if not path.is_absolute():
        path = DATA_DIR / path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        file_sr, y = wavfile.read(path)
    y = to_float(y)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        y = resample_poly(y, sr // g, file_sr // g)
    return y.astype(np.float32)
