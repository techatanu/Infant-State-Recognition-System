"""Clip-level feature vectors for the classical models (notebook 03 explains each step).

Groups, all computed over cry frames only so pauses don't dilute them:
  A  log-mel band statistics        B  MFCC 1-20 with deltas      C  spectral shape
  D  pitch and voicing              E  rhythm                     F  energy dynamics
Absolute loudness is left out everywhere: it depends on the phone, not the baby.
"""
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import butter, sosfiltfilt

from .audio import REPO, SR, load_audio, load_clean_index
from .descriptors import (HOP, N_FFT, cry_mask, power_spectrum, rms_db, segment_mask,
                          segments, yin)

N_MELS = 40
N_MFCC = 20
BAND = (150.0, 3800.0)        # default band limit: drops the edges where the two formats differ
MIN_FRAMES = 10               # fewer cry frames than this -> use every frame of the clip
CACHE_DIR = REPO / "data" / "features"   # gitignored: derived from the audio
GROUPS = {"A": "log-mel bands", "B": "MFCC", "C": "spectral shape",
          "D": "pitch and voicing", "E": "rhythm", "F": "energy dynamics"}


def hz_to_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=0.0, fmax=None):
    """Triangular filters equally spaced on the mel scale, shape (n_mels, n_fft // 2 + 1), peak 1."""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax or sr / 2), n_mels + 2))
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))


def bandlimit(y, band=BAND, sr=SR):
    """Zero-phase band-pass (8th-order Butterworth, applied forwards and backwards)."""
    sos = butter(8, band, "bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(y, dtype=np.float64))


def deltas(x, width=2):
    """Regression deltas along time (axis 0), edges padded by repetition."""
    pad = np.pad(x, ((width, width), (0, 0)), mode="edge")
    n = len(x)
    num = sum(k * (pad[width + k:width + k + n] - pad[width - k:width - k + n]) for k in range(1, width + 1))
    return num / (2 * sum(k * k for k in range(1, width + 1)))


def log_mel(P, fb):
    return 10 * np.log10(P @ fb.T + 1e-10)


def _stats(prefix, x, names):
    """Mean and std of each column of x."""
    out = {}
    for j, name in enumerate(names):
        out[f"{prefix}{name}_mean"] = x[:, j].mean()
        out[f"{prefix}{name}_std"] = x[:, j].std()
    return out


def _pct(x, q):
    return np.percentile(x, q) if len(x) else np.nan


def extract(y, sr=SR, band=BAND):
    """One feature vector (dict, fixed order) for one clip. band=None keeps the full 0-4 kHz."""
    y = np.asarray(y, dtype=np.float64)
    if band is not None:
        y = bandlimit(y, band, sr)
    level = rms_db(y)
    segs = segments(cry_mask(level))
    in_cry = segment_mask(segs, len(level))
    use = in_cry if in_cry.sum() >= MIN_FRAMES else np.ones(len(level), dtype=bool)
    freqs, P = power_spectrum(y, sr)
    lo, hi = band if band is not None else (0.0, sr / 2)
    keep = (freqs >= lo) & (freqs <= hi)
    f = {}

    # A: log-mel band statistics; band means are relative to the clip's overall mean (no loudness)
    mel = log_mel(P, mel_filterbank(sr=sr, fmin=lo, fmax=hi))
    m = mel[use]
    f.update(_stats("A_mel", m - m.mean(), [f"{b:02d}" for b in range(N_MELS)]))

    # B: MFCC 1-20 (c0 is loudness and is dropped), deltas over the whole clip, stats on cry frames
    mfcc = dct(mel, type=2, norm="ortho", axis=1)[:, 1:N_MFCC + 1]
    d1 = deltas(mfcc)
    d2 = deltas(d1)
    names = [f"{c:02d}" for c in range(1, N_MFCC + 1)]
    f.update(_stats("B_mfcc", mfcc[use], names))
    f.update(_stats("B_d_mfcc", d1[use], names))
    f.update(_stats("B_dd_mfcc", d2[use], names))

    # C: spectral shape inside the band
    Pb, fb_ = P[:, keep] + 1e-12, freqs[keep]
    p = Pb / Pb.sum(axis=1, keepdims=True)
    centroid = p @ fb_
    bandwidth = np.sqrt(np.sum(p * (fb_ - centroid[:, None]) ** 2, axis=1))
    rolloff = fb_[np.argmax(np.cumsum(p, axis=1) >= 0.85, axis=1)]
    flatness = np.exp(np.mean(np.log(Pb), axis=1)) / np.mean(Pb, axis=1)
    flux = np.r_[0, np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))]
    shape = np.column_stack([centroid, bandwidth, rolloff, flatness, flux])[use]
    f.update(_stats("C_", shape, ["centroid_hz", "bandwidth_hz", "rolloff_hz", "flatness", "flux"]))

    # D: pitch and voicing over cry frames
    f0, aperiodicity = yin(y, sr)
    voiced = use & ~np.isnan(f0)
    v = f0[voiced]
    ok = len(v) >= MIN_FRAMES
    st = 12 * np.log2(v / np.median(v)) if ok else np.array([])
    both = voiced[1:] & voiced[:-1]
    jumps = np.abs(12 * np.log2(f0[1:][both] / f0[:-1][both]))
    f["D_f0_median_hz"] = np.median(v) if ok else np.nan
    f["D_f0_iqr_st"] = _pct(st, 75) - _pct(st, 25) if ok else np.nan
    f["D_f0_range_st"] = _pct(st, 95) - _pct(st, 5) if ok else np.nan
    f["D_f0_change_st"] = np.median(jumps) if len(jumps) >= MIN_FRAMES else np.nan
    f["D_voiced_fraction"] = voiced.sum() / use.sum()
    f["D_aperiodicity"] = np.median(aperiodicity[use])

    # E: rhythm of bursts and pauses
    burst_s = (segs[:, 1] - segs[:, 0]) * HOP / sr
    pause_s = (segs[1:, 0] - segs[:-1, 1]) * HOP / sr
    f["E_cry_fraction"] = in_cry.mean()
    f["E_bursts_per_s"] = len(segs) / (len(y) / sr)
    f["E_burst_median_s"] = np.median(burst_s) if len(burst_s) else np.nan
    f["E_burst_max_s"] = burst_s.max() if len(burst_s) else np.nan
    f["E_pause_median_s"] = np.median(pause_s) if len(pause_s) else np.nan

    # F: energy dynamics, all relative (dB differences)
    lv = level[use]
    f["F_level_std_db"] = lv.std()
    f["F_level_range_db"] = _pct(lv, 95) - _pct(lv, 10)
    f["F_level_change_db"] = np.median(np.abs(np.diff(level)[use[1:] & use[:-1]])) if use.sum() > 1 else np.nan
    f["F_cry_contrast_db"] = _pct(lv, 95) - _pct(level[level > -90], 10)
    return f


def feature_table(band=BAND, index=None, cache=True):
    """Features for every clip in the clean index (rows aligned with it), cached under data/."""
    index = load_clean_index() if index is None else index
    name = "full" if band is None else f"band_{band[0]:g}_{band[1]:g}"
    path = CACHE_DIR / f"{name}.csv"
    if cache and path.exists():
        table = pd.read_csv(path)
        if table["path"].tolist() == index["path"].tolist():
            return table.drop(columns="path")
    table = pd.DataFrame([extract(load_audio(p), band=band) for p in index["path"]])
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.assign(path=index["path"].to_numpy()).to_csv(path, index=False)
    return table


def group_of(column):
    return column.split("_", 1)[0]
