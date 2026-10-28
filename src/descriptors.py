"""Interpretable descriptors of a cry: loudness, cry/pause segments, pitch, spectral centroid.

Every frame-level function uses one grid (25 ms frames every 10 ms at 8 kHz): frame i starts at
sample i * HOP, so their outputs line up index by index.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

from .audio import SR

FRAME = 200          # 25 ms
HOP = 80             # 10 ms
N_FFT = 256
FLOOR_DB = -90.0     # digital silence / dither: never cry, never voiced

F0_MIN, F0_MAX = 150.0, 1000.0   # infant cry pitch is mostly 300-600 Hz
YIN_FRAME = 400                  # 50 ms: room for two periods of the lowest pitch
# Phone recordings carry broadband noise and a steady 100-250 Hz hum, so the classic 0.1-0.15
# threshold leaves most clearly harmonic cry frames unvoiced; 0.3 and a 200 Hz high-pass were
# chosen by inspecting pitch tracks on spectrograms (notebook 02 shows them).
YIN_THRESHOLD = 0.3
YIN_HIGHPASS_HZ = 200.0


def n_frames(n_samples):
    return 1 + max(n_samples - FRAME, 0) // HOP


def frames(y, frame=FRAME, hop=HOP):
    """Overlapping frames as a (n_frames, frame) view."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    return sliding_window_view(y, frame)[::hop]


def rms_db(y):
    """Frame energy in dB relative to full scale."""
    return 10 * np.log10(np.mean(frames(y) ** 2, axis=1) + 1e-12)


def cry_mask(level_db, min_drop_db=10.0):
    """Frames loud enough to be crying.

    The threshold sits halfway between the clip's background level (10th percentile) and its
    loud level (95th percentile), but at least `min_drop_db` below the loud level so a clip
    of continuous crying is not cut up. Levels are relative, so quiet recordings work too.
    Frames under FLOOR_DB (digital silence, e.g. zero padding at the start of some files)
    never count and are left out of the percentiles.
    """
    audible = level_db > FLOOR_DB
    if not audible.any():
        return audible
    background, loud = np.percentile(level_db[audible], [10, 95])
    threshold = min(background + 0.5 * (loud - background), loud - min_drop_db)
    return audible & (level_db > threshold)


def segments(mask, min_len=10, max_gap=3):
    """Cry bursts as an (n, 2) array of [start, stop) frame indices.

    Gaps of up to `max_gap` frames (30 ms) are bridged, then bursts shorter than `min_len`
    frames (100 ms) are dropped.
    """
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts):
        keep = starts[1:] - stops[:-1] > max_gap
        starts = np.r_[starts[0], starts[1:][keep]]
        stops = np.r_[stops[:-1][keep], stops[-1]]
    long = stops - starts >= min_len
    return np.column_stack([starts[long], stops[long]]).astype(int)


def segment_mask(segs, n):
    mask = np.zeros(n, dtype=bool)
    for start, stop in segs:
        mask[start:stop] = True
    return mask


def yin(y, sr=SR, fmin=F0_MIN, fmax=F0_MAX, threshold=YIN_THRESHOLD):
    """YIN pitch tracker (de Cheveigne & Kawahara, 2002) on the shared frame grid.

    Returns (f0, aperiodicity): f0 in Hz with NaN for unvoiced frames, and the normalised
    difference at the chosen lag (0 = perfectly periodic, ~1 = noise).
    """
    y = np.asarray(y, dtype=np.float64)
    n = n_frames(len(y))
    level = rms_db(y)
    if len(y) > 30:
        y = sosfiltfilt(butter(4, YIN_HIGHPASS_HZ, "highpass", fs=sr, output="sos"), y)
    pad = (YIN_FRAME - FRAME) // 2           # centre the long frame on the short one
    fr = frames(np.pad(y, pad), YIN_FRAME, HOP)[:n]

    tau_min, tau_max = int(sr / fmax), int(np.ceil(sr / fmin))
    width = YIN_FRAME - tau_max - 1
    d = np.zeros((len(fr), tau_max + 2))
    for tau in range(1, tau_max + 2):
        d[:, tau] = np.sum((fr[:, :width] - fr[:, tau:tau + width]) ** 2, axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, d.shape[1]) / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)

    lags = cmnd[:, tau_min:tau_max + 1]
    below = lags < threshold
    voiced = below.any(axis=1) & (level[:len(fr)] > FLOOR_DB)
    # first lag under the threshold, then walk down to the bottom of that dip
    first = np.argmax(below, axis=1)
    bottom = np.ones_like(below)
    bottom[:, :-1] = lags[:, 1:] >= lags[:, :-1]
    best = np.argmax(bottom & (np.arange(lags.shape[1]) >= first[:, None]), axis=1)
    best = np.where(voiced, best, np.argmin(lags, axis=1))
    tau = best + tau_min

    rows = np.arange(len(fr))
    a, b, c = cmnd[rows, tau - 1], cmnd[rows, tau], cmnd[rows, tau + 1]
    curve = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(curve) > 1e-12, 0.5 * (a - c) / curve, 0.0)   # parabola vertex
    shift = np.clip(shift, -1, 1)
    f0 = sr / (tau + shift)
    f0 = np.where(voiced & (f0 >= fmin) & (f0 <= fmax), f0, np.nan)
    aperiodicity = np.clip(b - 0.25 * (a - c) * shift, 0, None)   # value at the vertex
    return f0, aperiodicity


def power_spectrum(y, sr=SR):
    """Power of Hann-windowed frames on the shared grid: (freqs, P of shape (n_frames, N_FFT//2 + 1))."""
    P = np.abs(np.fft.rfft(frames(y) * np.hanning(FRAME), n=N_FFT, axis=1)) ** 2
    return np.fft.rfftfreq(N_FFT, 1 / sr), P


def spectral_centroid(y, sr=SR):
    freqs, P = power_spectrum(y, sr)
    return (P @ freqs) / np.maximum(P.sum(axis=1), 1e-20)


def describe(y, sr=SR):
    """Clip-level descriptors for the EDA. Pitch and centroid use cry frames only."""
    level = rms_db(y)
    segs = segments(cry_mask(level))
    in_cry = segment_mask(segs, len(level))
    f0, _ = yin(y, sr)
    pitch = f0[in_cry & ~np.isnan(f0)]
    burst_s = (segs[:, 1] - segs[:, 0]) * HOP / sr
    pause_s = (segs[1:, 0] - segs[:-1, 1]) * HOP / sr
    has_pitch = len(pitch) >= 10
    return {
        "cry_fraction": in_cry.mean(),
        "bursts_per_s": len(segs) / (len(y) / sr),
        "burst_s": np.median(burst_s) if len(burst_s) else np.nan,
        "pause_s": np.median(pause_s) if len(pause_s) else np.nan,
        "voiced_fraction": np.mean(~np.isnan(f0[in_cry])) if in_cry.any() else np.nan,
        "f0_median_hz": np.median(pitch) if has_pitch else np.nan,
        "f0_iqr_st": 12 * np.log2(np.percentile(pitch, 75) / np.percentile(pitch, 25)) if has_pitch else np.nan,
        "centroid_hz": np.median(spectral_centroid(y, sr)[in_cry]) if in_cry.any() else np.nan,
    }
