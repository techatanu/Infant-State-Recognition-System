"""Audio loading and the 218-dim hand-crafted feature vector."""
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config


def load_audio(filepath, sr=config.SAMPLE_RATE):
    """Resample, trim silence (30 dB) and peak-normalise."""
    y, _ = librosa.load(filepath, sr=sr)
    return preprocess(y)


def preprocess(y):
    y, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)
    max_amp = np.max(np.abs(y)) if len(y) else 0
    if max_amp > 0:
        y = y / max_amp
    return y


def aggregate_features(feature_matrix):
    """(n_coeff, frames) -> (2 * n_coeff,): mean + std per coefficient."""
    return np.concatenate([np.mean(feature_matrix, axis=1), np.std(feature_matrix, axis=1)])


def extract_features(y, sr=config.SAMPLE_RATE):
    """
    218 features total:
    MFCC(40) + Delta-MFCC(40) + ZCR(2) + Centroid(2) + RMS(2)
    + Chroma(24) + Mel(80) + Contrast(16) + Tonnetz(12)
    """
    hop, win, nfft = config.HOP_LENGTH, config.WIN_LENGTH, config.N_FFT
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=nfft,
                                hop_length=hop, win_length=win)
    feats.append(aggregate_features(mfcc))
    feats.append(aggregate_features(librosa.feature.delta(mfcc)))
    feats.append(aggregate_features(librosa.feature.zero_crossing_rate(y)))
    feats.append(aggregate_features(
        librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=nfft, hop_length=hop)))
    feats.append(aggregate_features(librosa.feature.rms(y=y)))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=nfft, hop_length=hop,
                                         win_length=win, n_chroma=12)
    feats.append(aggregate_features(chroma))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=nfft, hop_length=hop,
                                         win_length=win, n_mels=config.N_MELS_RF)
    feats.append(aggregate_features(librosa.power_to_db(mel, ref=np.max)))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=nfft, hop_length=hop,
                                                 win_length=win, n_bands=7, fmin=100)
    feats.append(aggregate_features(contrast))

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    feats.append(aggregate_features(librosa.feature.tonnetz(chroma=chroma_cqt)))

    return np.concatenate(feats)


def feature_names():
    return ([f"mfcc_m{i}" for i in range(20)] + [f"mfcc_s{i}" for i in range(20)] +
            [f"dmfcc_m{i}" for i in range(20)] + [f"dmfcc_s{i}" for i in range(20)] +
            ["zcr_m", "zcr_s", "cent_m", "cent_s", "rms_m", "rms_s"] +
            [f"chroma_m{i}" for i in range(12)] + [f"chroma_s{i}" for i in range(12)] +
            [f"mel_m{i}" for i in range(40)] + [f"mel_s{i}" for i in range(40)] +
            [f"contrast_m{i}" for i in range(8)] + [f"contrast_s{i}" for i in range(8)] +
            [f"tonnetz_m{i}" for i in range(6)] + [f"tonnetz_s{i}" for i in range(6)])


def _features_for_file(filepath):
    try:
        return extract_features(load_audio(filepath))
    except Exception:
        return np.full(config.N_FEATURES, np.nan)


def build_feature_table(df, n_jobs=-1):
    """Feature matrix for every row of df, cached by filepath in data/cache."""
    cache = config.CACHE_DIR / "features_218.csv"
    cached = pd.read_csv(cache, index_col="filepath") if cache.exists() else None

    todo = [fp for fp in df["filepath"] if cached is None or fp not in cached.index]
    if todo:
        print(f"Extracting features for {len(todo)} clips...")
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_features_for_file)(fp) for fp in todo)
        new = pd.DataFrame(rows, columns=feature_names(), index=pd.Index(todo, name="filepath"))
        cached = new if cached is None else pd.concat([cached, new])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache)

    return cached.loc[df["filepath"]].to_numpy()
