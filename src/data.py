"""Dataset indexing, cleaning and the fixed train/test split."""
from pathlib import Path

import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def build_metadata(dataset_path=config.DATASET_PATH):
    """One row per .wav: filename, filepath, label (sub-folder name)."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set BABYCRY_DATA_DIR in .env")

    records = []
    for label in config.CLASSES:
        class_dir = dataset_path / label
        if not class_dir.is_dir():
            continue
        # sorted so the split is reproducible across filesystems
        for wav in sorted(class_dir.glob("*.wav")):
            records.append({"filename": wav.name, "filepath": str(wav), "label": label})
    return pd.DataFrame(records)


def duration_stats(filepath, top_db=config.TRIM_TOP_DB):
    """Raw vs trimmed duration at the file's native rate (as in the notebook)."""
    y_raw, sr = librosa.load(filepath, sr=None)
    y_trim, _ = librosa.effects.trim(y_raw, top_db=top_db)
    raw_dur = len(y_raw) / sr
    trim_dur = len(y_trim) / sr
    silence_pct = (raw_dur - trim_dur) / raw_dur * 100 if raw_dur > 0 else 0
    return raw_dur, trim_dur, round(silence_pct, 1)


def clean_metadata(df, verbose=True):
    """Drop corrupt clips, clips < 1 s after trimming, and clips > 80% silence."""
    keep, n_corrupt, n_short, n_silent = [], 0, 0, 0
    for fp in df["filepath"]:
        try:
            _, trim_dur, silence_pct = duration_stats(fp)
        except Exception:
            n_corrupt += 1
            keep.append(False)
            continue
        too_short = trim_dur < config.MIN_TRIMMED_SEC
        too_silent = silence_pct > config.MAX_SILENCE_PCT
        n_short += too_short
        n_silent += too_silent
        keep.append(not (too_short or too_silent))

    df_clean = df[keep].reset_index(drop=True)
    if verbose:
        print(f"Corrupt: {n_corrupt}  Too short (<{config.MIN_TRIMMED_SEC}s): {n_short}  "
              f"Heavy silence (>{config.MAX_SILENCE_PCT}%): {n_silent}")
        print(f"Original: {len(df)}  ->  Clean: {len(df_clean)}")
    return df_clean


def load_clean_metadata(dataset_path=config.DATASET_PATH, verbose=True):
    """Cleaned index, cached to data/cache so the duration pass runs once."""
    cache = config.CACHE_DIR / "clean_metadata.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if Path(df["filepath"].iloc[0]).parent.parent == Path(dataset_path):
            return df
    df = clean_metadata(build_metadata(dataset_path), verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def encode_labels(labels):
    return pd.Series(labels).map({c: i for i, c in enumerate(config.CLASSES)}).to_numpy()


def train_test_split_df(df):
    """Stratified 80/20 split shared by every model."""
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
