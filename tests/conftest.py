"""Synthetic audio fixtures, so tests (and CI) never need the real dataset."""
import numpy as np
import pytest
import soundfile as sf

from src import config

N_PER_CLASS = 6   # 48 clips -> stratified 80/20 leaves >= 1 test clip per class


def synth_cry(seed, seconds=2.0, sr=config.SAMPLE_RATE):
    """Cry-like tone: harmonics on a wobbling 300-600 Hz pitch, smooth envelope, faint noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    f0 = rng.uniform(300, 600) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 5))
    envelope = np.sin(np.pi * t / seconds)
    y = 0.5 * envelope * y / np.max(np.abs(y)) + 0.005 * rng.normal(size=len(t))
    return y.astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_wave():
    return synth_cry(seed=0)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Folder per class with N_PER_CLASS clips, plus one 0.5 s clip that cleaning must drop."""
    root = tmp_path_factory.mktemp("dataset")
    for c_idx, label in enumerate(config.CLASSES):
        (root / label).mkdir()
        for j in range(N_PER_CLASS):
            sf.write(root / label / f"{label}_{j}.wav", synth_cry(100 * c_idx + j),
                     config.SAMPLE_RATE)
    sf.write(root / "hungry" / "too_short.wav", synth_cry(999, seconds=0.5), config.SAMPLE_RATE)
    return root
