import numpy as np
import pytest
import torch

from src import config
from src.features import (extract_features, feature_names, fix_length, load_audio,
                          log_mel_spectrogram)
from src.model import CryCNN


def test_feature_vector_has_218_dims(synthetic_wave):
    feats = extract_features(synthetic_wave)
    assert feats.shape == (config.N_FEATURES,) == (218,)
    assert len(feature_names()) == 218


def test_features_have_no_nans(synthetic_wave):
    assert np.isfinite(extract_features(synthetic_wave)).all()


def test_features_from_wav_file(synthetic_dataset):
    y = load_audio(synthetic_dataset / "hungry" / "hungry_0.wav")
    assert np.max(np.abs(y)) == pytest.approx(1.0)   # peak-normalised
    feats = extract_features(y)
    assert feats.shape == (218,) and np.isfinite(feats).all()


@pytest.mark.parametrize("seconds", [1.0, 7.0, 9.5])
def test_spectrogram_shape_is_fixed(seconds):
    y = np.random.default_rng(0).normal(size=int(config.SAMPLE_RATE * seconds))
    spec = log_mel_spectrogram(fix_length(y.astype(np.float32)))
    assert spec.shape == (config.N_MELS_CNN, config.CNN_N_FRAMES) == (64, 603)


def test_spectrogram_has_no_nans_and_is_scaled(synthetic_wave):
    spec = log_mel_spectrogram(fix_length(synthetic_wave))   # 2 s clip -> zero padded
    assert np.isfinite(spec).all()
    assert spec.min() >= 0.0 and spec.max() <= 1.0


def test_fix_length_pads_and_crops():
    short, long = np.ones(100, np.float32), np.arange(config.CNN_N_SAMPLES + 50, dtype=np.float32)
    assert len(fix_length(short)) == config.CNN_N_SAMPLES
    assert len(fix_length(long)) == config.CNN_N_SAMPLES
    assert fix_length(long, offset=0)[0] == 0 and fix_length(long, offset=50)[0] == 50


def test_cnn_outputs_one_logit_per_class(synthetic_wave):
    spec = log_mel_spectrogram(fix_length(synthetic_wave))
    x = torch.from_numpy(np.stack([spec, spec]))[:, None]
    model = CryCNN().eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, len(config.CLASSES))
    assert torch.isfinite(logits).all()
