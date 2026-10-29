import numpy as np
import pytest

torch = pytest.importorskip("torch")     # notebook 07 runs on Colab, where torch is installed

from src.audio import CLASSES, SR
from src.cnn import (N_FRAMES, N_MELS, N_SAMPLES, CryCNN, LogMel, augment_wave, balanced_cross_entropy,
                     fix_length, log_mel_np, prediction_table, spec_augment, train_split, validation_split)
from src.features import hz_to_mel, mel_to_hz
from src.metrics import balanced_accuracy


def tone(freq, n=N_SAMPLES, amp=0.3):
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def test_fix_length_crops_and_pads_around_the_centre():
    y = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(fix_length(y, 4), [3, 4, 5, 6])
    padded = fix_length(y, 14)
    assert len(padded) == 14 and padded[:2].sum() == 0 and padded[-2:].sum() == 0
    np.testing.assert_array_equal(padded[2:12], y)


def test_log_mel_shape_and_tone_lands_in_its_band():
    spec = log_mel_np(tone(1000))
    assert spec.shape == (N_MELS, N_FRAMES) == (64, 698)
    centres = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MELS + 2))[1:-1]
    assert abs(centres[spec.mean(axis=1).argmax()] - 1000) < 50


def test_torch_front_end_matches_numpy_reference():
    rng = np.random.default_rng(0)
    waves = np.stack([tone(440) + 0.01 * rng.standard_normal(N_SAMPLES).astype(np.float32), tone(2500)])
    out = LogMel()(torch.as_tensor(waves)).numpy()
    for w, o in zip(waves, out):
        ref = log_mel_np(w)
        loud = ref > ref.max() - 60          # float32 vs float64 differ only far below the peak
        assert np.abs(o - ref)[loud].max() < 0.05


def test_model_output_shape_and_size():
    model = CryCNN(np.zeros(N_MELS), np.ones(N_MELS))
    logits = model(torch.as_tensor(np.stack([tone(500), tone(900), tone(1500)])))
    assert logits.shape == (3, len(CLASSES))
    n_weights = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_weights < 150_000


def test_validation_split_keeps_babies_apart():
    y = np.array([0] * 60 + [1] * 20 + [2] * 20)
    groups = np.repeat(np.arange(50), 2)
    fit, val = validation_split(y, groups, seed=0)
    assert set(groups[fit]).isdisjoint(groups[val])
    assert sorted(np.concatenate([fit, val])) == list(range(len(y)))
    assert 0.1 < len(val) / len(y) < 0.3


def test_augmentation_shapes_shift_and_noise_level():
    gen = torch.Generator().manual_seed(0)
    wave = torch.as_tensor(np.stack([tone(700), tone(1200)]))
    out = augment_wave(wave, gen, max_shift=SR, snr_db=(30.0, 30.0))
    assert out.shape == wave.shape
    # a circular shift keeps the energy; 30 dB noise adds only 0.1%
    np.testing.assert_allclose((out ** 2).mean(dim=1), (wave ** 2).mean(dim=1), rtol=0.01)
    spec = torch.randn(2, N_MELS, N_FRAMES, generator=gen) + 5
    masked = spec_augment(spec, gen, 2, 8, 2, 50)
    assert masked.shape == spec.shape and (masked == 0).any() and (masked == 0).float().mean() < 0.5
    shifted = spec_augment(spec, gen, 0, 0, 0, 0, pitch_bands=2)
    assert shifted.shape == spec.shape


def test_balanced_cross_entropy_weights_classes_equally():
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    t = torch.tensor([0, 0, 0, 1])
    ce = torch.nn.functional.cross_entropy(logits, t, reduction="none")
    assert torch.isclose(balanced_cross_entropy(logits, t), (ce[:3].mean() + ce[3]) / 2)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("variant", ["weighted_loss", "balanced_sampler"])
def test_train_split_learns_tones(variant):
    """Three classes = three pitches; a short training run must beat chance on unseen 'babies'."""
    rng = np.random.default_rng(0)
    freqs = {0: 400, 4: 1000, 7: 2200}
    t = np.array([0] * 12 + [4] * 30 + [7] * 12)
    waves = np.stack([tone(freqs[c] * rng.uniform(0.95, 1.05), amp=rng.uniform(0.1, 0.5))
                      + 0.02 * rng.standard_normal(N_SAMPLES).astype(np.float32) for c in t])
    groups = np.arange(len(t))
    test = np.arange(0, len(t), 4)
    train = np.setdiff1d(np.arange(len(t)), test)
    proba, log, history = train_split(waves, t, groups, train, test, variant, seed=0, device="cpu",
                                      channels=(8, 16), batch_size=8, max_epochs=20, patience=20)
    assert proba.shape == (len(test), len(CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)
    assert balanced_accuracy(t[test], proba.argmax(axis=1)) > 0.6      # chance: 1/3
    assert log["epochs_run"] == len(history) and 0 <= log["best_epoch"] < log["epochs_run"]
    table = prediction_table(proba, t, test, "r0", 0)
    assert list(table.columns[:5]) == ["clip", "repeat", "fold", "y_true", "y_pred"]
