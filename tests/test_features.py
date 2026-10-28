import numpy as np
import pytest

from src.audio import SR
from src.features import (BAND, N_MELS, bandlimit, deltas, extract, group_of, hz_to_mel,
                          mel_filterbank, mel_to_hz)
from tests.synth import bursts


def tone(freq, seconds=1.0, amp=0.5):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(SR * seconds)) / SR)


def test_mel_scale_round_trip_and_filterbank_shape():
    np.testing.assert_allclose(mel_to_hz(hz_to_mel([0, 150, 1000, 4000])), [0, 150, 1000, 4000])
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 129)
    assert (fb.max(axis=1) > 0.5).all()          # no empty filter, even the narrow low ones
    assert fb.max() <= 1


def test_pure_tone_peaks_in_the_right_mel_band():
    fb = mel_filterbank()
    centres = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MELS + 2))[1:-1]
    spectrum = np.abs(np.fft.rfft(tone(1000, 256 / SR) * np.hanning(256))) ** 2
    assert abs(centres[np.argmax(fb @ spectrum)] - 1000) < 60


def test_bandlimit_keeps_the_band_and_removes_the_edges():
    def gain_db(freq):
        y = tone(freq, 2.0)
        return 20 * np.log10(np.std(bandlimit(y)[2000:-2000]) / np.std(y[2000:-2000]))
    assert abs(gain_db(1000)) < 0.5
    assert gain_db(60) < -40
    assert gain_db(3950) < -20


def test_deltas_of_a_ramp_are_its_slope():
    ramp = np.arange(20.0)[:, None] * [1.0, -2.0]
    np.testing.assert_allclose(deltas(ramp)[2:-2], np.tile([1.0, -2.0], (16, 1)))


@pytest.mark.parametrize("band", [BAND, None])
def test_fixed_length_and_finite_on_a_normal_clip(band):
    f = extract(bursts(4, noise=1e-3), band=band)
    values = np.array(list(f.values()))
    assert len(f) == 2 * N_MELS + 6 * 20 + 2 * 5 + 6 + 5 + 4
    assert np.isfinite(values).all()
    assert {group_of(k) for k in f} == set("ABCDEF")


@pytest.mark.parametrize("y", [np.zeros(7 * SR), 1e-6 * np.random.default_rng(1).normal(size=7 * SR),
                               bursts(1)[: int(0.2 * SR)]], ids=["silence", "near-silent noise", "0.2 s"])
def test_edge_cases_give_same_length_and_no_inf(y):
    f = extract(y)
    values = np.array(list(f.values()))
    assert len(f) == len(extract(bursts(2)))
    assert not np.isinf(values).any()
    # only pitch, burst/pause timings and the cry-vs-background contrast may be undefined
    missing = {k for k, v in f.items() if np.isnan(v)}
    assert all(k.startswith(("D_f0", "E_", "F_cry_contrast")) for k in missing)


def test_features_ignore_absolute_loudness():
    y = bursts(3, noise=1e-3)
    loud, quiet = extract(y), extract(0.1 * y)
    for k in loud:
        assert quiet[k] == pytest.approx(loud[k], rel=1e-3, abs=1e-3), k
