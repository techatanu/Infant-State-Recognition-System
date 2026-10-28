import numpy as np
import pytest

from src.audio import SR
from src.descriptors import (HOP, cry_mask, describe, n_frames, rms_db, segments,
                             spectral_centroid, yin)
from tests.synth import bursts, harmonic

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("f0", [160, 250, 400, 550, 700, 900])
def test_yin_recovers_known_pitch_within_2_percent(f0):
    est, aperiodicity = yin(harmonic(f0))
    voiced = ~np.isnan(est)

    assert voiced.mean() > 0.95
    assert np.max(np.abs(est[voiced] / f0 - 1)) < 0.02
    # well under the 0.15 voicing threshold; not ~0 because periods like 8000/700 = 11.4 samples
    # fall between integer lags
    assert np.median(aperiodicity[voiced]) < 0.1


def test_yin_calls_noise_and_silence_unvoiced():
    noise, _ = yin(0.1 * RNG.normal(size=2 * SR))
    silence, _ = yin(np.zeros(2 * SR))

    assert np.mean(~np.isnan(noise)) < 0.05
    assert np.isnan(silence).all()


def test_frame_functions_share_one_grid():
    y = harmonic(400, 2.0)
    n = n_frames(len(y))

    assert len(rms_db(y)) == n
    assert len(yin(y)[0]) == n
    assert len(spectral_centroid(y)) == n


def test_segments_count_bursts_and_their_length():
    y = bursts(3)
    segs = segments(cry_mask(rms_db(y)))

    assert len(segs) == 3
    burst_s = (segs[:, 1] - segs[:, 0]) * HOP / SR
    pause_s = (segs[1:, 0] - segs[:-1, 1]) * HOP / SR
    np.testing.assert_allclose(burst_s, 0.5, atol=0.04)
    np.testing.assert_allclose(pause_s, 0.4, atol=0.04)


def test_segments_bridge_short_gaps_and_drop_short_runs():
    mask = np.zeros(100, dtype=bool)
    mask[10:30] = True
    mask[32:50] = True      # 2-frame gap: merged with the first run
    mask[70:75] = True      # 5 frames: too short
    np.testing.assert_array_equal(segments(mask), [[10, 50]])
    assert segments(np.zeros(50, dtype=bool)).shape == (0, 2)


def test_centroid_of_pure_tone():
    t = np.arange(SR) / SR
    centroid = spectral_centroid(0.5 * np.sin(2 * np.pi * 1000 * t))
    assert abs(np.median(centroid) - 1000) < 50


def test_describe_on_bursts_and_on_silence():
    d = describe(bursts(4, f0=450))
    assert d["bursts_per_s"] == pytest.approx(4 / (len(bursts(4)) / SR))
    assert d["f0_median_hz"] == pytest.approx(450, rel=0.02)
    assert d["f0_iqr_st"] < 0.2
    assert d["voiced_fraction"] > 0.9

    quiet = describe(np.zeros(7 * SR))
    assert quiet["cry_fraction"] == 0
    assert np.isnan(quiet["f0_median_hz"]) and np.isnan(quiet["centroid_hz"])


def test_cry_mask_keeps_continuous_crying_whole():
    level = np.full(500, -20.0) + RNG.normal(0, 1.5, 500)   # no pauses, small level wobble
    assert cry_mask(level).mean() > 0.99


def test_quiet_recording_with_leading_zeros_is_still_segmented():
    y = 0.003 * bursts(3)                    # peaks around -55 dBFS
    y = np.concatenate([np.zeros(SR // 2), y])
    segs = segments(cry_mask(rms_db(y)))
    assert len(segs) == 3
    assert describe(y)["f0_median_hz"] == pytest.approx(400, rel=0.02)
