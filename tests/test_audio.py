import numpy as np
from scipy.io import wavfile

from src.audio import CLASSES, SR, load_audio, load_clean_index

CODES = {"bp": "belly_pain", "bu": "burping", "ch": "cold_hot", "dc": "discomfort",
         "hu": "hungry", "lo": "lonely", "sc": "scared", "ti": "tired"}


def tone(freq, sr, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def peak_hz(y, sr):
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(spectrum)]


def test_44k_stereo_becomes_8k_mono(tmp_path):
    y = tone(440, 44100)
    stereo = (np.column_stack([y, y]) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "s.wav", 44100, stereo)

    out = load_audio(tmp_path / "s.wav")

    assert out.dtype == np.float32 and out.ndim == 1
    assert len(out) == SR
    assert abs(peak_hz(out, SR) - 440) <= 2
    assert abs(np.abs(out[1000:-1000]).max() - 0.5) < 0.01


def test_8k_mono_int16_is_only_scaled(tmp_path):
    pcm = (tone(300, SR) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "m.wav", SR, pcm)

    out = load_audio(tmp_path / "m.wav")

    np.testing.assert_allclose(out, pcm / 32768, atol=1e-7)


def test_clean_index_has_no_generated_or_mismatched_files():
    idx = load_clean_index()
    names = idx["path"].str.split("/").str[-1]

    assert len(idx) == 479
    assert set(idx["label"]) == set(CLASSES)
    assert not names.str.startswith("generated").any()
    assert (idx["path"].str.split("/").str[0] == idx["label"]).all()
    assert (names.str.removesuffix(".wav").str.split("-").str[-1].map(CODES) == idx["label"]).all()
    assert idx["path"].is_unique
