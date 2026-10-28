"""Synthetic test signals with known answers (no dataset needed)."""
import numpy as np

from src.audio import SR


def harmonic(f0, seconds=1.0):
    """A tone with harmonics up to 3.9 kHz, amplitudes 1/k."""
    t = np.arange(int(SR * seconds)) / SR
    return sum(0.5 / k * np.sin(2 * np.pi * k * f0 * t) for k in range(1, int(3900 / f0) + 1))


def bursts(n, on_s=0.5, off_s=0.4, f0=400, noise=1e-4, seed=0):
    """n cry-like bursts separated by low noise, with 0.3 s of noise at both ends."""
    rng = np.random.default_rng(seed)
    gap = noise * rng.normal(size=int(off_s * SR))
    edge = noise * rng.normal(size=int(0.3 * SR))
    parts = [edge]
    for i in range(n):
        parts += [harmonic(f0, on_s), gap if i < n - 1 else edge]
    return np.concatenate(parts)
