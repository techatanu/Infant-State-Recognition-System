"""Zip the 479 clean clips, as 8 kHz mono 16-bit WAV, for notebook 07 on Google Colab.

    python scripts/make_colab_bundle.py            # writes data/colab/babycry_8k.zip (gitignored)

Every clip goes through src.audio.load_audio, exactly as for the classical models, and keeps its
relative path from metadata/clean_index.csv (e.g. hungry/<file>.wav). Upload the zip to
"My Drive/babycry/" in Google Drive. The audio never goes into git.
"""
import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import REPO, SR, load_audio, load_clean_index  # noqa: E402


def to_int16(y):
    return np.round(np.clip(y, -1.0, 32767 / 32768) * 32768).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "data" / "colab" / "babycry_8k.zip")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    index = load_clean_index()
    worst = 0.0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in index["path"]:
            y = load_audio(path)
            pcm = to_int16(y)
            worst = max(worst, float(np.abs(pcm / 32768 - y).max()))
            buf = io.BytesIO()
            wavfile.write(buf, SR, pcm)
            zf.writestr(path, buf.getvalue())
        zf.writestr("clean_index.csv", index.to_csv(index=False))

    with zipfile.ZipFile(args.out) as zf:
        wavs = [n for n in zf.namelist() if n.endswith(".wav")]
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB): {len(wavs)} clips at {SR} Hz mono, 16-bit")
    print(f"largest rounding error from 16-bit storage: {worst:.2e} (1 LSB = {1 / 32768:.2e})")
    print(f"sha256 {digest}")
    if len(wavs) != len(index):
        sys.exit(f"expected {len(index)} clips, found {len(wavs)}")


if __name__ == "__main__":
    main()
