"""Log-mel input, the small CNN and its training loop for notebook 07 (runs on Google Colab).

Input: 64 log-mel bands over 0-4 kHz from 25 ms Hann frames every 10 ms, for a fixed 7 s (698
frames). The FFT is zero-padded to 512 points because with 256 the lowest of 64 bands get no full
FFT bin. The front end exists twice: `log_mel_np` in numpy is the reference, and `LogMel` in torch
runs on the GPU so augmentation can act on the waveform. A test checks that the two agree.

Training on one outer training part (`train_split`):
  - a grouped, stratified validation part (about 1/5 of the babies) is carved out of it;
  - the network trains on the rest with AdamW, with one of two imbalance treatments:
      "weighted_loss": cross-entropy with square-root inverse-frequency class weights,
      "balanced_sampler": plain cross-entropy, batches drawn with inverse-frequency weights;
  - training batches only are augmented: random circular time shift, low-level noise, SpecAugment
    (plus an optional shift along the mel axis, the pitch-shift ablation);
  - after every epoch the validation part is scored by class-balanced cross-entropy (every class
    present counts equally, like balanced accuracy, but smooth). The best epoch's weights are kept
    and training stops after `patience` epochs without improvement;
  - those weights predict the outer test fold, which plays no part in any choice.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from .audio import CLASSES, SR
from .features import mel_filterbank
from .models import class_weights
from .splits import PROBA_COLS

N_MELS = 64
WIN = 200                 # 25 ms
HOP = 80                  # 10 ms
N_FFT = 512
SECONDS = 7.0
N_SAMPLES = int(SECONDS * SR)
N_FRAMES = 1 + (N_SAMPLES - WIN) // HOP
VARIANTS = ["weighted_loss", "balanced_sampler"]

DEFAULTS = {
    "channels": (16, 32, 64, 128), "dropout": 0.3,
    "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-2, "max_epochs": 80, "patience": 15,
    "max_shift_s": 1.0, "snr_db": (20.0, 40.0),
    "freq_masks": 2, "freq_mask_bands": 8, "time_masks": 2, "time_mask_frames": 50,
    "pitch_shift_bands": 0,      # > 0 only in the pitch-shift ablation
}


def fix_length(y, n=N_SAMPLES):
    """Centre-crop or zero-pad a waveform to n samples."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def _filterbank():
    return mel_filterbank(N_MELS, N_FFT, SR, 0.0, SR / 2)


def log_mel_np(y):
    """Reference front end: (N_MELS, n_frames) log-mel power in dB."""
    y = np.asarray(y, dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP]
    P = np.abs(np.fft.rfft(frames * np.hanning(WIN), n=N_FFT, axis=1)) ** 2
    return (10 * np.log10(P @ _filterbank().T + 1e-10)).T


class LogMel(nn.Module):
    """Torch version of log_mel_np for a batch of waveforms (batch, samples) -> (batch, N_MELS, frames)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.tensor(np.hanning(WIN), dtype=torch.float32))
        self.register_buffer("fb", torch.tensor(_filterbank(), dtype=torch.float32))

    def forward(self, wave):
        frames = wave.unfold(-1, WIN, HOP) * self.window
        P = torch.fft.rfft(frames, n=N_FFT).abs() ** 2
        return (10 * torch.log10(P @ self.fb.T + 1e-10)).transpose(1, 2)


class CryCNN(nn.Module):
    """Waveform -> log-mel -> per-band standardisation -> conv blocks -> pool over time -> linear.

    One 3x3 convolution, batch norm, ReLU and 2x2 max pooling per block; about 100k weights with
    the default channels, small enough for ~380 training clips. Pooling (mean and max) is over time
    only: pooling over frequency as well would discard where in the spectrum the energy is, i.e.
    pitch, and the network then failed to separate even pure tones in testing.
    """

    def __init__(self, mean, std, n_classes=len(CLASSES), channels=DEFAULTS["channels"],
                 dropout=DEFAULTS["dropout"]):
        super().__init__()
        self.front = LogMel()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1))
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c_in = c
        self.blocks = nn.Sequential(*blocks)
        n_bands = N_MELS // 2 ** len(channels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in * n_bands, n_classes))

    def spectrogram(self, wave):
        return (self.front(wave) - self.mean) / self.std

    def classify(self, spec):
        x = self.blocks(spec.unsqueeze(1))                        # (batch, channels, bands, frames)
        x = torch.cat([x.mean(dim=3), x.amax(dim=3)], dim=1)       # pool over time only
        return self.head(x.flatten(1))

    def forward(self, wave):
        return self.classify(self.spectrogram(wave))


def band_stats(waves, device, batch=64):
    """Per-band mean and std of the log-mel over a set of (unaugmented) training waveforms."""
    front = LogMel().to(device)
    s, s2, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(waves), batch):
            m = front(torch.as_tensor(waves[i:i + batch], device=device)).double()
            s, s2, n = s + m.sum(dim=(0, 2)), s2 + (m ** 2).sum(dim=(0, 2)), n + m.shape[0] * m.shape[2]
    mean = s / n
    return mean.float().cpu().numpy(), (s2 / n - mean ** 2).clamp_min(1e-6).sqrt().float().cpu().numpy()


def augment_wave(wave, gen, max_shift, snr_db):
    """Random circular shift of up to +-max_shift samples, then Gaussian noise at a random SNR."""
    b, n = wave.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (b,), generator=gen, device=wave.device)
    idx = (torch.arange(n, device=wave.device)[None, :] - shifts[:, None]) % n
    wave = torch.gather(wave, 1, idx)
    lo, hi = snr_db
    snr = lo + (hi - lo) * torch.rand(b, 1, generator=gen, device=wave.device)
    rms = wave.pow(2).mean(dim=1, keepdim=True).sqrt()
    return wave + torch.randn(wave.shape, generator=gen, device=wave.device) * rms * 10 ** (-snr / 20)


def spec_augment(spec, gen, freq_masks, freq_width, time_masks, time_width, pitch_bands=0):
    """SpecAugment on standardised log-mel (masked cells set to 0, the band mean); optional mel-axis shift."""
    spec = spec.clone()
    b, f, t = spec.shape
    dev = spec.device
    for n_masks, width, size, axis in [(freq_masks, freq_width, f, 1), (time_masks, time_width, t, 2)]:
        for _ in range(n_masks):
            w = torch.randint(0, width + 1, (b, 1), generator=gen, device=dev)
            start = (torch.rand(b, 1, generator=gen, device=dev) * (size - w + 1)).long()
            pos = torch.arange(size, device=dev)[None, :]
            mask = (pos >= start) & (pos < start + w)
            spec = spec.masked_fill(mask[:, :, None] if axis == 1 else mask[:, None, :], 0.0)
    if pitch_bands:
        k = torch.randint(-pitch_bands, pitch_bands + 1, (b,), generator=gen, device=dev)
        src = (torch.arange(f, device=dev)[None, :] - k[:, None]).clamp(0, f - 1)     # edges repeat
        spec = torch.gather(spec, 1, src[:, :, None].expand(b, f, t))
    return spec


def validation_split(y, groups, seed):
    """Positions (within a training part) of a grouped, stratified ~1/5 validation part, and the rest."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(cv.split(np.zeros(len(y)), y, groups=groups))
    return fit_idx, val_idx


def balanced_cross_entropy(logits, t):
    """Mean over the classes present in t of each class's mean cross-entropy."""
    ce = nn.functional.cross_entropy(logits, t, reduction="none")
    present = torch.unique(t)
    return torch.stack([ce[t == k].mean() for k in present]).mean()


def _loss_weights(y_codes):
    w = class_weights(y_codes, "sqrt")
    return torch.tensor([w.get(k, 0.0) for k in range(len(CLASSES))], dtype=torch.float32)


def _sample_weights(y_codes):
    counts = np.bincount(y_codes, minlength=len(CLASSES))
    return torch.tensor(1.0 / counts[y_codes], dtype=torch.double)


def predict(model, waves, device, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(waves), batch):
            out.append(torch.softmax(model(torch.as_tensor(waves[i:i + batch], device=device)), dim=1).cpu())
    return torch.cat(out).numpy()


def train_split(waves, t, groups, train, test, variant, seed, device, **overrides):
    """Train on one outer training part and predict its test fold.

    waves: (n_clips, N_SAMPLES) float32; t: integer class codes; train/test: positions.
    Returns (test probabilities in CLASSES order, log dict).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    cfg = {**DEFAULTS, **overrides}
    start = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)

    fit_rel, val_rel = validation_split(t[train], groups[train], seed)
    fit, val = train[fit_rel], train[val_rel]
    mean, std = band_stats(waves[fit], device)
    model = CryCNN(mean, std, channels=cfg["channels"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_w = _loss_weights(t[fit]).to(device) if variant == "weighted_loss" else None
    sample_w = _sample_weights(t[fit]) if variant == "balanced_sampler" else None

    X_fit = torch.as_tensor(waves[fit], device=device)
    t_fit = torch.as_tensor(t[fit], device=device)
    X_val = torch.as_tensor(waves[val], device=device)
    t_val = torch.as_tensor(t[val], device=device)
    max_shift = int(cfg["max_shift_s"] * SR)

    best, best_state, best_epoch, history = np.inf, None, -1, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        if sample_w is None:
            order = rng.permutation(len(fit))
        else:
            order = torch.multinomial(sample_w, len(fit), replacement=True,
                                      generator=torch.Generator().manual_seed(seed * 1000 + epoch)).numpy()
        train_loss = 0.0
        for i in range(0, len(order), cfg["batch_size"]):
            b = torch.as_tensor(order[i:i + cfg["batch_size"]], device=device)
            if len(b) < 2:            # batch norm needs two clips
                continue
            wave = augment_wave(X_fit[b], gen, max_shift, cfg["snr_db"])
            spec = spec_augment(model.spectrogram(wave), gen, cfg["freq_masks"], cfg["freq_mask_bands"],
                                cfg["time_masks"], cfg["time_mask_frames"], cfg["pitch_shift_bands"])
            loss = nn.functional.cross_entropy(model.classify(spec), t_fit[b], weight=loss_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val_loss = balanced_cross_entropy(model(X_val), t_val).item()
        history.append({"epoch": epoch, "train_loss": train_loss / len(order), "val_balanced_ce": val_loss})
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    proba = predict(model, waves[test], device)
    log = {"variant": variant, "seed": seed, "fit_clips": len(fit), "val_clips": len(val),
           "val_babies": len(np.unique(groups[val])), "best_epoch": best_epoch, "epochs_run": len(history),
           "best_val_balanced_ce": best, "seconds": time.time() - start,
           "pitch_shift_bands": cfg["pitch_shift_bands"]}
    return proba, log, pd.DataFrame(history)


def prediction_table(proba, t, test, repeat, fold):
    """Out-of-fold rows in the same layout as splits.oof_predict (argmax decision)."""
    part = pd.DataFrame(proba, columns=PROBA_COLS)
    part.insert(0, "y_pred", np.array(CLASSES)[proba.argmax(axis=1)])
    part.insert(0, "y_true", np.array(CLASSES)[t[test]])
    part.insert(0, "fold", fold)
    part.insert(0, "repeat", repeat)
    part.insert(0, "clip", test)
    return part
