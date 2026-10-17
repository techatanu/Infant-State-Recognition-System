"""PyTorch CNN on 64-band log-mel spectrograms.

Protocol (test split from src.data is never touched here):
  1. Stratified 5-fold CV on the training split. Each fold trains with class-weighted
     cross-entropy, augmentation on the training folds only, and early stopping on the
     validation fold's loss.
  2. Retrain on the full training split for the median best epoch from CV and save to
     models/cnn.pt. Test-set evaluation happens in src/evaluate.py.

Usage: python -m src.train_cnn [--epochs 120] [--patience 20] [--folds 5]
"""
import argparse
import json
import random
import time

import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data import encode_labels, load_clean_metadata, train_test_split_df
from src.features import fix_length, load_audio, log_mel_spectrogram
from src.model import CryCNN

PITCH_STEPS = [-2, -1, 1, 2]      # semitones
N_PITCH_VARIANTS = 2              # pre-computed per training clip (pitch shift is slow)
MAX_SHIFT_SEC = 0.5
NOISE_SNR_DB = (25, 40)
P_PITCH = P_NOISE = P_SPECAUG = 0.5
MIN_EPOCHS = 20              # val loss is noisy early on; don't stop before this
LABELS = list(range(len(config.CLASSES)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_waves(filepaths):
    return Parallel(n_jobs=-1, prefer="threads")(delayed(load_audio)(fp) for fp in filepaths)


def build_pitch_bank(waves, seed=config.SEED):
    """N_PITCH_VARIANTS pitch-shifted copies of each clip, used only in training role."""
    rng = np.random.default_rng(seed)
    steps = [rng.choice(PITCH_STEPS, N_PITCH_VARIANTS, replace=False) for _ in waves]
    jobs = [(i, s) for i, ss in enumerate(steps) for s in ss]
    shifted = Parallel(n_jobs=-1, prefer="threads")(
        delayed(librosa.effects.pitch_shift)(waves[i], sr=config.SAMPLE_RATE, n_steps=float(s))
        for i, s in jobs)
    bank = [[] for _ in waves]
    for (i, _), y in zip(jobs, shifted):
        bank[i].append(y.astype(np.float32))
    return bank


# ---------------------------------------------------------------- augmentation
def time_shift(y, rng, max_shift=int(MAX_SHIFT_SEC * config.SAMPLE_RATE)):
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s > 0:
        return np.concatenate([np.zeros(s, dtype=y.dtype), y[:-s]])
    if s < 0:
        return np.concatenate([y[-s:], np.zeros(-s, dtype=y.dtype)])
    return y


def add_noise(y, rng, snr_db=NOISE_SNR_DB):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    noise_rms = rms / 10 ** (rng.uniform(*snr_db) / 20)
    return (y + rng.normal(0, noise_rms, len(y))).astype(np.float32)


def spec_augment(spec, rng, n_freq=2, max_f=6, n_time=2, max_t=25):
    """SpecAugment frequency and time masks (masked to 0 = silence)."""
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    for _ in range(n_freq):
        f = int(rng.integers(0, max_f + 1))
        f0 = int(rng.integers(0, n_mels - f + 1))
        spec[f0:f0 + f, :] = 0
    for _ in range(n_time):
        t = int(rng.integers(0, max_t + 1))
        t0 = int(rng.integers(0, n_frames - t + 1))
        spec[:, t0:t0 + t] = 0
    return spec


class CrySpecDataset(Dataset):
    def __init__(self, waves, labels, augment=False, pitch_bank=None, seed=0):
        self.waves, self.labels = waves, np.asarray(labels)
        self.augment, self.pitch_bank = augment, pitch_bank
        self.rng = np.random.default_rng(seed)
        if not augment:   # deterministic, so compute once
            self.specs = [log_mel_spectrogram(fix_length(y)) for y in waves]

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        if not self.augment:
            return torch.from_numpy(self.specs[i][None]), self.labels[i]
        rng, y = self.rng, self.waves[i]
        if self.pitch_bank is not None and rng.random() < P_PITCH:
            y = self.pitch_bank[i][rng.integers(len(self.pitch_bank[i]))]
        y = time_shift(y, rng)
        if rng.random() < P_NOISE:
            y = add_noise(y, rng)
        y = fix_length(y, offset=int(rng.integers(0, abs(len(y) - config.CNN_N_SAMPLES) + 1)))
        spec = log_mel_spectrogram(y)
        if rng.random() < P_SPECAUG:
            spec = spec_augment(spec, rng)
        return torch.from_numpy(spec[None]), self.labels[i]


# ---------------------------------------------------------------- training
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, probs = 0.0, []
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * len(y)
        probs.append(torch.softmax(logits, dim=1).numpy())
    return total / len(loader.dataset), np.concatenate(probs)


def train_model(waves, y, pitch_bank, args, seed, val_waves=None, val_y=None, n_epochs=None):
    """With a validation set: early stopping on val loss, returns (model, best_epoch, val_probs).
    Without: trains for exactly n_epochs."""
    seed_everything(seed)
    model = CryCNN()
    weights = compute_class_weight("balanced", classes=np.array(LABELS), y=y)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    train_loader = DataLoader(
        CrySpecDataset(waves, y, augment=True, pitch_bank=pitch_bank, seed=seed),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    # warmup + cosine over the full epoch budget, so the final retrain sees the same
    # learning-rate curve as CV up to its stopping epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)

    if val_waves is None:
        for _ in range(n_epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        return model, n_epochs, None

    val_loader = DataLoader(CrySpecDataset(val_waves, val_y), batch_size=64)
    min_epochs = min(MIN_EPOCHS, args.epochs)
    best = (np.inf, 0, None, None)   # loss, epoch, state, probs
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        val_loss, probs = evaluate(model, val_loader, loss_fn)
        if epoch >= min_epochs and val_loss < best[0]:
            best = (val_loss, epoch, {k: v.clone() for k, v in model.state_dict().items()}, probs)
        if epoch % 5 == 0:
            f1 = f1_score(val_y, probs.argmax(1), labels=LABELS, average="macro", zero_division=0)
            print(f"    epoch {epoch:3d}  val_loss {val_loss:.3f}  val_macroF1 {f1:.3f}")
        if epoch >= min_epochs and epoch - best[1] >= args.patience:
            print(f"    early stop at epoch {epoch} (best {best[1]})")
            break
    model.load_state_dict(best[2])
    return model, best[1], best[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    train_df, _ = train_test_split_df(load_clean_metadata())
    y = encode_labels(train_df["label"])
    t0 = time.time()
    waves = load_waves(train_df["filepath"])
    pitch_bank = build_pitch_bank(waves)
    print(f"Loaded {len(waves)} training clips + pitch bank in {time.time() - t0:.0f}s")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    oof = np.zeros((len(y), len(LABELS)))
    folds = []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"Fold {k + 1}/{args.folds}  train {len(tr)}  val {len(va)}")
        _, best_epoch, probs = train_model(
            [waves[i] for i in tr], y[tr], [pitch_bank[i] for i in tr], args,
            seed=config.SEED + k, val_waves=[waves[i] for i in va], val_y=y[va])
        oof[va] = probs
        pred = probs.argmax(1)
        folds.append({
            "best_epoch": best_epoch,
            "accuracy": float(accuracy_score(y[va], pred)),
            "macro_f1": float(f1_score(y[va], pred, labels=LABELS, average="macro",
                                       zero_division=0)),
        })
        print(f"  fold {k + 1}: best epoch {best_epoch}  acc {folds[-1]['accuracy']:.3f}  "
              f"macro-F1 {folds[-1]['macro_f1']:.3f}")

    f1s = np.array([f["macro_f1"] for f in folds])
    oof_pred = oof.argmax(1)
    print(f"\nCV macro-F1: {f1s.mean():.3f} ± {f1s.std():.3f}   "
          f"(pooled out-of-fold: {f1_score(y, oof_pred, average='macro', zero_division=0):.3f})")

    n_final = int(np.median([f["best_epoch"] for f in folds]))
    print(f"Retraining on all {len(y)} training clips for {n_final} epochs...")
    model, _, _ = train_model(waves, y, pitch_bank, args, seed=config.SEED, n_epochs=n_final)

    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": config.CLASSES,
                "epochs": n_final, "n_mels": config.N_MELS_CNN,
                "duration_sec": config.CNN_DURATION_SEC},
               config.MODELS_DIR / "cnn.pt")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "model": "cnn",
        "n_train": int(len(y)),
        "cv_macro_f1_mean": float(f1s.mean()), "cv_macro_f1_std": float(f1s.std()),
        "cv_macro_f1_folds": f1s.tolist(),
        "cv_oof_macro_f1": float(f1_score(y, oof_pred, average="macro", zero_division=0)),
        "cv_oof_per_class_f1": dict(zip(config.CLASSES, map(float, f1_score(
            y, oof_pred, labels=LABELS, average=None, zero_division=0)))),
        "folds": folds, "final_epochs": n_final, "args": vars(args),
    }
    with open(config.REPORTS_DIR / "cnn_cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {config.MODELS_DIR / 'cnn.pt'} and metrics to reports/ "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
