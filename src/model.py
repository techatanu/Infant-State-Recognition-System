"""Small 2D CNN over log-mel spectrograms, plus load/predict helpers."""
import numpy as np
import torch
from torch import nn

from src import config
from src.features import fix_length, log_mel_spectrogram


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2),
    )


class CryCNN(nn.Module):
    """PANNs-style small CNN: per-mel-band BatchNorm on the input, 4 double-conv blocks
    (16-32-64-128), mean over frequency, then max + mean over time -> dropout -> linear.
    Max over time keeps short loud cry segments from being averaged away by silence/padding."""

    def __init__(self, n_classes=len(config.CLASSES), n_mels=config.N_MELS_CNN, dropout=0.3):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):            # x: (batch, 1, n_mels, frames)
        x = self.bn0(x.transpose(1, 2)).transpose(1, 2)
        x = self.features(x).mean(dim=2)          # (batch, 128, frames')
        x = x.max(dim=2).values + x.mean(dim=2)
        return self.head(x)


def load_cnn(path=config.MODELS_DIR / "cnn.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CryCNN(n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


@torch.no_grad()
def predict_proba(model, waves):
    """Preprocessed waveforms (list of 1-D arrays) -> (n, n_classes) probabilities."""
    specs = np.stack([log_mel_spectrogram(fix_length(y)) for y in waves])[:, None]
    logits = model(torch.from_numpy(specs))
    return torch.softmax(logits, dim=1).numpy()
