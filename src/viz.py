"""One plotting style for every figure.

Classes are told apart by position (one row or panel per class), never by colour: data are drawn
in one blue, and the reference (all clips, overall median) in grey.
"""
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .audio import CLASSES, SR
from .descriptors import HOP, power_spectrum

BLUE = "#2a78d6"
GREY = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
# sequential, one hue: quiet = surface, loud = dark blue
SPEC_CMAP = LinearSegmentedColormap.from_list("surface_blue", [SURFACE, "#a9cbf2", BLUE, "#0b3b75"])


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, GREY]),
    })


def small_multiples(n, ncols=4, panel=(2.7, 2.0), **kwargs):
    """A grid of n panels; returns (fig, list of axes)."""
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False, constrained_layout=True, **kwargs)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def spectrogram(ax, y, sr=SR, top_db=70):
    """Log-power spectrogram 0 - sr/2 on the shared frame grid."""
    freqs, P = power_spectrum(y, sr)
    db = 10 * np.log10(P.T + 1e-12)
    db = np.maximum(db, db.max() - top_db)
    t_end = len(P) * HOP / sr
    ax.imshow(db, origin="lower", aspect="auto", cmap=SPEC_CMAP,
              extent=[0, t_end, freqs[0], freqs[-1]], interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")


def strip_by_class(ax, df, value, by="label", order=CLASSES, size=None):
    """One row per class: a dot per row of `df` (jittered), a black tick at the class median and a
    grey line at the overall median. Rows are labelled with the number of dots."""
    rng = np.random.default_rng(0)
    for i, cls in enumerate(order):
        v = df.loc[df[by] == cls, value].dropna().to_numpy()
        ax.scatter(v, i + rng.uniform(-0.22, 0.22, len(v)), s=size or 12, color=BLUE,
                   alpha=0.55, linewidths=0)
        if len(v):
            ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=2, solid_capstyle="butt")
    ax.axvline(df[value].median(), color=GREY, lw=1, ls="--", zorder=0)
    counts = df.dropna(subset=[value]).groupby(by).size()
    ax.set_yticks(range(len(order)), [f"{c} ({counts.get(c, 0)})" for c in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(value)
