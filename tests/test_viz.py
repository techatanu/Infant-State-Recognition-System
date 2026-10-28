import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src import viz
from src.audio import CLASSES, SR


def test_plot_helpers_draw_without_error():
    viz.set_style()
    fig, axes = viz.small_multiples(6, ncols=4)
    assert len(axes) == 6

    viz.spectrogram(axes[0], np.sin(2 * np.pi * 440 * np.arange(SR) / SR))
    assert len(axes[0].images) == 1

    df = pd.DataFrame({"label": np.repeat(CLASSES, 3), "x": np.arange(3 * len(CLASSES), dtype=float)})
    viz.strip_by_class(axes[1], df, "x")
    assert [t.get_text() for t in axes[1].get_yticklabels()][0] == "belly_pain (3)"
