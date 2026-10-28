import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.audio import CLASSES
from src.splits import inner_cv, make_folds, oof_predict, outer_splits, repeats


@pytest.fixture(scope="module")
def index():
    """Synthetic index shaped like the real one: many-clip babies, rare classes, mixed-label babies."""
    rng = np.random.default_rng(0)
    rows = []
    for b in range(120):
        label = CLASSES[4] if b < 80 else CLASSES[b % 8]
        for _ in range(rng.integers(1, 6)):
            rows.append({"path": f"{label}/{b}-{len(rows)}.wav", "label": label, "source": f"baby{b}"})
    rows.append({"path": "tired/x.wav", "label": "tired", "source": "baby0"})   # baby with two labels
    return pd.DataFrame(rows)


def test_no_baby_on_both_sides_and_every_clip_tested_once(index):
    folds = make_folds(index)
    assert repeats(folds) == ["r0", "r1", "r2", "r3", "r4"]
    for r in repeats(folds):
        tested = []
        for _, train, test in outer_splits(folds, r):
            assert not set(index["source"].iloc[train]) & set(index["source"].iloc[test])
            tested.extend(test)
        assert sorted(tested) == list(range(len(index)))


def test_folds_are_reproducible_and_repeats_differ(index):
    a, b = make_folds(index), make_folds(index)
    pd.testing.assert_frame_equal(a, b)
    assert (a["r0"] != a["r1"]).any()


def test_clip_level_folds_do_leak_babies(index):
    folds = make_folds(index, grouped=False)
    leaks = [bool(set(index["source"].iloc[tr]) & set(index["source"].iloc[te]))
             for _, tr, te in outer_splits(folds, "r0")]
    assert any(leaks)


@pytest.mark.filterwarnings("ignore:The least populated class")
def test_inner_cv_keeps_babies_apart(index):
    train = np.arange(len(index))[: len(index) * 4 // 5]
    sub = index.iloc[train]
    for tr, va in inner_cv(0).split(sub, sub["label"], groups=sub["source"]):
        assert not set(sub["source"].iloc[tr]) & set(sub["source"].iloc[va])


def test_oof_predict_gives_one_row_per_clip_per_repeat(index):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(index), 3))
    folds = make_folds(index, seeds=[0, 1])
    pred = oof_predict(lambda: LogisticRegression(max_iter=500), X, index["label"], folds)
    assert len(pred) == 2 * len(index)
    assert (pred.groupby("repeat")["clip"].nunique() == len(index)).all()
    proba = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert set(pred["y_pred"]) <= set(CLASSES)


def test_saved_predictions_keep_tiny_probabilities(index, tmp_path, monkeypatch):
    import src.splits as splits
    monkeypatch.setattr(splits, "PREDICTIONS_DIR", tmp_path)
    pred = pd.DataFrame({"clip": [0, 1], "repeat": "r0", "fold": 0, "y_true": "hungry", "y_pred": "hungry"})
    for c in CLASSES:
        pred[f"p_{c}"] = [1e-9, 0.123456789]
    splits.save_predictions("tiny", pred, index)
    back = splits.load_predictions("tiny", index)
    np.testing.assert_allclose(back[f"p_{CLASSES[0]}"], [1e-9, 0.123457], rtol=1e-5)
