import numpy as np
import pandas as pd
import pytest

from src.audio import CLASSES
from src.metrics import balanced_accuracy, encode
from src.models import MENU, TAUS, VARIANTS, class_weights, corrected, nested_oof, tune
from src.splits import make_folds


@pytest.fixture(scope="module")
def data():
    """Synthetic clips: class shifts feature 0, babies add their own offset, hungry dominates."""
    rng = np.random.default_rng(0)
    rows, X = [], []
    for b in range(90):
        label = "hungry" if b < 50 else CLASSES[b % 8]
        offset = rng.normal(0, 0.3, 4)
        for _ in range(rng.integers(1, 4)):
            x = offset + rng.normal(0, 0.3, 4)
            x[0] += 2 * CLASSES.index(label)
            rows.append({"path": f"{label}/{len(rows)}.wav", "label": label, "source": f"b{b}"})
            X.append(x)
    return pd.DataFrame(rows), np.array(X)


def test_class_weights():
    y = np.array(["hungry"] * 90 + ["tired"] * 10)
    assert class_weights(y, None) is None
    bal, sq = class_weights(y, "balanced"), class_weights(y, "sqrt")
    assert bal["tired"] / bal["hungry"] == pytest.approx(9)
    assert sq["tired"] / sq["hungry"] == pytest.approx(3)
    for w in (bal, sq):          # mean sample weight 1
        assert (90 * w["hungry"] + 10 * w["tired"]) / 100 == pytest.approx(1)


def test_prior_correction_rule():
    proba = np.array([[0.6, 0.4], [0.9, 0.1]])
    prior = np.array([0.8, 0.2])
    np.testing.assert_array_equal(corrected(proba, prior, 0.0), [0, 0])
    # tau = 1: compare p / prior -> 0.75 vs 2.0 and 1.125 vs 0.5
    np.testing.assert_array_equal(corrected(proba, prior, 1.0), [1, 0])


def test_menu_is_the_fixed_plan():
    assert list(MENU) == ["logistic_regression", "knn", "svm_rbf", "random_forest", "hist_gradient_boosting"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tune_picks_from_the_grid(data):
    index, X = data
    choice = tune("logistic_regression", X, index["label"].to_numpy(), index["source"].to_numpy())
    assert set(choice) == set(VARIANTS)
    assert choice["argmax"]["tau"] == 0
    assert choice["prior_corrected"]["tau"] in TAUS
    assert choice["prior_corrected"]["inner_ba"] >= choice["argmax"]["inner_ba"]
    assert choice["argmax"]["params"] in MENU["logistic_regression"][1]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nested_oof_covers_every_clip_and_learns_the_signal(data):
    index, X = data
    folds = make_folds(index, seeds=[0])
    pred, log = nested_oof("logistic_regression", X, index["label"], index["source"], folds, n_jobs=1)
    for v in VARIANTS:
        assert sorted(pred[v]["clip"]) == list(range(len(index)))
        assert balanced_accuracy(encode(pred[v]["y_true"]), encode(pred[v]["y_pred"])) > 0.5
    assert len(log) == 5 * len(VARIANTS)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", list(MENU))
def test_every_menu_model_fits_with_every_weighting(name, data):
    from src.models import fit
    index, X = data
    y = index["label"].to_numpy()
    for params in MENU[name][1]:
        if params.get("weight") in (None, "sqrt") or "weight" not in params:
            fit(name, params, X, y).predict_proba(X[:3])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parallel_svm_is_reproducible(data):
    # libsvm's probability calibration uses C's global rand(); parallel workers must not share it
    index, X = data
    folds = make_folds(index, seeds=[0])
    runs = [nested_oof("svm_rbf", X, index["label"], index["source"], folds, n_jobs=5)[0]["argmax"]
            for _ in range(2)]
    pd.testing.assert_frame_equal(runs[0], runs[1])
